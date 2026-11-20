"""Готовит открытые снимки RDD2020 (Чехия) в формате эксперимента 1.

Кадры набора перемешаны, но один и тот же участок дороги встречается в нескольких кадрах.
Поэтому кадры объединяются в группы-«участки» по сходству (эмбеддинги YOLOv8s-cls + dHash),
и каждая группа целиком попадает в одну часть. Тестовые группы выбираются один раз до обучения
и записываются в test_groups.txt.

Запуск: python prep_czech.py <папка Czech с images/ и annotations/xmls/> <куда положить датасет>
"""
import sys, random, shutil, json, xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

SRC, DST = Path(sys.argv[1]), Path(sys.argv[2])
COS_T = float(sys.argv[3]) if len(sys.argv) > 3 else 0.08   # порог косинусного расстояния (средняя связь; подобран по картинкам)
DH_T = 30                                                     # dHash 16x16: <= 30 бит — точно одно место
SEED, TEST_SHARE, N_NEG = 0, 0.25, 200
CLASSES = ['D00', 'D10', 'D20', 'D40']

items = []
for x in sorted((SRC / 'annotations' / 'xmls').glob('*.xml')):
    r = ET.parse(x).getroot()
    W, H = float(r.findtext('size/width')), float(r.findtext('size/height'))
    boxes = []
    for o in r.iter('object'):
        c = o.findtext('name')
        if c not in CLASSES:
            continue
        b = o.find('bndbox')
        x1, y1, x2, y2 = (float(b.findtext(k)) for k in ('xmin', 'ymin', 'xmax', 'ymax'))
        if x2 > x1 and y2 > y1:
            boxes.append((CLASSES.index(c), (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H))
    items.append({'src': SRC / 'images' / (x.stem + '.jpg'), 'boxes': boxes})

# --- сходство кадров: середина кадра без неба/стекла сверху и капота снизу
def crop(p):
    im = Image.open(p).convert('RGB'); w, h = im.size
    return im.crop((0, int(h * 0.2), w, int(h * 0.82)))

def dhash(im, s=16):
    a = np.asarray(im.convert('L').resize((s + 1, s)), dtype=np.int16)
    return np.packbits(a[:, 1:] > a[:, :-1])

from ultralytics import YOLO
emb_model = YOLO(str(Path(__file__).parent / 'yolov8s-cls.pt'))
crops = [crop(it['src']) for it in items]
E = []
for i in range(0, len(crops), 64):
    E += [e.numpy() for e in emb_model.embed([np.asarray(c)[:, :, ::-1] for c in crops[i:i + 64]], imgsz=224, verbose=False)]
E = np.array(E); E /= np.linalg.norm(E, axis=1, keepdims=True)
Hs = np.array([dhash(c) for c in crops])
Dh = np.unpackbits(Hs[:, None, :] ^ Hs[None, :, :], axis=2).sum(2)
Dc = np.clip(1 - E @ E.T, 0, 2)
Dc[Dh <= DH_T] = 0                      # явные дубликаты склеиваем всегда
np.fill_diagonal(Dc, 0)
Z = linkage(squareform(Dc, checks=False), method='average')
lab = fcluster(Z, COS_T, criterion='distance')
np.save(DST.parent / 'cos_dist.npy', Dc) if DST.parent.exists() else None

# --- группы: кластер = участок. Позитивные и пустые кадры одного кластера не разделяются
clusters = {}
for it, l in zip(items, lab):
    clusters.setdefault(int(l), []).append(it)
sizes = Counter(len(v) for v in clusters.values())
print('кадров', len(items), 'кластеров', len(clusters), 'крупнейшие', sorted((len(v) for v in clusters.values()), reverse=True)[:10])

pos_clusters = [k for k, v in clusters.items() if any(it['boxes'] for it in v)]
rng = random.Random(SEED); rng.shuffle(pos_clusters)
gname = {k: f'G{i + 1:04d}' for i, k in enumerate(sorted(pos_clusters))}
n_test = round(len(pos_clusters) * TEST_SHARE)
test_k = set(pos_clusters[:n_test])

if DST.exists():
    shutil.rmtree(DST)
(DST / 'positives' / 'images').mkdir(parents=True); (DST / 'positives' / 'labels').mkdir(parents=True)
(DST / 'negatives').mkdir()
(DST / 'positives' / 'data.yaml').write_text('names:\n' + ''.join(f'  {i}: {c}\n' for i, c in enumerate(CLASSES)) + 'nc: 4\n')
rows = []
for k in sorted(pos_clusters):
    for j, it in enumerate(it for it in clusters[k] if it['boxes']):
        name = f'{gname[k]}_{j + 1:02d}'
        shutil.copy(it['src'], DST / 'positives' / 'images' / (name + '.jpg'))
        (DST / 'positives' / 'labels' / (name + '.txt')).write_text(
            ''.join(f'{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n' for c, x, y, w, h in it['boxes']))
        rows.append((name + '.jpg', it['src'].name, gname[k], 'test' if k in test_k else 'train/val'))
(DST / 'test_groups.txt').write_text('\n'.join(sorted(gname[k] for k in test_k)) + '\n')

# негативы: пустые кадры только из кластеров без повреждений (чтобы не было того же участка, что в обучении)
neg_pool = [it for k, v in clusters.items() if k not in pos_clusters for it in v]
rng.shuffle(neg_pool)
for i, it in enumerate(neg_pool[:N_NEG]):
    name = f'neg_empty_{i + 1:03d}.jpg'
    shutil.copy(it['src'], DST / 'negatives' / name)
    rows.append((name, it['src'].name, '', 'negative'))
with open(DST / 'sources.csv', 'w', encoding='utf-8') as f:
    f.write('file,original_file,group,part\n' + ''.join(','.join(r) + '\n' for r in rows))
info = {'frames': len(items), 'clusters': len(clusters), 'pos_groups': len(pos_clusters), 'test_groups': n_test,
        'pos_images': sum(1 for r in rows if r[3] != 'negative'), 'neg_pool': len(neg_pool), 'neg_used': min(N_NEG, len(neg_pool)),
        'cos_t': COS_T, 'dh_t': DH_T, 'seed': SEED}
(DST / 'prep_info.json').write_text(json.dumps(info, ensure_ascii=False, indent=1))
print(info)

"""Черновая разметка фото моделью M0 для загрузки в Roboflow.

Запуск:  python prelabel_m0.py <папка с фото G###_##.jpg> <куда сохранить>
Получится папка images/ + labels/ + data.yaml (формат YOLOv8): Roboflow → Upload → перетащить папку целиком.

ВАЖНО: это только черновик. Каждую рамку нужно проверить и исправить: удалить лишние,
дорисовать пропущенные, поправить класс. Если принять рамки M0 без проверки, тест будет
подыгрывать M0 и сравнение M1 − M0 станет нечестным.
"""
import shutil, sys
from pathlib import Path
from ultralytics import YOLO

CLASSES = ['D00', 'D10', 'D20', 'D40']
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
weights = sys.argv[3] if len(sys.argv) > 3 else str(Path(__file__).parent.parent / 'models' / 'm0_rdd2022.pt')
(dst / 'images').mkdir(parents=True, exist_ok=True); (dst / 'labels').mkdir(parents=True, exist_ok=True)
(dst / 'data.yaml').write_text('names:\n' + ''.join(f'  {i}: {c}\n' for i, c in enumerate(CLASSES)) + f'nc: {len(CLASSES)}\n')

model = YOLO(weights)
imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'} and not p.name.startswith('neg_'))
for img in imgs:
    r = model.predict(str(img), conf=0.25, imgsz=640, verbose=False)[0]
    shutil.copy(img, dst / 'images' / img.name)
    lines = [f'{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n' for c, (x, y, w, h) in zip(r.boxes.cls.tolist(), r.boxes.xywhn.tolist())]
    (dst / 'labels' / (img.stem + '.txt')).write_text(''.join(lines))
    print(img.name, len(lines), 'рамок')
print('Готово:', dst)

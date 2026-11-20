"""Этап 2 дообучения: добавляет в обучение кадры без разметки повреждений (фон).
Берутся только кадры из кластеров без повреждений, в которых нет ни одного тестового снимка «без повреждений»."""
import random, shutil, csv, numpy as np
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
SRC=Path('/tmp/gh/YOLOROAD/datasets/road2020/train/Czech'); W=Path('/tmp/exp/run/working'); D=Path('/tmp/exp/run/input/czech'); OUT=Path('/tmp/exp/imp/split2')
xmls=sorted((SRC/'annotations'/'xmls').glob('*.xml')); names=[x.stem+'.jpg' for x in xmls]
Dc=np.load('/tmp/exp/data/cos_dist.npy'); lab=fcluster(linkage(squareform(Dc,checks=False),method='average'),0.08,criterion='distance')
rows=list(csv.DictReader(open(D/'sources.csv',encoding='utf-8')))
pos_orig={r['original_file'] for r in rows if r['part']!='negative'}; testneg={r['original_file'] for r in rows if r['part']=='negative'}
test_orig={r['original_file'] for r in rows if r['part']=='test'}
idx={n:i for i,n in enumerate(names)}
pos_cl={lab[idx[n]] for n in pos_orig}; tn_cl={lab[idx[n]] for n in testneg}
assert len(set(lab))==1293, len(set(lab))
pool=[n for n in names if lab[idx[n]] not in pos_cl and lab[idx[n]] not in tn_cl]
tst=[idx[n] for n in testneg|test_orig]
pool=[n for n in pool if Dc[idx[n]][tst].min()>0.08]     # и не ближе порога к любому тестовому кадру
cl=sorted({lab[idx[n]] for n in pool}); random.Random(1).shuffle(cl)
nval=round(len(cl)*0.15); vcl=set(cl[:nval])
if OUT.exists(): shutil.rmtree(OUT)
shutil.copytree(W/'own_split',OUT,ignore=shutil.ignore_patterns('*.cache'))
k={'train':0,'val':0}; out=[]
for n in pool:
    part='val' if lab[idx[n]] in vcl else 'train'; k[part]+=1
    new=f'bg_{part}_{k[part]:03d}.jpg'; shutil.copy(SRC/'images'/n,OUT/part/'images'/new); out.append((new,n,part))
(OUT/'data.yaml').write_text(f'names:\n  0: D00\n  1: D10\n  2: D20\n  3: D40\npath: {OUT}\ntrain: train\nval: val\n')
csv.writer(open(OUT/'background_sources.csv','w',encoding='utf-8')).writerows([('file','original_file','part')]+out)
print('пул',len(pool),'кластеров',len(cl),k)

import sys, random, shutil
from pathlib import Path
sys.path.insert(0,'/tmp/web'); from boxes import B
CL={'D00':0,'D10':1,'D20':2,'D40':3}
W=Path('/tmp/exp/run/working/own_split'); OUT=Path('/tmp/web/split3'); T=Path('/tmp/web/test3')
for p in (OUT,T):
    if p.exists(): shutil.rmtree(p)
ks=sorted(B); rng=random.Random(0); rng.shuffle(ks)
test, val, train = ks[:16], ks[16:22], ks[22:]
for part in ('train','val'):
    for sub in ('images','labels'): (OUT/part/sub).mkdir(parents=True)
(T/'images').mkdir(parents=True); (T/'labels').mkdir()
def put(n, part):
    img=f'/tmp/web/{n}.jpg'; lab=''.join(f'{CL[c]} {(x1+x2)/2:.5f} {(y1+y2)/2:.5f} {x2-x1:.5f} {y2-y1:.5f}\n' for c,x1,y1,x2,y2 in B[n])
    if part=='test': d=T
    else: d=OUT/part
    shutil.copy(img,d/'images'/f'{n}.jpg'); (d/'labels'/f'{n}.txt').write_text(lab)
for n in test: put(n,'test')
for n in val: put(n,'val')
for n in train:
    for r in range(6):
        shutil.copy(f'/tmp/web/{n}.jpg',OUT/'train'/'images'/f'{n}_r{r}.jpg'); put(n,'train'); 
        (OUT/'train'/'labels'/f'{n}_r{r}.txt').write_text((OUT/'train'/'labels'/f'{n}.txt').read_text())
    (OUT/'train'/'images'/f'{n}.jpg').unlink(); (OUT/'train'/'labels'/f'{n}.txt').unlink()
# Чехия: случайные 300 снимков обучения + вся проверочная часть
cz=sorted((W/'train'/'images').glob('*.jpg')); rng.shuffle(cz)
for p in cz[:300]:
    shutil.copy(p,OUT/'train'/'images'/p.name); shutil.copy(W/'train'/'labels'/(p.stem+'.txt'),OUT/'train'/'labels'/(p.stem+'.txt'))
for p in (W/'val'/'images').glob('*.jpg'):
    shutil.copy(p,OUT/'val'/'images'/p.name); shutil.copy(W/'val'/'labels'/(p.stem+'.txt'),OUT/'val'/'labels'/(p.stem+'.txt'))
(OUT/'data.yaml').write_text(f'names:\n  0: D00\n  1: D10\n  2: D20\n  3: D40\npath: {OUT}\ntrain: train\nval: val\n')
open('/tmp/web/split_info.txt','w').write(f'test {test}\nval {val}\ntrain {train}\n')
print(len(test),len(val),len(train), len(list((OUT/'train'/'images').glob('*.jpg'))), len(list((OUT/'val'/'images').glob('*.jpg'))))

import pickle, sys, numpy as np
from pathlib import Path
RUN=Path('/tmp/exp/run'); W=RUN/'working'; D=RUN/'input'/'czech'
tg=set((D/'test_groups.txt').read_text().split())
test=sorted(p for p in (D/'positives'/'images').glob('*.jpg') if p.name.split('_')[0] in tg)
neg=sorted((D/'negatives').glob('*.jpg')); val=sorted((W/'own_split'/'val'/'images').glob('*.jpg'))
def gt(lp):
    o=[]
    for l in lp.read_text().split('\n'):
        v=l.split()
        if len(v)==5:
            c,x,y,w,h=int(v[0]),*map(float,v[1:]); o.append((c,np.array([x-w/2,y-h/2,x+w/2,y+h/2])))
    return o
def iou(a,b):
    ix=max(0,min(a[2],b[2])-max(a[0],b[0])); iy=max(0,min(a[3],b[3])-max(a[1],b[1])); i=ix*iy
    return i/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i+1e-9)
def ap(imgs,lab,P,cls=None,thr=0.5):
    recs=[];n=0
    for p in imgs:
        g=[x for x in gt(lab/(p.stem+'.txt')) if cls is None or x[0]==cls]; n+=len(g); used=[0]*len(g)
        c,s,b=P[p.name]; o=np.argsort(-s)
        for k in o:
            if cls is not None and c[k]!=cls: continue
            best,bj=thr,-1
            for j,(gc,gb) in enumerate(g):
                if not used[j]:
                    v=iou(b[k],gb)
                    if v>=best: best,bj=v,j
            if bj>=0: used[bj]=1
            recs.append((s[k],bj>=0))
    recs.sort(key=lambda r:-r[0]); tp=np.cumsum([r[1] for r in recs]); fp=np.cumsum([not r[1] for r in recs])
    rec=tp/max(n,1); pr=tp/np.maximum(tp+fp,1)
    mp=np.concatenate([[1],pr,[0]]); mr=np.concatenate([[0],rec,[1]])
    for i in range(len(mp)-2,-1,-1): mp[i]=max(mp[i],mp[i+1])
    a=np.sum((mr[1:]-mr[:-1])*mp[1:]); sc=np.array([r[0] for r in recs])
    return a,rec,pr,sc
for f in sys.argv[1:]:
    P=pickle.load(open(f,'rb')); print('==',f)
    L=D/'positives'/'labels'
    aps=[ap(test,L,P,c)[0] for c in range(4)]; print(' test AP D00 D10 D20 D40:',np.round(aps,3),'mAP',round(np.mean(aps),3))
    for thr in (0.3,0.2):
        print('  mAP@IoU%.1f'%thr, round(np.mean([ap(test,L,P,c,thr)[0] for c in range(4)]),3))
    a,rec,pr,sc=ap(test,L,P,None); print(' class-agnostic AP',round(a,3))
    for t in (0.1,0.15,0.25,0.4,0.5):
        k=(sc>=t).sum(); 
        if k: print('  conf %.2f P %.3f R %.3f'%(t,pr[k-1],rec[k-1]))
    # image-level: threshold chosen as max score
    mx=lambda p: (P[p.name][1].max() if len(P[p.name][1]) else 0)
    sp=np.array([mx(p) for p in test]); sn=np.array([mx(p) for p in neg])
    for t in (0.15,0.2,0.25,0.3,0.4,0.5):
        tpr=(sp>=t).mean(); tnr=(sn<t).mean(); acc=((sp>=t).sum()+(sn<t).sum())/(len(sp)+len(sn))
        print('  img-level conf %.2f: sens %.3f spec %.3f acc %.3f bal %.3f'%(t,tpr,tnr,acc,(tpr+tnr)/2))

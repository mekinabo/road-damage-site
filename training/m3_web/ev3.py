import pickle, numpy as np, sys
from pathlib import Path
from ultralytics import YOLO
src=open('/tmp/exp/imp/ev.py').read().split("for f in sys.argv[1:]:")[0]
sys.argv=['x']; exec(src)          # даёт ap(), gt(), test, neg, D (чешский тест)
T=Path('/tmp/web/test3'); wimgs=sorted((T/'images').glob('*.jpg'))
W={'M0':'/tmp/exp/run/input/m0_rdd2022.pt','M1':'/tmp/exp/run/working/m1_finetuned.pt','M3':'/tmp/web/m3_web.pt'}
def pred(m,imgs):
    o={}
    for p in imgs:
        b=m.predict(str(p),imgsz=640,conf=0.01,verbose=False)[0].boxes
        o[p.name]=(b.cls.numpy().astype(int),b.conf.numpy(),b.xyxyn.numpy())
    return o
res={}
for k,w in W.items():
    m=YOLO(w); res[k]={'web':pred(m,wimgs),'cz':pred(m,test),'neg':pred(m,neg)}
pickle.dump(res,open('/tmp/web/res3.pkl','wb'))
Lc=D/'positives'/'labels'; Lw=T/'labels'
for k in W:
    P=res[k]
    w_cls=np.mean([ap(wimgs,Lw,P['web'],c)[0] if any(int(l.split()[0])==c for p in wimgs for l in (Lw/(p.stem+'.txt')).read_text().split('\n') if l) else np.nan for c in (3,)])
    wa=ap(wimgs,Lw,P['web'],None)[0]; wd=ap(wimgs,Lw,P['web'],3)[0]
    c4=np.mean([ap(test,Lc,P['cz'],c)[0] for c in range(4)])
    nf=np.mean([(P['neg'][p.name][1]>=0.25).sum() for p in neg])
    # доля рамок разметки, найденных при уверенности 0,25
    f=0;n=0
    for p in wimgs:
        g=gt(Lw/(p.stem+'.txt')); n+=len(g); used=[0]*len(g)
        c,s,b=P['web'][p.name]
        for i in np.argsort(-s):
            if s[i]<0.25: break
            for j,(gc,gb) in enumerate(g):
                if not used[j] and iou(b[i],gb)>=0.5: used[j]=1; break
        f+=sum(used)
    print(k,'web AP50 D40 %.3f agnostic %.3f | recall@0.25 %d/%d | Czech mAP50 %.3f | FP/neg %.2f'%(wd,wa,f,n,c4,nf))

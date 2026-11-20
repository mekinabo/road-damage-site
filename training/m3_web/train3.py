import shutil, json, time
from pathlib import Path
from ultralytics import YOLO
H=Path('/tmp/web'); last=H/'runs'/'m3'/'weights'/'last.pt'; t=time.time()
if last.exists(): m=YOLO(str(last)); m.train(resume=True)
else:
    m=YOLO('/tmp/exp/run/working/m1_finetuned.pt')
    m.train(data=str(H/'split3'/'data.yaml'),epochs=20,patience=8,imgsz=640,batch=16,workers=4,device='cpu',optimizer='AdamW',lr0=0.0005,warmup_epochs=1,project=str(H/'runs'),name='m3',exist_ok=True,seed=0,verbose=False)
shutil.copy(Path(m.trainer.save_dir)/'weights'/'best.pt',H/'m3_web.pt'); json.dump({'minutes':round((time.time()-t)/60,1)},open(H/'train3_info.json','w'))

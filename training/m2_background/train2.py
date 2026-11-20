"""Этап 2: продолжение дообучения M1 с добавлением кадров без повреждений (фон). Результат — m2_finetuned.pt."""
import json, shutil, time, sys
from pathlib import Path
from ultralytics import YOLO
HERE = Path('/tmp/exp/imp'); last = HERE / 'runs' / 'm2' / 'weights' / 'last.pt'
t = time.time()
if last.exists():
    m = YOLO(str(last)); m.train(resume=True)
else:
    m = YOLO('/tmp/exp/run/working/m1_finetuned.pt')
    m.train(data=str(HERE / 'split2' / 'data.yaml'), epochs=30, patience=10, imgsz=640, batch=16, workers=4, device='cpu',
            optimizer='AdamW', lr0=0.0005, warmup_epochs=1, project=str(HERE / 'runs'), name='m2', exist_ok=True, seed=0, verbose=False)
shutil.copy(Path(m.trainer.save_dir) / 'weights' / 'best.pt', HERE / 'm2_finetuned.pt')
json.dump({'minutes': round((time.time() - t) / 60, 1)}, open(HERE / 'train2_info.json', 'w'))

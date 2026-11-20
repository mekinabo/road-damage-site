"""Сайт обнаружения повреждений дорожного покрытия (FastAPI + YOLO).

Запуск:  python app.py            (или uvicorn app:app)
Модель:  переменная окружения MODEL_PATH, иначе первая найденная из DEFAULT_MODELS.
Настройки (переменные окружения): HOST, PORT, MAX_UPLOAD_MB, RATE_PER_MIN, DOCS=1.
"""
import asyncio
import base64
import io
import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.exceptions import HTTPException as StarletteHTTPException
os.environ.setdefault('YOLO_AUTOINSTALL', 'False')  # ultralytics не должен сам ставить пакеты во время запроса
from ultralytics import YOLO

try:  # фото iPhone в формате HEIC
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC = True
except ImportError:
    HEIC = False

log = logging.getLogger('road-damage')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

HERE = Path(__file__).parent
STATIC = HERE / 'static'
MODEL_DIRS = [HERE / 'models', HERE.parent / 'models']  # папка models/ рядом с app.py (архив сайта) или уровнем выше
# M3 (дообучена с добавлением снимков из интернета) важнее M1, M1 важнее базовой M0; COCO-модель только как крайняя заглушка
DEFAULT_MODELS = [d / n for n in ('m3_web.pt', 'm1_finetuned.pt', 'm0_rdd2022.pt') for d in MODEL_DIRS] + [Path('yolov8s.pt')]
IMGSZ = 640
MIN_CONF = 0.05  # сервер отдаёт всё от 5 %, порог выбирается ползунком на странице

MAX_UPLOAD = int(float(os.environ.get('MAX_UPLOAD_MB', 20)) * 1024 * 1024)
MAX_PIXELS = 25_000_000         # защита от «бомб»: маленький PNG 30000×30000 занял бы гигабайты памяти
WORK_SIDE = 2048                # дальше работаем с уменьшенной копией, модели больше не нужно
MIN_SIDE = 32
ALLOWED_FORMATS = {'JPEG', 'MPO', 'PNG', 'WEBP', 'BMP', 'GIF', 'TIFF', 'HEIF', 'AVIF'}
RATE_PER_MIN = int(os.environ.get('RATE_PER_MIN', 30))  # анализов в минуту с одного адреса
MAX_QUEUE = 12                  # сколько запросов может ждать модель, остальным «сервер занят»
UPLOAD_TIMEOUT = 60             # секунд на загрузку файла: защита от сверхмедленных соединений
QUEUE_TIMEOUT = 60              # секунд ожидания своей очереди к модели
PREVIEW_SIDE = 1600

Image.MAX_IMAGE_PIXELS = MAX_PIXELS

CLASSES = {
    'D00': 'продольная трещина',
    'D10': 'поперечная трещина',
    'D20': 'сетка трещин',
    'D40': 'выбоина',
}
# имена классов в разных весах -> код RDD
ALIASES = {
    'd00': 'D00', 'longitudinal crack': 'D00', 'longitudinal': 'D00',
    'd10': 'D10', 'transverse crack': 'D10', 'transverse': 'D10',
    'd20': 'D20', 'alligator crack': 'D20', 'alligator': 'D20',
    'd40': 'D40', 'pothole': 'D40', 'potholes': 'D40',
}


def pick_model_path():
    env = os.environ.get('MODEL_PATH')
    if env:
        return Path(env)
    for p in DEFAULT_MODELS:
        if p.exists():
            return p
    return DEFAULT_MODELS[-1]  # ultralytics скачает её сам


MODEL_PATH = pick_model_path()
log.info('Загружаю модель %s', MODEL_PATH)
model = YOLO(str(MODEL_PATH))
CODES = {i: ALIASES.get(str(n).lower().strip(), str(n)) for i, n in model.names.items()}
IS_ROAD_MODEL = any(c in CLASSES for c in CODES.values())
model.predict(Image.new('RGB', (IMGSZ, IMGSZ)), imgsz=IMGSZ, verbose=False)  # прогрев: первый запрос не будет долгим

# модель не потокобезопасна: один анализ за раз, остальные ждут в короткой очереди
model_lock = threading.Lock()
waiting = 0

docs = os.environ.get('DOCS') == '1'
app = FastAPI(title='Обнаружение повреждений дорожного покрытия',
              docs_url='/docs' if docs else None, redoc_url=None,
              openapi_url='/openapi.json' if docs else None)


# ---------- защита ----------

CSP = ("default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; "
       "connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; "
       "form-action 'self'; frame-ancestors 'none'")
SECURITY_HEADERS = {
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'Referrer-Policy': 'no-referrer',
    'Permissions-Policy': 'camera=(), microphone=(), geolocation=(), payment=(), usb=()',
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Resource-Policy': 'same-origin',
}


class BodyTooLarge(Exception):
    pass


class BodyTimeout(Exception):
    pass


async def reply(send, status, detail, headers=()):
    body = json.dumps({'detail': detail}, ensure_ascii=False).encode()
    try:
        await send({'type': 'http.response.start', 'status': status,
                    'headers': [(b'content-type', b'application/json; charset=utf-8'),
                                (b'connection', b'close'), *headers]})
        await send({'type': 'http.response.body', 'body': body})
    except Exception:
        pass  # ответ уже начат или клиент ушёл


class Guard:
    """Отсекает лишнее ещё до приёма файла: частоту запросов, очередь, размер и время загрузки."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        global waiting
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        queued = False
        too_big = f'Файл слишком большой: максимум {MAX_UPLOAD // 2**20} МБ'
        if scope['method'] == 'POST' and scope['path'] == '/api/detect':
            ip = scope['client'][0] if scope.get('client') else '?'
            retry = rate_limited(ip)
            if retry:
                return await reply(send, 429, f'Слишком много запросов, подождите {retry} с',
                                   [(b'retry-after', str(retry).encode())])
            if waiting >= MAX_QUEUE:
                return await reply(send, 503, 'Сервер занят, попробуйте через несколько секунд',
                                   [(b'retry-after', b'5')])
            waiting += 1  # место в очереди занимаем сразу, иначе пачка одновременных запросов обходит предел
            queued = True
        try:
            await self._guarded(scope, receive, send, too_big)
        finally:
            if queued:
                waiting -= 1

    async def _guarded(self, scope, receive, send, too_big):
        limit = MAX_UPLOAD + 64 * 1024  # запас на служебные поля формы
        try:
            declared = int(dict(scope['headers']).get(b'content-length', b'0'))
        except ValueError:
            declared = 0
        if declared > limit:
            return await reply(send, 413, too_big)
        received, done = 0, False
        deadline = time.monotonic() + UPLOAD_TIMEOUT

        async def guarded_receive():
            nonlocal received, done
            if done:
                return await receive()
            try:
                message = await asyncio.wait_for(receive(), max(0.1, deadline - time.monotonic()))
            except asyncio.TimeoutError:
                scope['upload_error'] = (408, 'Файл загружался слишком долго, проверьте интернет и попробуйте ещё раз')
                raise BodyTimeout
            if message['type'] == 'http.request':
                received += len(message.get('body', b''))
                if received > limit:
                    scope['upload_error'] = (413, too_big)
                    raise BodyTooLarge
                done = not message.get('more_body', False)
            return message

        try:
            await self.app(scope, guarded_receive, send)
        except BodyTooLarge:
            await reply(send, 413, too_big)
        except BodyTimeout:
            await reply(send, 408, 'Файл загружался слишком долго, проверьте интернет и попробуйте ещё раз')


app.add_middleware(Guard)

hits = defaultdict(deque)
hits_lock = threading.Lock()


def rate_limited(ip):
    """Секунд до следующей разрешённой попытки, 0 — можно."""
    now = time.monotonic()
    with hits_lock:
        q = hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_PER_MIN:
            return int(60 - (now - q[0])) + 1
        q.append(now)
        if len(hits) > 10_000:  # не даём словарю расти бесконечно
            for k in [k for k, v in hits.items() if not v or now - v[-1] > 60]:
                del hits[k]
    return 0


@app.middleware('http')
async def security_headers(request: Request, call_next):
    try:
        response = await call_next(request)
    except (BodyTooLarge, BodyTimeout):
        raise
    except Exception:
        log.exception('Необработанная ошибка: %s %s', request.method, request.url.path)
        response = JSONResponse({'detail': 'Внутренняя ошибка сервера, попробуйте ещё раз'}, 500)
    for k, v in SECURITY_HEADERS.items():
        response.headers.setdefault(k, v)
    if not request.url.path.startswith('/docs'):  # Swagger грузит скрипты с CDN
        response.headers.setdefault('Content-Security-Policy', CSP)
    if request.url.scheme == 'https' or request.headers.get('x-forwarded-proto') == 'https':
        response.headers.setdefault('Strict-Transport-Security', 'max-age=31536000')
    if request.url.path.startswith('/api/'):
        response.headers['Cache-Control'] = 'no-store'
    else:
        response.headers.setdefault('Cache-Control', 'no-cache')  # после обновления сайта сразу новая версия
    return response


@app.exception_handler(StarletteHTTPException)
async def http_error(request, exc):
    if exc.status_code == 400 and 'upload_error' in request.scope:  # FastAPI превратил обрыв загрузки в 400
        status, detail = request.scope['upload_error']
        return JSONResponse({'detail': detail}, status)
    detail = exc.detail if isinstance(exc.detail, str) else 'Ошибка запроса'
    if exc.status_code == 404:
        detail = 'Страница не найдена'
    elif exc.status_code == 405:
        detail = 'Метод не поддерживается'
    elif exc.status_code == 400:  # FastAPI не смог разобрать форму, чаще всего из-за оборванной загрузки
        detail = f'Не удалось прочитать загруженный файл (максимум {MAX_UPLOAD // 2**20} МБ)'
    elif exc.status_code >= 500:
        detail = 'Внутренняя ошибка сервера, попробуйте ещё раз'
    return JSONResponse({'detail': detail}, exc.status_code, headers=getattr(exc, 'headers', None))


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    return JSONResponse({'detail': 'Не передан файл с фото'}, 422)


# ---------- страницы и API ----------

@app.get('/', include_in_schema=False)
def index():
    return FileResponse(STATIC / 'index.html')


@app.get('/favicon.ico', include_in_schema=False)
def favicon():
    return FileResponse(STATIC / 'icon.svg', media_type='image/svg+xml')


@app.get('/api/health')
def health():
    return {'ok': True, 'queue': waiting}


@app.get('/api/model')
def model_info():
    return {
        'file': MODEL_PATH.name,
        'road_model': IS_ROAD_MODEL,
        'classes': {code: CLASSES.get(code, code) for code in CODES.values()},
        'max_upload_mb': MAX_UPLOAD // 2**20,
        'heic': HEIC,
    }


decode_slots = threading.BoundedSemaphore(2)  # раскодированное фото занимает много памяти: не больше двух разом


def open_image(data):
    try:
        img = Image.open(io.BytesIO(data))
        if img.format not in ALLOWED_FORMATS:
            raise HTTPException(415, 'Этот формат не поддерживается: загрузите JPG, PNG, WEBP или HEIC')
        if img.width * img.height > MAX_PIXELS:  # Pillow сам падает только при вдвое большем размере
            raise Image.DecompressionBombError
        with decode_slots:
            img.draft('RGB', (WORK_SIDE, WORK_SIDE))  # JPEG сразу раскодируется в уменьшенном виде
            img.thumbnail((WORK_SIDE, WORK_SIDE))     # в очереди ждёт уже маленькая копия
            img = ImageOps.exif_transpose(img).convert('RGB')  # фото с телефона бывают повёрнуты в EXIF
    except HTTPException:
        raise
    except Image.DecompressionBombError:
        raise HTTPException(413, f'Слишком большое разрешение: максимум {MAX_PIXELS // 1_000_000} мегапикселей')
    except Exception:  # битый или подменённый файл: Pillow бросает самые разные ошибки
        raise HTTPException(415, 'Не удалось открыть файл как изображение: загрузите JPG, PNG, WEBP или HEIC')
    if min(img.size) < MIN_SIDE:
        raise HTTPException(422, 'Изображение слишком маленькое')
    return img


def analyse(img):
    if not model_lock.acquire(timeout=QUEUE_TIMEOUT):
        raise HTTPException(503, 'Сервер занят, попробуйте через несколько секунд', headers={'Retry-After': '5'})
    try:
        t0 = time.perf_counter()
        r = model.predict(img, imgsz=IMGSZ, conf=MIN_CONF, verbose=False)[0]
        ms = round((time.perf_counter() - t0) * 1000)
    finally:
        model_lock.release()
    W, H = img.size
    found = []
    for cls, conf, (x1, y1, x2, y2) in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
        code = CODES.get(int(cls), str(int(cls)))
        found.append({
            'code': code,
            'name': CLASSES.get(code, code),
            'conf': round(conf, 3),
            'box': [max(0, round(x1)), max(0, round(y1)), min(W, round(x2)), min(H, round(y2))],
            # доля кадра под рамкой, а не площадь повреждения: в м² без масштаба снимка не перевести
            'frame_share': round((x2 - x1) * (y2 - y1) / (W * H) * 100, 1),
        })
    found.sort(key=lambda d: -d['conf'])
    return {'width': W, 'height': H, 'ms': ms, 'model': MODEL_PATH.name, 'detections': found}


def make_preview(img, result):
    """Картинка для браузеров, которые сами не умеют показывать формат (HEIC в Chrome)."""
    scale = min(1, PREVIEW_SIDE / max(img.size))
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
        for d in result['detections']:  # рамки в координатах уменьшенной картинки
            d['box'] = [round(v * scale) for v in d['box']]
        result['width'], result['height'] = img.size
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=85)
    result['preview'] = 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


@app.post('/api/detect')
async def detect(file: UploadFile = File(...), preview: str = Form('0')):
    # частота запросов и длина очереди уже проверены и учтены в Guard
    data = await file.read(MAX_UPLOAD + 1)
    await file.close()
    if not data:
        raise HTTPException(422, 'Файл пустой')
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, f'Файл слишком большой: максимум {MAX_UPLOAD // 2**20} МБ')

    img = await run_in_threadpool(open_image, data)
    del data
    result = await run_in_threadpool(analyse, img)
    if preview == '1':
        await run_in_threadpool(make_preview, img, result)
    return result


app.mount('/static', StaticFiles(directory=STATIC), name='static')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host=os.environ.get('HOST', '127.0.0.1'), port=int(os.environ.get('PORT', 8000)),
                limit_concurrency=100, timeout_keep_alive=10, proxy_headers=True)

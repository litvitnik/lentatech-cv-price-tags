import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Dict

import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

# Импорт пайплайна должен быть после установки переменной окружения
os.environ['DYLD_LIBRARY_PATH'] = '/opt/homebrew/opt/zbar/lib'
from detect_price_tags import PriceTagPipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'web_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(BASE_DIR, 'runs/detect/runs/detect/price_tag_v2/weights/best.pt')

app = FastAPI(title="Price Tag Detector")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, 'templates'))


@dataclass
class TaskInfo:
    status: str = 'pending'  # pending, processing, done, error
    progress: float = 0.0
    message: str = ''
    result_dir: str = ''
    error: str = ''


tasks: Dict[str, TaskInfo] = {}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload(video: UploadFile = File(...), assume_99_kopecks: str = Form('1')):
    task_id = uuid.uuid4().hex[:12]
    task_dir = os.path.join(RESULTS_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)

    video_path = os.path.join(task_dir, video.filename or 'video.mp4')
    content = await video.read()
    with open(video_path, 'wb') as f:
        f.write(content)

    assume99 = assume_99_kopecks == '1'
    tasks[task_id] = TaskInfo(status='processing', message='Задача создана', result_dir=task_dir)

    asyncio.create_task(run_pipeline(task_id, video_path, assume99))

    return {"task_id": task_id}


@app.websocket("/ws/{task_id}")
async def ws_progress(ws: WebSocket, task_id: str):
    await ws.accept()
    try:
        while True:
            task = tasks.get(task_id)
            if task is None:
                await ws.send_json({"error": "Task not found"})
                break

            await ws.send_json({
                "status": task.status,
                "progress": task.progress,
                "message": task.message,
            })

            if task.status in ('done', 'error'):
                await ws.close()
                break

            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass


@app.get("/result/{task_id}", response_class=HTMLResponse)
async def result_page(request: Request, task_id: str):
    task = tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    csv_path = os.path.join(task.result_dir, 'result.csv')
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="CSV not found yet")

    df = pd.read_csv(csv_path)
    rows = df.to_dict('records')

    # Собираем base64 миниатюры для отображения
    from PIL import Image as PILImage
    from io import BytesIO as B
    import base64 as b64

    for row in rows:
        img_path = row.get('warped_image', '')
        if img_path and os.path.exists(img_path):
            pil_img = PILImage.open(img_path)
            pil_img.thumbnail((300, 300))
            buf = B()
            pil_img.save(buf, format='PNG')
            row['_thumbnail'] = b64.b64encode(buf.getvalue()).decode('utf-8')
        else:
            row['_thumbnail'] = ''

    return templates.TemplateResponse("result.html", {
        "request": request,
        "task_id": task_id,
        "rows": rows,
        "columns": [
            ('_thumbnail', 'Фото'),
            ('product_name', 'Название'),
            ('price_default', 'Цена'),
            ('price_discount', 'Цена со скидкой'),
            ('discount_amount', 'Скидка'),
            ('barcode', 'Штрихкод'),
            ('color', 'Цвет'),
        ],
    })


@app.get("/download/{task_id}/csv")
async def download_csv(task_id: str):
    task = tasks.get(task_id)
    if not task or not task.result_dir:
        raise HTTPException(status_code=404, detail="Task not found")
    csv_path = os.path.join(task.result_dir, 'result.csv')
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="CSV not found")
    return FileResponse(csv_path, media_type='text/csv', filename='result.csv')


@app.get("/download/{task_id}/html")
async def download_html(task_id: str):
    task = tasks.get(task_id)
    if not task or not task.result_dir:
        raise HTTPException(status_code=404, detail="Task not found")
    html_path = os.path.join(task.result_dir, 'report.html')
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="HTML report not found")
    return FileResponse(html_path, media_type='text/html', filename='report.html')


async def run_pipeline(task_id: str, video_path: str, assume_99_kopecks: bool = True):
    task = tasks[task_id]
    task.status = 'processing'

    _max_progress = 0.0

    def on_progress(message: str, fraction: float):
        nonlocal _max_progress
        if fraction < _max_progress:
            fraction = _max_progress
        else:
            _max_progress = fraction
        task.progress = fraction
        task.message = message

    def _run():
        p = PriceTagPipeline(
            detection_model_path=MODEL_PATH,
            orientation_mode='color',
            assume_99_kopecks=assume_99_kopecks,
        )
        csv_path = os.path.join(task.result_dir, 'result.csv')
        debug_dir = os.path.join(task.result_dir, 'debug_output')
        html_path = os.path.join(task.result_dir, 'report.html')

        p.run_to_csv(
            video_path,
            debug=True,
            debug_dir=debug_dir,
            csv_path=csv_path,
            progress_callback=on_progress,
        )
        p.generate_html_report(csv_path, html_path)

    loop = asyncio.get_event_loop()

    try:
        await loop.run_in_executor(None, _run)
        task.status = 'done'
        task.message = 'Готово'
    except Exception as e:
        import traceback
        traceback.print_exc()
        task.status = 'error'
        task.error = str(e)
        task.message = f'Ошибка: {e}'


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)

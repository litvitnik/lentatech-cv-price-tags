# Lenta Price Tag Detector

Система обнаружения и распознавания ценников супермаркетов «Лента» из видео.
Робот с камерой проезжает вдоль полок, алгоритм находит ценники, распознаёт с них текст
(название товара, цены, скидку, штрихкод) и выдаёт структурированный результат в CSV.

## Стек технологий

| Компонент | Технология |
|---|---|
| Детекция объектов | YOLOv8n (Ultralytics) + BoTSORT трекер |
| Распознавание текста | PaddleOCR (русская модель) |
| QR/штрихкод | pyzbar + OpenCV QRCodeDetector (резервный) |
| Нормализация ориентации | HSV-анализ доминирующего цвета |
| Веб-интерфейс | FastAPI + WebSocket + Jinja2 |
| Данные | pandas, OpenCV, Pillow, numpy |


## Ограничения текущего подхода:

Алгоритм в данный момент работает только с ценниками соответствующими формату как в обучающем видео.
С желтыми ценниками не тестировался, с только белыми - тоже и скорее всего не сработает. 
Для работы алгоритма понадобилось предобучение модели обнаружения ценнков.
Затраченное время на предобучене - **10 минут чистого времени, 30 кадров из видео.**
Если иметь видео с каждым из типов ценников, то оцениваю время на обучение модели обнаружения в две рабочих недели.



## Пайплайн обработки

```
Видео
  │
  ├─ 1. Детекция и трекинг (model.track)
  │     YOLOv8n на каждом кадре + BoTSORT —
  │     каждый уникальный ценник получает свой track_id.
  │     Для каждого трека выбирается лучший кадр
  │     (максимальная площадь bounding box).
  │
  ├─ 2. Вырезка и перспективное преобразование
  │     По bounding box вырезается фрагмент кадра (warp).
  │
  ├─ 3. Нормализация ориентации
  │     Цветной части ценника (красная/жёлтая) всегда внизу.
  │     Определяется по HSV-гистограмме: если доминирующий цвет
  │     в верхней половине — изображение поворачивается на 180°.
  │
  ├─ 4. Детекция QR-кода
  │     Основной: pyzbar, резервный: OpenCV QRCodeDetector.
  │     Из QR парсятся поля: barcode, price1-4, wholesale,
  │     actionPrice, actionCode.
  │
  ├─ 5. OCR (PaddleOCR)
  │     Распознаётся текст на выровненном изображении.
  │
  └─ 6. Парсинг полей
        Из распознанного текста извлекаются:
        • product_name — название товара
        • price_default — цена без скидки
        • price_card — цена по карте
        • price_discount — цена со скидкой
        • discount_amount — размер скидки

        Постобработка:
        • Если OCR не распознал копейки, их можно заменить на 99
          (опция assume_99_kopecks, включена по умолчанию).
        • Если price_default < price_discount, price_default очищается
          (цена без скидки менее надёжна — меньший шрифт).
        • discount_amount нормализуется: добавляются «-» и «%».
        • Цены форматируются с запятой: 599.99 → 599,99.
```

## Структура проекта

```
.
├── detect_price_tags.py      # Основной пайплайн (PriceTagPipeline)
├── web_app.py                # FastAPI веб-приложение
├── train_price_tag.py        # Скрипт дообучения модели
├── extract_for_roboflow.py   # Экстракция кадров для разметки
├── templates/
│   ├── index.html            # Страница загрузки видео
│   └── result.html           # Страница с результатами
├── runs/detect/runs/detect/
│   └── price_tag_v2/weights/best.pt  # Текущая модель
├── cv_datasets/              # Датасеты (в gitignore)
│   ├── LentaTechPriceTags.yolov8_v1/
│   └── LentaTechPriceTags.yolov8_v2/
└── .gitignore
```

## CSV: поля результата

```
filename, product_name, price_default, price_card, price_discount,
barcode, discount_amount, id_sku, print_datetime, code,
additional_info, color, special_symbols, frame_timestamp,
x_min, y_min, x_max, y_max, warped_image, raw_text,
qr_code_barcode, price1_qr, price2_qr, price3_qr, price4_qr,
wholesale_level_1_count, wholesale_level_1_price,
wholesale_level_2_count, wholesale_level_2_price,
action_price_qr, action_code_qr
```

- `barcode` — сырые данные QR-кода целиком
- `qr_code_barcode` — штрихкод, извлечённый из QR (поле `barcode`)
- `price1_qr`–`price4_qr` — цены из QR (поля `price1/p1`–`price4/p4`)
- `warped_image` — путь к вырезанному и выровненному изображению ценника
- `raw_text` — полный текст OCR для отладки

## Веб-приложение

Запуск:

```bash
uvicorn web_app:app --host 0.0.0.0 --port 8000
```

1. Открыть `http://localhost:8000`
2. Перетащить или выбрать видео (MP4, AVI, MOV)
3. При необходимости отключить галочку «Заменять 00 копеек на 99»
4. Нажать «Обработать»
5. Следить за прогрессом в реальном времени (WebSocket)
6. После завершения откроется страница результатов:
   - Таблица со всеми распознанными полями (30 колонок)
   - Миниатюры ценников с лайтбоксом по клику
   - Исходный текст OCR для отладки
   - Кнопки «Скачать CSV» и «Скачать HTML-отчёт» в верхней панели

### API endpoints

| Endpoint | Описание |
|---|---|
| `GET /` | Страница загрузки |
| `POST /upload` | Загрузка видео, возвращает `task_id` |
| `WS /ws/{task_id}` | WebSocket с прогрессом обработки |
| `GET /result/{task_id}` | Страница с таблицей результатов |
| `GET /download/{task_id}/csv` | Скачивание CSV |
| `GET /download/{task_id}/html` | Скачивание HTML-отчёта |

## Обучение модели

Датасет размечается в Roboflow в формате YOLOv8. Для дообучения:

```bash
python train_price_tag.py
```

Скрипт выполняет fine-tune модели v1 → v2:
- Стартовые веса: `price_tag_v1/weights/best.pt`
- `lr0=0.001`, `epochs=100`, `patience=15` (ранняя остановка)
- `device='mps'` (Apple Silicon)
- Аугментация включена

Текущие метрики (v2): mAP@0.5 = 0.957, mAP@0.5:0.95 = 0.783.

## Запуск из Python (без веба)

```python
from detect_price_tags import PriceTagPipeline

pipeline = PriceTagPipeline(
    detection_model_path='runs/detect/runs/detect/price_tag_v2/weights/best.pt',
    orientation_mode='color',
    assume_99_kopecks=True,
)
pipeline.run_to_csv(
    video_path='video.mp4',
    debug=True,
    debug_dir='debug_output',
    csv_path='result.csv',
)
pipeline.generate_html_report('result.csv', 'report.html')
```

## Подготовка датасета

1. Снять видео полок магазина
2. `python extract_for_roboflow.py` — извлечь ключевые кадры
3. Загрузить кадры в Robowflow, разметить bounding boxes ценников
4. Экспортировать в формате YOLOv8
5. Разделить на train/val (`split_dataset.py --val_ratio 0.2`)
6. Обновить `data.yaml` и запустить `train_price_tag.py`

from ultralytics import YOLO


def main():
    # 1. Загружаем предобученную nano-модель YOLOv8
    model = YOLO('yolov8n.pt')

    # 2. Запускаем обучение
    results = model.train(
        data='cv_datasets/LentaTechPriceTags.yolov8_v1/data.yaml',  # путь к вашему data.yaml
        epochs=60,  # число эпох (можно начать с 60)
        imgsz=640,  # размер входного изображения
        batch=8,  # размер батча (если GPU с малым VRAM, уменьшите до 4)
        device='mps',  # 'cuda' для GPU, 'cpu' если нет видеокарты
        name='price_tag_v1',  # имя эксперимента
        patience=10,  # ранняя остановка, если val loss не улучшается 10 эпох
        lr0=0.01,  # начальный learning rate
        augment=True,  # стандартные аугментации YOLO
        workers=4,  # число потоков загрузки данных
        project='runs/detect',  # папка для сохранения результатов
        exist_ok=True,  # перезаписывать, если папка существует
    )

    # 3. Метрики после обучения
    print("\n--- Результаты валидации ---")
    metrics = model.val()
    print(f"mAP@0.5: {metrics.box.map50:.3f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall: {metrics.box.mr:.3f}")

    # 4. Экспорт в формат для инференса (опционально)
    model.export(format='onnx')  # можно будет использовать без Python


if __name__ == '__main__':
    main()
from ultralytics import YOLO


def main():
    # 1. Загружаем текущую модель v1 как стартовую точку (fine-tune)
    model = YOLO('runs/detect/runs/detect/price_tag_v1/weights/best.pt')

    # 2. Запускаем fine-tune на датасете v2
    results = model.train(
        data='cv_datasets/LentaTechPriceTags.yolov8_v2/data.yaml',
        epochs=100,
        imgsz=640,
        batch=8,
        device='mps',
        name='price_tag_v2',
        patience=15,
        lr0=0.001,  # ниже чем при обучении с нуля — fine-tune
        augment=True,
        workers=4,
        project='runs/detect',
        exist_ok=True,
    )

    # 3. Метрики после обучения
    print("\n--- Результаты валидации ---")
    metrics = model.val()
    print(f"mAP@0.5: {metrics.box.map50:.3f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall: {metrics.box.mr:.3f}")


if __name__ == '__main__':
    main()
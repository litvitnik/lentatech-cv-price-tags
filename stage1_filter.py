import cv2
import numpy as np
import os
import math
import argparse
from skimage.metrics import structural_similarity as ssim


def extract_keyframes(video_path, output_dir="keyframes", max_frames=50,
                      laplacian_threshold=100, ssim_threshold=0.95, collage_cols=10):
    """
    Извлекает до max_frames чётких и не дублирующихся кадров из видео.
    Сохраняет кадры в output_dir и создаёт обзорный коллаж overview_collage.jpg.
    """
    # Создаём папку для сохранения
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Видео: {video_path}")
    print(f"FPS: {fps:.2f}, всего кадров: {total_frames}")

    keyframes = []  # список словарей: timestamp_ms, image, sharpness
    last_gray = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int((frame_idx / fps) * 1000) if fps > 0 else frame_idx * 40  # fallback

        # 1. Проверка чёткости (дисперсия Лапласиана)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < laplacian_threshold:
            frame_idx += 1
            continue

        # 2. Проверка на дубликат по SSIM
        if last_gray is not None:
            # Приведение к одному размеру (на случай изменения разрешения)
            if gray.shape != last_gray.shape:
                last_gray = cv2.resize(last_gray, (gray.shape[1], gray.shape[0]))
            score, _ = ssim(gray, last_gray, full=True)
            if score > ssim_threshold:
                frame_idx += 1
                continue

        # Если прошли оба фильтра — сохраняем кадр
        keyframes.append({
            "timestamp_ms": timestamp_ms,
            "image": frame.copy(),
            "sharpness": lap_var,
        })
        last_gray = gray

        # Останавливаемся, если набрали нужное количество
        if len(keyframes) >= max_frames:
            print(f"Набрано {max_frames} ключевых кадров, останавливаемся на кадре {frame_idx}")
            break

        frame_idx += 1

    cap.release()
    print(f"Всего отобрано ключевых кадров: {len(keyframes)} из {total_frames}")

    # Сохраняем каждый кадр в файл
    for kf in keyframes:
        fname = f"frame_{kf['timestamp_ms']:06d}.png"
        cv2.imwrite(os.path.join(output_dir, fname), kf['image'])

    # Создаём коллаж для отладки
    if keyframes:
        create_collage(keyframes, os.path.join(output_dir, "overview_collage.jpg"), cols=collage_cols)
        print(f"Коллаж сохранён в {os.path.join(output_dir, 'overview_collage.jpg')}")

    return keyframes


def create_collage(keyframes, collage_path, cols=10, thumb_size=(320, 180)):
    """
    Создаёт изображение-коллаж из миниатюр ключевых кадров с подписями времени и резкости.
    """
    n = len(keyframes)
    if n == 0:
        return
    rows = math.ceil(n / cols)
    canvas = np.ones((rows * thumb_size[1] + 30, cols * thumb_size[0], 3), dtype=np.uint8) * 255

    for i, kf in enumerate(keyframes):
        r, c = divmod(i, cols)
        x = c * thumb_size[0]
        y = r * thumb_size[1]
        thumb = cv2.resize(kf['image'], thumb_size)
        canvas[y:y+thumb_size[1], x:x+thumb_size[0]] = thumb
        label = f"t={kf['timestamp_ms']}ms L={kf['sharpness']:.0f}"
        cv2.putText(canvas, label, (x+5, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 1)

    # Заполняем пустые ячейки белым
    # (уже белые, так как инициализировали canvas белым)
    cv2.imwrite(collage_path, canvas)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Извлечение чётких ключевых кадров из видео.")
    parser.add_argument("video", help="Путь к видеофайлу")
    parser.add_argument("--output", "-o", default="keyframes", help="Папка для сохранения кадров")
    parser.add_argument("--max_frames", "-n", type=int, default=50, help="Максимум ключевых кадров")
    parser.add_argument("--laplacian", "-l", type=float, default=100.0, help="Порог чёткости (Laplacian variance)")
    parser.add_argument("--ssim", "-s", type=float, default=0.95, help="Порог схожести (SSIM), выше которого кадр считается дубликатом")
    parser.add_argument("--cols", "-c", type=int, default=10, help="Количество столбцов в коллаже")
    args = parser.parse_args()

    extract_keyframes(
        video_path=args.video,
        output_dir=args.output,
        max_frames=args.max_frames,
        laplacian_threshold=args.laplacian,
        ssim_threshold=args.ssim,
        collage_cols=args.cols,
    )
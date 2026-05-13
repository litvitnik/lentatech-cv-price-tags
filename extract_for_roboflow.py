import cv2
import numpy as np
import os
import argparse
from skimage.metrics import structural_similarity as ssim

def extract_frames_for_roboflow(video_path, output_dir="roboflow", max_frames=200,
                                laplacian_threshold=100, ssim_threshold=0.95):
    """
    Извлекает чёткие и не дублирующиеся кадры из видео и сохраняет в output_dir.
    Имена файлов: frame_<timestamp_ms>.png
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Видео: {video_path}")
    print(f"FPS: {fps:.2f}, всего кадров: {total_frames}")

    keyframes_count = 0
    last_gray = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int((frame_idx / fps) * 1000) if fps > 0 else frame_idx * 40

        # 1. Проверка чёткости
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < laplacian_threshold:
            frame_idx += 1
            continue

        # 2. Проверка на дубликат (сравнение с последним сохранённым)
        if last_gray is not None:
            if gray.shape != last_gray.shape:
                last_gray_resized = cv2.resize(last_gray, (gray.shape[1], gray.shape[0]))
            else:
                last_gray_resized = last_gray
            score, _ = ssim(gray, last_gray_resized, full=True)
            if score > ssim_threshold:
                frame_idx += 1
                continue

        # Сохраняем кадр
        fname = f"frame_{timestamp_ms:06d}.png"
        cv2.imwrite(os.path.join(output_dir, fname), frame)
        print(f"Сохранён {fname}  (чёткость: {lap_var:.1f})")
        keyframes_count += 1

        # Обновляем last_gray на текущий кадр (оригинальная яркость)
        last_gray = gray

        if keyframes_count >= max_frames:
            print(f"Достигнут лимит {max_frames} кадров, остановка.")
            break

        frame_idx += 1

    cap.release()
    print(f"Готово. Всего сохранено кадров: {keyframes_count} в папку '{output_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Извлечение кадров для разметки в Roboflow.")
    parser.add_argument("video", help="Путь к видеофайлу")
    parser.add_argument("--output", "-o", default="roboflow", help="Папка для сохранения кадров")
    parser.add_argument("--max_frames", "-n", type=int, default=200,
                        help="Максимум сохраняемых кадров (по умолчанию 200)")
    parser.add_argument("--laplacian", "-l", type=float, default=100.0,
                        help="Порог чёткости (Laplacian variance)")
    parser.add_argument("--ssim", "-s", type=float, default=0.95,
                        help="Порог схожести (SSIM), выше которого кадр считается дубликатом")
    args = parser.parse_args()

    extract_frames_for_roboflow(
        video_path=args.video,
        output_dir=args.output,
        max_frames=args.max_frames,
        laplacian_threshold=args.laplacian,
        ssim_threshold=args.ssim,
    )
import cv2
import numpy as np
import os
from typing import List, Dict
from ultralytics import YOLO
from skimage.metrics import structural_similarity as ssim


class PriceTagPipeline:
    def __init__(self,
                 detection_model_path: str = "yolov8n.pt",
                 laplacian_thr: float = 100,
                 ssim_thr: float = 0.95):
        """
        Инициализация всех моделей.
        """
        self.laplacian_thr = laplacian_thr
        self.ssim_thr = ssim_thr
        print(f"Загружаю модель детекции: {detection_model_path}")
        self.detector = YOLO(detection_model_path)

    # -------------------- ЭТАП 1 --------------------
    def extract_keyframes(self, video_path: str, max_frames: int = 50) -> List[Dict]:
        """Возвращает список keyframes с полями timestamp_ms, image, sharpness."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Не удалось открыть видео: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        keyframes = []
        last_gray = None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp_ms = int((frame_idx / fps) * 1000) if fps > 0 else frame_idx * 40

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if lap_var < self.laplacian_thr:
                frame_idx += 1
                continue

            if last_gray is not None:
                if gray.shape != last_gray.shape:
                    last_gray = cv2.resize(last_gray, (gray.shape[1], gray.shape[0]))
                score, _ = ssim(gray, last_gray, full=True)
                if score > self.ssim_thr:
                    frame_idx += 1
                    continue

            keyframes.append({
                "timestamp_ms": timestamp_ms,
                "image": frame.copy(),
                "sharpness": lap_var
            })
            last_gray = gray
            if len(keyframes) >= max_frames:
                break
            frame_idx += 1

        cap.release()
        print(f"Этап 1: отобрано {len(keyframes)} ключевых кадров")
        return keyframes

    # -------------------- ЭТАП 2 --------------------
    def detect_price_tags_on_keyframes(self, keyframes: List[Dict]) -> List[Dict]:
        """Добавляет в каждый кадр поле 'price_tags' со списком детекций."""
        print("Этап 2: детекция ценников...")
        for kf in keyframes:
            image = kf['image']
            results = self.detector(image, verbose=False)

            tags = []
            if hasattr(results[0], 'boxes') and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].item()
                    # Обычный прямоугольник (заготовка под OBB)
                    corners = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    tags.append({
                        "corners": corners,
                        "confidence": conf
                    })
            kf['price_tags'] = tags
        return keyframes

    def warp_price_tags(self, keyframes: List[Dict]) -> List[Dict]:
        """
        Для каждого ценника вырезает и выпрямляет область.
        Добавляет к каждому тегу поле 'warped' с изображением (numpy).
        """
        print("Этап 3: вырезание и выпрямление ценников...")
        for kf in keyframes:
            image = kf['image']
            for tag in kf.get('price_tags', []):
                corners = np.array(tag['corners'], dtype=np.float32)
                # Пока corners – прямоугольник: [tl, tr, br, bl]
                # Определяем ширину и высоту выходного изображения
                width = int(max(
                    np.linalg.norm(corners[1] - corners[0]),
                    np.linalg.norm(corners[2] - corners[3])
                ))
                height = int(max(
                    np.linalg.norm(corners[3] - corners[0]),
                    np.linalg.norm(corners[2] - corners[1])
                ))
                # Если размеры мизерные – пропускаем
                if width < 10 or height < 10:
                    tag['warped'] = None
                    continue
                dst = np.array([
                    [0, 0],
                    [width - 1, 0],
                    [width - 1, height - 1],
                    [0, height - 1]
                ], dtype=np.float32)
                M = cv2.getPerspectiveTransform(corners, dst)
                warped = cv2.warpPerspective(image, M, (width, height))
                tag['warped'] = warped
        return keyframes


    # -------------------- ВИЗУАЛЬНАЯ ОТЛАДКА --------------------
    def save_debug_frames(self, keyframes: List[Dict], output_dir: str):
        """Сохраняет исходные кадры и версии с выделенными ценниками."""
        good_dir = os.path.join(output_dir, "good_frames")
        processed_dir = os.path.join(output_dir, "good_frames_processed")
        os.makedirs(good_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)

        for kf in keyframes:
            ts = kf['timestamp_ms']
            img = kf['image'].copy()
            fname = f"frame_{ts:06d}.png"

            # 1. Сохраняем оригинал
            cv2.imwrite(os.path.join(good_dir, fname), img)

            # 2. Затемнённое изображение
            dark = (img * 0.3).astype(np.uint8)

            if len(kf['price_tags']) == 0:
                # Если ценников нет – сохраняем полностью затемнённый кадр
                cv2.imwrite(os.path.join(processed_dir, fname), dark)
                continue

            # Создаём маску областей ценников
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            for tag in kf['price_tags']:
                corners = np.array(tag['corners'], dtype=np.int32)
                cv2.fillPoly(mask, [corners], 255)

            # Переносим пиксели оригинальной яркости в затемнённое изображение по маске
            processed = dark.copy()
            processed[mask == 255] = img[mask == 255]

            # Рисуем контуры и подписи
            for tag in kf['price_tags']:
                corners = np.array(tag['corners'], dtype=np.int32)
                cv2.polylines(processed, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
                x, y = corners[0]
                cv2.putText(processed, f"conf:{tag['confidence']:.2f}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imwrite(os.path.join(processed_dir, fname), processed)

        print(f"Отладочные кадры сохранены в: {output_dir}/")

    # -------------------- ЗАПУСК --------------------
    def run(self, video_path: str, max_frames: int = 50,
            debug: bool = False, debug_dir: str = "debug_output") -> List[Dict]:
        keyframes = self.extract_keyframes(video_path, max_frames)
        keyframes = self.detect_price_tags_on_keyframes(keyframes)
        if debug:
            self.save_debug_frames(keyframes, debug_dir)
        return keyframes


# -------------------- Пример использования --------------------
if __name__ == "__main__":
    pipeline = PriceTagPipeline(detection_model_path='runs/detect/runs/detect/price_tag_v1/weights/best.pt', laplacian_thr=100)
    result = pipeline.run("videos/43_15.mp4", max_frames=50, debug=True, debug_dir="debug_output")

    # Быстрый вывод статистики
    for kf in result:
        print(f"Кадр {kf['timestamp_ms']} мс: найдено ценников {len(kf['price_tags'])}")
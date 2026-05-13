import os
os.environ['DYLD_LIBRARY_PATH'] = '/opt/homebrew/opt/zbar/lib'  # подставьте ваш путь
import cv2
import numpy as np
import os
import re
import pandas as pd
from typing import List, Dict
from ultralytics import YOLO
from skimage.metrics import structural_similarity as ssim
from paddleocr import PaddleOCR
from pyzbar.pyzbar import decode as pyzbar_decode


class PriceTagPipeline:
    def __init__(self,
                 detection_model_path: str = "yolov8n.pt",
                 laplacian_thr: float = 100,
                 ssim_thr: float = 0.95):
        self.laplacian_thr = laplacian_thr
        self.ssim_thr = ssim_thr
        print(f"Загружаю модель детекции: {detection_model_path}")
        self.detector = YOLO(detection_model_path)
        self.ocr = None  # будет загружен при первом обращении

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
                    corners = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    tags.append({
                        "corners": corners,
                        "confidence": conf
                    })
            kf['price_tags'] = tags
        return keyframes

    # -------------------- ЭТАП 3: ВЫРЕЗАНИЕ И ВЫПРЯМЛЕНИЕ --------------------
    def warp_price_tags(self, keyframes: List[Dict]) -> List[Dict]:
        """Вырезает и выпрямляет ценники, добавляет поле 'warped'."""
        print("Этап 3: вырезание и выпрямление ценников...")
        for kf in keyframes:
            image = kf['image']
            for tag in kf.get('price_tags', []):
                corners = np.array(tag['corners'], dtype=np.float32)
                width = int(max(
                    np.linalg.norm(corners[1] - corners[0]),
                    np.linalg.norm(corners[2] - corners[3])
                ))
                height = int(max(
                    np.linalg.norm(corners[3] - corners[0]),
                    np.linalg.norm(corners[2] - corners[1])
                ))
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

    # -------------------- ЭТАП 3.5: ПОИСК QR-КОДА --------------------
    def detect_qr_on_warped(self, keyframes: List[Dict]) -> List[Dict]:
        """Ищет QR-код на каждом выпрямленном ценнике."""
        print("Этап 3.5: поиск QR-кодов...")
        for kf in keyframes:
            for tag in kf.get('price_tags', []):
                warped = tag.get('warped')
                if warped is None:
                    tag['qr_data'] = ''
                    tag['qr_bbox'] = None
                    continue

                # Сначала pyzbar
                qr_data = ''
                qr_bbox = None
                decoded_objs = pyzbar_decode(warped)
                for obj in decoded_objs:
                    if obj.type == 'QRCODE':
                        qr_data = obj.data.decode('utf-8', errors='ignore')
                        points = [(p.x, p.y) for p in obj.polygon]
                        qr_bbox = points
                        break

                # Резерв: встроенный детектор OpenCV
                if not qr_data:
                    detector = cv2.QRCodeDetector()
                    data, bbox_pts, _ = detector.detectAndDecode(warped)
                    if data:
                        qr_data = data
                        if bbox_pts is not None:
                            bbox_pts = bbox_pts.reshape(4, 2).tolist()
                            qr_bbox = bbox_pts

                tag['qr_data'] = qr_data
                tag['qr_bbox'] = qr_bbox
        return keyframes

    # -------------------- ЭТАП 4: OCR --------------------
    def run_ocr_on_tags(self, keyframes: List[Dict]) -> List[Dict]:
        if self.ocr is None:
            print("Загружаю PaddleOCR (русский)...")
            # убрали use_angle_cls, теперь use_textline_orientation
            self.ocr = PaddleOCR(lang='ru', use_textline_orientation=True)
        print("Этап 4: распознавание текста...")
        for kf in keyframes:
            for tag in kf.get('price_tags', []):
                warped = tag.get('warped')
                if warped is None:
                    tag['ocr_text'] = []
                    continue
                # Новый API: predict вместо ocr, без cls
                result = self.ocr.predict(warped)
                lines = []
                if result and len(result) > 0:
                    res = result[0]  # одно изображение
                    if isinstance(res, dict) and 'rec_texts' in res:
                        rec_texts = res['rec_texts']
                        rec_scores = res.get('rec_scores', [])
                        dt_polys = res.get('dt_polys', [])
                        for i, text in enumerate(rec_texts):
                            conf = rec_scores[i] if i < len(rec_scores) else 0.0
                            bbox = dt_polys[i] if i < len(dt_polys) else [[0, 0], [0, 0], [0, 0], [0, 0]]
                            lines.append({'bbox': bbox, 'text': text, 'conf': conf})
                    else:
                        # на случай, если структура иная, пытаемся итерировать
                        for item in res:
                            if isinstance(item, (list, tuple)) and len(item) == 2:
                                bbox, (text, conf) = item
                                lines.append({'bbox': bbox, 'text': text, 'conf': conf})
                tag['ocr_text'] = lines
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
            cv2.imwrite(os.path.join(good_dir, fname), img)

            dark = (img * 0.3).astype(np.uint8)
            if len(kf['price_tags']) == 0:
                cv2.imwrite(os.path.join(processed_dir, fname), dark)
                continue

            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            for tag in kf['price_tags']:
                corners = np.array(tag['corners'], dtype=np.int32)
                cv2.fillPoly(mask, [corners], 255)

            processed = dark.copy()
            processed[mask == 255] = img[mask == 255]

            for tag in kf['price_tags']:
                corners = np.array(tag['corners'], dtype=np.int32)
                cv2.polylines(processed, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
                x, y = corners[0]
                cv2.putText(processed, f"conf:{tag['confidence']:.2f}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imwrite(os.path.join(processed_dir, fname), processed)
        print(f"Отладочные кадры сохранены в: {output_dir}/")

    def save_warped_debug(self, keyframes: List[Dict], output_dir='debug_warped'):
        """Сохраняет выпрямленные ценники как отдельные изображения."""
        os.makedirs(output_dir, exist_ok=True)
        for kf in keyframes:
            ts = kf['timestamp_ms']
            for i, tag in enumerate(kf.get('price_tags', [])):
                warped = tag.get('warped')
                if warped is not None:
                    fname = f"warped_{ts}_{i}.png"
                    cv2.imwrite(os.path.join(output_dir, fname), warped)

    def save_warped_qr_debug(self, keyframes: List[Dict], output_dir='debug_warped_qr'):
        """Сохраняет ценники с обведёнными QR-кодами."""
        os.makedirs(output_dir, exist_ok=True)
        for kf in keyframes:
            ts = kf['timestamp_ms']
            for i, tag in enumerate(kf.get('price_tags', [])):
                warped = tag.get('warped')
                if warped is None:
                    continue
                vis = warped.copy()
                qr_bbox = tag.get('qr_bbox')
                if qr_bbox:
                    pts = np.array(qr_bbox, dtype=np.int32)
                    cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                    qr_text = tag.get('qr_data', '')[:20]
                    cv2.putText(vis, f"QR: {qr_text}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                fname = f"warped_{ts}_{i}.png"
                cv2.imwrite(os.path.join(output_dir, fname), vis)

    # -------------------- ГЛАВНЫЙ МЕТОД (полный пайплайн) --------------------
    def run_to_csv(self, video_path: str, max_frames=50,
                   debug=True, debug_dir='debug_output',
                   csv_path='output.csv') -> List[Dict]:
        """Запускает все этапы и сохраняет CSV с распознанными данными."""
        keyframes = self.extract_keyframes(video_path, max_frames)
        keyframes = self.detect_price_tags_on_keyframes(keyframes)
        if debug:
            self.save_debug_frames(keyframes, debug_dir)

        keyframes = self.warp_price_tags(keyframes)
        keyframes = self.detect_qr_on_warped(keyframes)
        keyframes = self.run_ocr_on_tags(keyframes)

        if debug:
            self.save_warped_debug(keyframes, os.path.join(debug_dir, 'warped_tags'))
            self.save_warped_qr_debug(keyframes, os.path.join(debug_dir, 'warped_qr'))

        # Сборка CSV
        records = []
        filename = os.path.basename(video_path)
        for kf in keyframes:
            ts = kf['timestamp_ms']
            for tag in kf.get('price_tags', []):
                corners = tag['corners']
                x_min = int(min(c[0] for c in corners))
                y_min = int(min(c[1] for c in corners))
                x_max = int(max(c[0] for c in corners))
                y_max = int(max(c[1] for c in corners))

                # Сырой OCR-текст
                ocr_lines = tag.get('ocr_text', [])
                raw_text = ' '.join([line['text'] for line in ocr_lines])

                # Простейшее извлечение цены (можно будет усложнить)
                prices = re.findall(r'\b\d{1,6}[.,]\d{2}\b', raw_text)
                price_default = prices[0].replace(',', '.') if prices else ''
                price_card = prices[1].replace(',', '.') if len(prices) > 1 else ''
                discount_match = re.search(r'(\d+)\s*%', raw_text)
                discount_amount = discount_match.group(1) + '%' if discount_match else ''

                record = {
                    'filename': filename,
                    'product_name': raw_text,            # сырой текст для дальнейшего анализа
                    'price_default': price_default,
                    'price_card': price_card,
                    'price_discount': '',
                    'barcode': tag.get('qr_data', ''),   # данные QR-кода
                    'discount_amount': discount_amount,
                    'id_sku': '',
                    'print_datetime': '',
                    'code': '',
                    'additional_info': '',
                    'color': '',
                    'special_symbols': '',
                    'frame_timestamp': ts,
                    'x_min': x_min,
                    'y_min': y_min,
                    'x_max': x_max,
                    'y_max': y_max,
                }
                records.append(record)

        df = pd.DataFrame(records)
        # Гарантируем порядок и наличие колонок
        columns = [
            'filename', 'product_name', 'price_default', 'price_card', 'price_discount',
            'barcode', 'discount_amount', 'id_sku', 'print_datetime', 'code',
            'additional_info', 'color', 'special_symbols', 'frame_timestamp',
            'x_min', 'y_min', 'x_max', 'y_max'
        ]
        for col in columns:
            if col not in df.columns:
                df[col] = ''
        df = df[columns]
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"CSV сохранён: {csv_path} (строк: {len(records)})")
        return keyframes


# -------------------- Пример использования --------------------
if __name__ == "__main__":
    pipeline = PriceTagPipeline(
        detection_model_path='runs/detect/runs/detect/price_tag_v1/weights/best.pt',
        laplacian_thr=100
    )
    pipeline.run_to_csv(
        video_path='videos/43_15.mp4',
        max_frames=50,
        debug=True,
        debug_dir='debug_output',
        csv_path='result.csv'
    )
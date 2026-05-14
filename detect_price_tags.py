import os
os.environ['DYLD_LIBRARY_PATH'] = '/opt/homebrew/opt/zbar/lib'  # для pyzbar на macOS

import cv2
import numpy as np
import re
import base64
from io import BytesIO
from typing import List, Dict

import pandas as pd
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO
from skimage.metrics import structural_similarity as ssim
from paddleocr import PaddleOCR
from pyzbar.pyzbar import decode as pyzbar_decode


class PriceTagPipeline:
    def __init__(self,
                 detection_model_path: str = "yolov8n.pt",
                 detection_mode: str = "color",
                 laplacian_thr: float = 100,
                 ssim_thr: float = 0.95):
        """
        Инициализация пайплайна.
        :param detection_model_path: путь к весам YOLO (.pt) — нужен только при detection_mode="yolo"
        :param detection_mode: "color" (по умолчанию, без обученной модели) или "yolo" (обученная модель)
        :param laplacian_thr: порог дисперсии Лапласиана (резкость)
        :param ssim_thr: порог структурного сходства для дедупликации
        """
        if detection_mode not in ("color", "yolo"):
            raise ValueError(f"detection_mode должен быть 'color' или 'yolo', получено: {detection_mode}")
        self.detection_mode = detection_mode
        self.laplacian_thr = laplacian_thr
        self.ssim_thr = ssim_thr
        self.detector = None
        if self.detection_mode == "yolo":
            print(f"Загружаю модель детекции: {detection_model_path}")
            self.detector = YOLO(detection_model_path)
        else:
            print("Режим детекции: цветовой (без обученной модели)")
        self.ocr = None

    # -------------------- ЭТАП 1 --------------------
    def extract_keyframes(self, video_path: str, max_frames: int = 50) -> List[Dict]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Не удалось открыть видео: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30  # fallback, если видео не сообщает FPS

        keyframes = []
        last_gray = None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp_ms = int((frame_idx / fps) * 1000)
            # timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC)) #потом вернемся сюда если надо

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
        """Диспетчер: выбирает метод детекции по self.detection_mode."""
        if self.detection_mode == "color":
            return self._detect_price_tags_color(keyframes)
        else:
            return self._detect_price_tags_yolo(keyframes)

    def _detect_price_tags_yolo(self, keyframes: List[Dict]) -> List[Dict]:
        """Детекция ценников через YOLO. Добавляет в каждый кадр поле 'price_tags'."""
        print("Этап 2: детекция ценников (YOLO)...")
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

    # -------------------- ЭТАП 2b: ЦВЕТОВАЯ ДЕТЕКЦИЯ --------------------
    @staticmethod
    def _nms(boxes: List[tuple], iou_threshold: float = 0.3) -> List[tuple]:
        """
        Non-Maximum Suppression для слияния перекрывающихся bbox.
        boxes: список кортежей (x1, y1, x2, y2, conf)
        Возвращает отфильтрованный список.
        """
        if not boxes:
            return []
        # Сортируем по confidence (по убыванию)
        boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
        keep = []

        def iou(a, b):
            x1 = max(a[0], b[0])
            y1 = max(a[1], b[1])
            x2 = min(a[2], b[2])
            y2 = min(a[3], b[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area_a = (a[2] - a[0]) * (a[3] - a[1])
            area_b = (b[2] - b[0]) * (b[3] - b[1])
            union = area_a + area_b - inter
            return inter / union if union > 0 else 0

        while boxes:
            best = boxes.pop(0)
            keep.append(best)
            boxes = [b for b in boxes if iou(best, b) < iou_threshold]

        return keep

    def _detect_price_tags_color(self, keyframes: List[Dict],
                                  min_area: int = 500,
                                  min_tag_aspect: float = 0.25,
                                  max_tag_aspect: float = 4.0,
                                  nms_iou: float = 0.3,
                                  white_top_ratio: float = 0.15,
                                  color_bottom_ratio: float = 0.10,
                                  scale: int = 4) -> List[Dict]:
        """
        Детекция ценников по цветовому признаку (без обученной модели).
        Работает на уменьшенной копии кадра для устойчивости к шуму.

        :param min_area: минимальная площадь цветного контура (на уменьшенном изображении)
        :param min_tag_aspect: минимальный aspect ratio итогового bbox
        :param max_tag_aspect: максимальный aspect ratio итогового bbox
        :param nms_iou: порог IoU для NMS
        :param white_top_ratio: минимальная доля белых пикселей в верхней половине
        :param color_bottom_ratio: минимальная доля цветных пикселей в нижней половине
        :param scale: фактор уменьшения изображения (4 = кадр 4K → ~960x540)
        """
        print("Этап 2: детекция ценников (цветовой признак)...")

        kern_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kern_big = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

        for kf in keyframes:
            image = kf['image']
            h_img, w_img = image.shape[:2]

            # Уменьшаем изображение для устойчивости
            small = cv2.resize(image, (w_img // scale, h_img // scale))
            sh, sw = small.shape[:2]
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

            # --- Маски ---
            white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 50, 255]))
            red_mask = cv2.bitwise_or(
                cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255])),
                cv2.inRange(hsv, np.array([160, 70, 50]), np.array([180, 255, 255]))
            )
            yellow_mask = cv2.inRange(hsv, np.array([20, 70, 50]), np.array([35, 255, 255]))
            color_mask = cv2.bitwise_or(red_mask, yellow_mask)

            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kern_big)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kern_small)
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kern_small)
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kern_small)

            # --- Контуры цветных регионов ---
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            raw_boxes = []
            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)
                area = bw * bh

                if area < min_area:
                    continue

                # Ищем белую зону над цветным контуром
                expand = min(bh * 2, sh - y)
                search_top = max(0, int(y - expand))
                col_white = white_mask[search_top:y, x:x + bw]
                white_rows = np.where(col_white.sum(axis=1) > max(bw * 0.05, 1))[0]
                if len(white_rows) == 0:
                    continue

                tag_top = max(0, search_top + int(white_rows.min()))
                max_top = y - bh * 2
                if max_top > 0 and tag_top < max_top:
                    tag_top = int(max_top)
                tag_bottom = min(sh - 1, y + bh)
                tag_left = max(0, x - 2)
                tag_right = min(sw - 1, x + bw + 2)
                tag_w = tag_right - tag_left + 1
                tag_h = tag_bottom - tag_top + 1

                if tag_w < 5 or tag_h < 5:
                    continue

                # Фильтр по aspect ratio итогового bbox
                tag_aspect = tag_w / max(tag_h, 1)
                if tag_aspect < min_tag_aspect or tag_aspect > max_tag_aspect:
                    continue

                mid_y = (tag_top + tag_bottom) // 2
                bbox_white = white_mask[tag_top:tag_bottom + 1, tag_left:tag_right + 1]
                bbox_color = color_mask[tag_top:tag_bottom + 1, tag_left:tag_right + 1]

                top_half_area = (mid_y - tag_top) * tag_w
                bottom_half_area = (tag_bottom - mid_y) * tag_w

                if top_half_area == 0 or bottom_half_area == 0:
                    continue

                white_in_top = cv2.countNonZero(bbox_white[:mid_y - tag_top, :]) / top_half_area
                color_in_bottom = cv2.countNonZero(bbox_color[mid_y - tag_top:, :]) / bottom_half_area

                if white_in_top < white_top_ratio:
                    continue
                if color_in_bottom < color_bottom_ratio:
                    continue

                conf = min(white_in_top + color_in_bottom, 1.0)

                # Масштабируем обратно к оригинальному размеру
                raw_boxes.append((
                    tag_left * scale, tag_top * scale,
                    tag_right * scale, tag_bottom * scale,
                    conf
                ))

            # NMS
            kept = self._nms(raw_boxes, iou_threshold=nms_iou)

            tags = []
            for x1, y_top, x2, y_bottom, conf in kept:
                # Ограничиваем координаты рамками оригинального изображения
                x1 = min(x1, w_img - 1)
                y_top = min(y_top, h_img - 1)
                x2 = min(x2, w_img - 1)
                y_bottom = min(y_bottom, h_img - 1)
                tags.append({
                    "corners": [[x1, y_top], [x2, y_top],
                                [x2, y_bottom], [x1, y_bottom]],
                    "confidence": round(conf, 3)
                })

            kf['price_tags'] = tags

        total_tags = sum(len(kf['price_tags']) for kf in keyframes)
        print(f"Этап 2 (цветовой): найдено {total_tags} ценников")
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

                qr_data = ''
                qr_bbox = None
                # основной детектор pyzbar
                decoded_objs = pyzbar_decode(warped)
                for obj in decoded_objs:
                    if obj.type == 'QRCODE':
                        qr_data = obj.data.decode('utf-8', errors='ignore')
                        points = [(p.x, p.y) for p in obj.polygon]
                        qr_bbox = points
                        break

                # резервный детектор OpenCV
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
        """Распознаёт текст на выпрямленных ценниках с прогресс-баром."""
        if self.ocr is None:
            print("Загружаю PaddleOCR (русский)...")
            self.ocr = PaddleOCR(lang='ru', use_textline_orientation=True)

        # Соберём все ценники для обработки
        tags_to_process = []
        for kf in keyframes:
            for tag in kf.get('price_tags', []):
                if tag.get('warped') is not None:
                    tags_to_process.append(tag)
                else:
                    tag['ocr_text'] = []

        print(f"Этап 4: распознавание текста на {len(tags_to_process)} ценниках...")
        for tag in tqdm(tags_to_process, desc="OCR", unit="tag"):
            warped = tag['warped']
            result = self.ocr.predict(warped)
            lines = []
            if result and len(result) > 0:
                res = result[0]
                if isinstance(res, dict) and 'rec_texts' in res:
                    rec_texts = res['rec_texts']
                    rec_scores = res.get('rec_scores', [])
                    dt_polys = res.get('dt_polys', [])
                    for i, text in enumerate(rec_texts):
                        conf = rec_scores[i] if i < len(rec_scores) else 0.0
                        bbox = dt_polys[i] if i < len(dt_polys) else [[0, 0], [0, 0], [0, 0], [0, 0]]
                        lines.append({'bbox': bbox, 'text': text, 'conf': conf})
                else:
                    for item in res:
                        if isinstance(item, (list, tuple)) and len(item) == 2:
                            bbox, (text, conf) = item
                            lines.append({'bbox': bbox, 'text': text, 'conf': conf})
            tag['ocr_text'] = lines
        return keyframes

    # -------------------- ВИЗУАЛЬНАЯ ОТЛАДКА --------------------
    def save_debug_frames(self, keyframes: List[Dict], output_dir: str):
        """Сохраняет исходные кадры и версии с выделенными ценниками.
        Имя содержит качество резкости: frame_XXXXXX_quality=YYY.png"""
        good_dir = os.path.join(output_dir, "good_frames")
        processed_dir = os.path.join(output_dir, "good_frames_processed")
        os.makedirs(good_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)

        for kf in keyframes:
            ts = kf['timestamp_ms']
            sharp = kf['sharpness']
            img = kf['image'].copy()
            fname = f"frame_{ts:06d}_quality={sharp:.0f}.png"

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
        """Сохраняет выпрямленные ценники и добавляет в tag путь к файлу."""
        os.makedirs(output_dir, exist_ok=True)
        for kf in keyframes:
            ts = kf['timestamp_ms']
            for i, tag in enumerate(kf.get('price_tags', [])):
                warped = tag.get('warped')
                if warped is not None:
                    fname = f"warped_{ts}_{i}.png"
                    path = os.path.join(output_dir, fname)
                    cv2.imwrite(path, warped)
                    tag['warped_image_path'] = path
                else:
                    tag['warped_image_path'] = ''

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

    # -------------------- ГЛАВНЫЙ МЕТОД --------------------
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
        keyframes = self.normalize_orientation(keyframes)  # <-- вот здесь
        keyframes = self.run_ocr_on_tags(keyframes)

        if debug:
            self.save_warped_debug(keyframes, os.path.join(debug_dir, 'warped_tags'))
            self.save_warped_qr_debug(keyframes, os.path.join(debug_dir, 'warped_qr'))

        # Сборка CSV
        records = []
        filename = os.path.basename(video_path)
        for kf in keyframes:
            ts = kf['timestamp_ms']
            for tag in kf.get('price_tags', []): #основной цикл
                corners = tag['corners']
                x_min = int(min(c[0] for c in corners))
                y_min = int(min(c[1] for c in corners))
                x_max = int(max(c[0] for c in corners))
                y_max = int(max(c[1] for c in corners))

                warped = tag.get('warped')
                color = self.detect_color(warped) if warped is not None else 'unknown'
                tag['color'] = color  # сохраняем цвет в тег для parse_price_tag_fields

                # Полный сырой текст для отладки
                ocr_lines = tag.get('ocr_text', [])
                raw_text = ' '.join([line['text'] for line in ocr_lines])

                # Парсим поля с учётом цвета и расположения
                fields = self.parse_price_tag_fields(tag)
                # fields уже содержит product_name, price_default, price_card, price_discount, discount_amount

                record = {
                    'filename': filename,
                    'product_name': fields['product_name'],
                    'price_default': fields['price_default'],
                    'price_card': fields['price_card'],
                    'price_discount': fields['price_discount'],
                    'barcode': tag.get('qr_data', ''),
                    'discount_amount': fields['discount_amount'],
                    'id_sku': '',
                    'print_datetime': '',
                    'code': '',
                    'additional_info': '',
                    'color': color,
                    'special_symbols': '',
                    'frame_timestamp': ts,
                    'x_min': x_min,
                    'y_min': y_min,
                    'x_max': x_max,
                    'y_max': y_max,
                    'warped_image': tag.get('warped_image_path', ''),
                    'raw_text': raw_text,  # <-- новое поле
                }
                records.append(record)
        df = pd.DataFrame(records)
        # Порядок колонок
        columns = [
            'filename', 'product_name', 'price_default', 'price_card', 'price_discount',
            'barcode', 'discount_amount', 'id_sku', 'print_datetime', 'code',
            'additional_info', 'color', 'special_symbols', 'frame_timestamp',
            'x_min', 'y_min', 'x_max', 'y_max', 'warped_image', 'raw_text'
        ]
        for col in columns:
            if col not in df.columns:
                df[col] = ''
        df = df[columns]
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"CSV сохранён: {csv_path} (строк: {len(records)})")
        return keyframes


    # -------------------- Парсим ценник сам --------------------
    @staticmethod
    def _line_center_y(line: Dict) -> float:
        """Y-координата центра строки OCR."""
        return sum(p[1] for p in line['bbox']) / 4

    @staticmethod
    def _line_center_x(line: Dict) -> float:
        """X-координата центра строки OCR."""
        return sum(p[0] for p in line['bbox']) / 4

    @staticmethod
    def _line_height(line: Dict) -> float:
        """Высота строки OCR в пикселях."""
        return max(p[1] for p in line['bbox']) - min(p[1] for p in line['bbox'])

    @staticmethod
    def _extract_price(text: str) -> str:
        """Ищет цену в тексте. Поддерживает форматы: 319.99, 319,99, 319 99, 319."""
        # Сначала пытаемся найти цены с десятичной частью: 319.99, 319,99, 319 99
        m = re.search(r'(\d{1,6})[.,\s](\d{2})\b', text)
        if m:
            return m.group(1) + '.' + m.group(2)
        # Fallback: просто крупное число (цена без копеек, часто встречается в OCR)
        m = re.search(r'\b(\d{2,6})\b', text)
        if m:
            return m.group(1) + '.00'
        return ''

    def parse_price_tag_fields(self, tag: Dict) -> Dict:
        """
        Извлекает структурированные поля с учётом точной геометрии ценника.
        Изображение должно быть нормализовано (горизонтальный текст).

        Структура ценника «Лента» (красный):
        ┌────────────────────────────┐
        │  Название продукта   QR-код│  ← белая часть (верхние ~55%)
        │                  цена без │
        │                  скидки   │
        ├────────────────────────────┤
        │  -30%           234.99 р. │  ← красная часть (нижние ~45%)
        │  (круг)         (крупно)  │
        └────────────────────────────┘
        """
        ocr_lines = tag.get('ocr_text', [])
        qr_bbox = tag.get('qr_bbox')
        color = tag.get('color', 'unknown')
        warped = tag.get('warped')
        h, w = warped.shape[:2] if warped is not None else (0, 0)

        # Граница между белой (верх) и цветной (низ) частями
        # У ценников Лента красная нижняя часть занимает ~40-45% высоты
        split_y = int(h * 0.55)

        # Разделяем строки по Y-центру
        top_lines = [l for l in ocr_lines if self._line_center_y(l) < split_y]
        bottom_lines = [l for l in ocr_lines if self._line_center_y(l) >= split_y]

        # --- Верхняя часть ---
        # Название: левая половина верхней части
        left_top = [l for l in top_lines if self._line_center_x(l) < w / 2]
        product_name = ' '.join([l['text'] for l in left_top]).strip()

        # Цена без скидки (price_default): правая половина верхней части
        # Обычно мелкими цифрами рядом с QR-кодом
        right_top = [l for l in top_lines if self._line_center_x(l) >= w / 2]
        if qr_bbox:
            qr_bottom = max(p[1] for p in qr_bbox)
            candidates = [l for l in right_top if self._line_center_y(l) > qr_bottom]
            if not candidates:
                candidates = right_top
        else:
            candidates = right_top

        # Извлекаем цены из кандидатов
        price_default = ''
        for line in candidates:
            price = self._extract_price(line['text'])
            if price:
                price_default = price
                break

        # --- Нижняя часть (только для цветных ценников) ---
        price_discount = ''
        discount_amount = ''

        if color == 'red' and bottom_lines:
            # Скидка в круге: левая половина нижней части, содержит '%'
            left_bottom = [l for l in bottom_lines if self._line_center_x(l) < w / 2]
            discount_text = ''

            for line in left_bottom:
                if '%' in line['text']:
                    pm = re.search(r'(-?\d+)\s*%', line['text'])
                    if pm:
                        discount_text = pm.group(1) + '%'
                        break

            if not discount_text:
                # Fallback: искать минус + число (OCR иногда видит «-30» без %)
                for line in left_bottom:
                    pm = re.search(r'(-?\d+)\s*%?', line['text'])
                    if pm and pm.group(1):
                        discount_text = pm.group(1) + '%'
                        break

            if not discount_text:
                # Последний fallback: поиск по всей нижней части
                for line in bottom_lines:
                    if '%' in line['text']:
                        pm = re.search(r'(-?\d+)\s*%', line['text'])
                        if pm:
                            discount_text = pm.group(1) + '%'
                            break

            discount_amount = discount_text

            # Цена со скидкой: правая половина нижней части, крупный шрифт
            right_bottom = [l for l in bottom_lines if self._line_center_x(l) >= w / 2]
            best_price = None
            best_height = 0
            for line in right_bottom:
                lh = self._line_height(line)
                price = self._extract_price(line['text'])
                if price and lh > best_height:
                    best_height = lh
                    best_price = price

            if best_price:
                price_discount = best_price
            elif right_bottom:
                # Fallback: берём цену из любой строки справа снизу
                for line in right_bottom:
                    price = self._extract_price(line['text'])
                    if price:
                        price_discount = price
                        break

        # Для жёлтых и белых ценников — пробуем извлечь цену из всего текста
        if color in ('yellow', 'unknown') and not price_default:
            all_text = ' '.join([l['text'] for l in ocr_lines])
            price = self._extract_price(all_text)
            if price:
                price_default = price

        return {
            'product_name': product_name,
            'price_default': price_default,
            'price_card': '',  # пока не используется
            'price_discount': price_discount,
            'discount_amount': discount_amount
        }

    # -------------------- Нормализуем ориентацию --------------------
    def normalize_orientation(self, keyframes: List[Dict]) -> List[Dict]:
        """
        Если выпрямленный ценник имеет высоту > ширины, поворачивает его на 90°,
        чтобы текст всегда был горизонтальным. Пересчитывает qr_bbox.
        """
        print("Нормализация ориентации ценников...")
        for kf in keyframes:
            for tag in kf.get('price_tags', []):
                warped = tag.get('warped')
                if warped is None:
                    continue
                h, w = warped.shape[:2]
                if h > w:
                    # Поворачиваем на 90° против часовой (текст станет горизонтальным)
                    tag['warped'] = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    # Обновляем координаты QR-кода, если есть
                    if tag.get('qr_bbox') is not None:
                        pts = np.array(tag['qr_bbox'])
                        # Преобразование: (x,y) -> (y, w-1-x) для поворота против часовой
                        pts[:, [0, 1]] = pts[:, [1, 0]]  # меняем местами
                        pts[:, 0] = w - 1 - pts[:, 0]  # отражаем по горизонтали
                        tag['qr_bbox'] = pts.tolist()
        return keyframes

    # -------------------- HTML-ОТЧЁТ --------------------
    def generate_html_report(self, csv_path: str, html_path: str = 'report.html'):
        """Создаёт HTML-страницу с таблицей, где каждая строка содержит миниатюру ценника."""
        if not os.path.exists(csv_path):
            print(f"CSV файл не найден: {csv_path}")
            return
        df = pd.read_csv(csv_path)

        # Добавляем столбец с base64 изображением
        image_tags = []
        for _, row in df.iterrows():
            img_path = row.get('warped_image', '')
            if img_path and os.path.exists(img_path):
                pil_img = Image.open(img_path)
                pil_img.thumbnail((200, 200))
                buffer = BytesIO()
                pil_img.save(buffer, format='PNG')
                b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                img_tag = f'<img src="data:image/png;base64,{b64}" style="max-width:200px;">'
            else:
                img_tag = ''
            image_tags.append(img_tag)

        df.insert(0, 'image', image_tags)

        html_template = """
        <html>
        <head>
        <meta charset="utf-8">
        <style>
            table { border-collapse: collapse; }
            th, td { border: 1px solid #ccc; padding: 8px; font-size: 12px; vertical-align: top; }
            th { background: #f0f0f0; }
        </style>
        </head>
        <body>
        <h2>Результаты распознавания ценников</h2>
        {table}
        </body>
        </html>
        """
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_template.replace('{table}', df.to_html(escape=False, index=False)))
        print(f"HTML-отчёт сохранён: {html_path}")



    # -------------------- Определяем цвет ценника --------------------
    def detect_color(self, warped: np.ndarray) -> str:
        """
        Определяет доминирующий цвет ценника (red, yellow, unknown),
        игнорируя белые области.
        """
        if warped is None or warped.size == 0:
            return 'unknown'

        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

        # Маска белого: яркость > 200, насыщенность < 50
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 50, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # Маска красного (два диапазона из-за цикличности Hue)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red1, upper_red1),
            cv2.inRange(hsv, lower_red2, upper_red2)
        )

        # Маска жёлтого
        lower_yellow = np.array([20, 70, 50])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Убираем белые пиксели из цветных масок
        red_mask = cv2.bitwise_and(red_mask, cv2.bitwise_not(white_mask))
        yellow_mask = cv2.bitwise_and(yellow_mask, cv2.bitwise_not(white_mask))

        total_pixels = warped.shape[0] * warped.shape[1]
        red_pixels = cv2.countNonZero(red_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)

        min_color_ratio = 0.05  # 5% площади

        if red_pixels / total_pixels > min_color_ratio:
            return 'red'
        elif yellow_pixels / total_pixels > min_color_ratio:
            return 'yellow'
        else:
            return 'unknown'

    # -------------------- Дедупликация. Потом применю если потребуется --------------------
    def deduplicate_price_tags(self, keyframes: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
        """
        Удаляет дубликаты ценников, встречающиеся в соседних кадрах.
        Для каждой группы оставляет лучший (по резкости кадра или confidence детекции).
        Возвращает новый список keyframes, где у дублирующих тегов установлен флаг _duplicate = True.
        """
        # Соберём все детекции с информацией о кадре
        all_detections = []
        for kf_idx, kf in enumerate(keyframes):
            for tag_idx, tag in enumerate(kf.get('price_tags', [])):
                corners = tag['corners']
                x_min = min(c[0] for c in corners)
                y_min = min(c[1] for c in corners)
                x_max = max(c[0] for c in corners)
                y_max = max(c[1] for c in corners)
                all_detections.append({
                    'kf_idx': kf_idx,
                    'tag_idx': tag_idx,
                    'bbox': [x_min, y_min, x_max, y_max],
                    'sharpness': kf['sharpness'],
                    'confidence': tag['confidence'],
                    'timestamp_ms': kf['timestamp_ms']
                })

        # Простой жадный группировщик по IoU в скользящем окне
        # Сортируем по времени
        all_detections.sort(key=lambda d: d['timestamp_ms'])
        groups = []  # список групп, каждая группа - список индексов детекций
        used = set()

        def iou(bbox1, bbox2):
            # стандартный IoU
            x1 = max(bbox1[0], bbox2[0])
            y1 = max(bbox1[1], bbox2[1])
            x2 = min(bbox1[2], bbox2[2])
            y2 = min(bbox1[3], bbox2[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
            area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
            union = area1 + area2 - inter
            return inter / union if union > 0 else 0

        for i, det in enumerate(all_detections):
            if i in used:
                continue
            group = [i]
            used.add(i)
            # ищем похожие в пределах ближайших 5 секунд (или кадров)
            for j in range(i + 1, len(all_detections)):
                if j in used:
                    continue
                # проверяем, не слишком ли далеко по времени (опционально)
                if all_detections[j]['timestamp_ms'] - det['timestamp_ms'] > 5000:
                    break
                if iou(det['bbox'], all_detections[j]['bbox']) > iou_threshold:
                    group.append(j)
                    used.add(j)
            groups.append(group)

        # В каждой группе выбираем лучший (по сумме sharpness+confidence*100)
        best_per_group = []
        for group in groups:
            best_idx = max(group,
                           key=lambda idx: all_detections[idx]['sharpness'] + all_detections[idx]['confidence'] * 100)
            best_per_group.append(all_detections[best_idx])

        # Помечаем все детекции, которые не являются лучшими в группе, как дубликаты
        best_set = set((d['kf_idx'], d['tag_idx']) for d in best_per_group)
        for kf in keyframes:
            for tag in kf.get('price_tags', []):
                tag['_duplicate'] = True  # по умолчанию все дубликаты
        for det in best_per_group:
            keyframes[det['kf_idx']]['price_tags'][det['tag_idx']]['_duplicate'] = False

        print(f"Дедупликация: оставлено {len(best_per_group)} ценников из {len(all_detections)} первоначальных")
        return keyframes

# -------------------- Пример использования --------------------
if __name__ == "__main__":
    pipeline = PriceTagPipeline(
        detection_mode='color',  # "color" (по умолчанию, без модели) или "yolo" (обученная модель)
        # detection_model_path='runs/detect/runs/detect/price_tag_v1/weights/best.pt',  # нужен только для yolo
        laplacian_thr=100   # порог чёткости, можно менять
    )
    pipeline.run_to_csv(
        video_path='videos/43_15.mp4',
        max_frames=50,
        debug=True,
        debug_dir='debug_output',
        csv_path='result.csv'
    )
    pipeline.generate_html_report('result.csv', 'report.html')
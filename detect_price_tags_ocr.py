import os
os.environ['DYLD_LIBRARY_PATH'] = '/opt/homebrew/opt/zbar/lib'

import cv2
import numpy as np
import re
import imagehash
import base64
from io import BytesIO
from typing import List, Dict, Callable, Optional, Tuple
from collections import defaultdict

import pandas as pd
from tqdm import tqdm
from PIL import Image
from paddleocr import PaddleOCR
from pyzbar.pyzbar import decode as pyzbar_decode
from sklearn.cluster import DBSCAN


EAST_INPUT_SIZE = 960


class TextBasedPriceTagPipeline:
    def __init__(self,
                 east_model_path: str = "models/frozen_east_text_detection.pb",
                 assume_99_kopecks: bool = True,
                 candidate_score_threshold: float = 2.0,
                 dbscan_eps_factor: float = 2.0,
                 qr_expand_ratio: float = 2.5,
                 horizontal_expand: float = 0.5,
                 vertical_expand: float = 0.3,
                 simple_rotate: bool = False,
                 trim_method: str = 'aspect'):
        self.assume_99_kopecks = assume_99_kopecks
        self.candidate_score_threshold = candidate_score_threshold
        self.dbscan_eps_factor = dbscan_eps_factor
        self.qr_expand_ratio = qr_expand_ratio
        self.horizontal_expand = horizontal_expand
        self.vertical_expand = vertical_expand
        self.simple_rotate = simple_rotate
        self.trim_method = trim_method
        self.ocr = None

        print(f"Загружаю EAST модель: {east_model_path}")
        if not os.path.exists(east_model_path):
            raise FileNotFoundError(
                f"EAST модель не найдена: {east_model_path}\n"
                "Скачайте: curl -L -o models/frozen_east_text_detection.pb "
                "https://raw.githubusercontent.com/oyyd/frozen_east_text_detection.pb/master/frozen_east_text_detection.pb"
            )
        self.east_net = cv2.dnn.readNet(east_model_path)
        print("EAST модель загружена")

    # ================================================================
    #  ЭТАП 1: Выбор ключевых кадров
    # ================================================================
    def _sample_keyframes(self, video_path: str,
                          sharpness_threshold: float = 50.0,
                          max_keyframes: int = 50) -> List[Dict]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Не удалось открыть видео: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Этап 1: выбор ключевых кадров ({total_frames} кадров, fps={fps:.1f})...")

        prev_gray = None
        sharp_frames = []

        for frame_idx in tqdm(range(total_frames), desc="Сканирование кадров"):
            ret, frame = cap.read()
            if not ret:
                break

            if self.simple_rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

            if sharpness < sharpness_threshold:
                prev_gray = gray
                continue

            diff_score = 0.0
            if prev_gray is not None:
                diff = cv2.absdiff(gray, cv2.resize(prev_gray, (gray.shape[1], gray.shape[0])))
                diff_score = np.mean(diff)
            prev_gray = gray

            timestamp_ms = int((frame_idx / fps) * 1000)
            sharp_frames.append({
                'frame_idx': frame_idx,
                'timestamp_ms': timestamp_ms,
                'sharpness': sharpness,
                'diff_score': diff_score,
                'frame': frame,
            })

        cap.release()

        if not sharp_frames:
            print("  Не найдено чётких кадров")
            return []

        diffs = [f['diff_score'] for f in sharp_frames]
        median_diff = np.median(diffs) if diffs else 0

        stable = [f for f in sharp_frames if f['diff_score'] < median_diff * 2.0]
        if not stable:
            stable = sharp_frames

        stable.sort(key=lambda f: (-f['sharpness'], f['frame_idx']))
        step = max(1, len(stable) // max_keyframes)
        selected = stable[::step][:max_keyframes]
        selected.sort(key=lambda f: f['frame_idx'])

        for f in selected:
            f['image'] = f.pop('frame')

        print(f"  Отобрано {len(selected)} ключевых кадров")
        return selected

    # ================================================================
    #  ЭТАП 2: Детекция текста (EAST) и QR
    # ================================================================
    def _detect_east(self, image: np.ndarray) -> List[Dict]:
        h, w = image.shape[:2]
        new_w = max(32, (min(w, EAST_INPUT_SIZE) // 32) * 32)
        new_h = max(32, (min(h, EAST_INPUT_SIZE * 2) // 32) * 32)
        scale_x = w / new_w
        scale_y = h / new_h

        blob = cv2.dnn.blobFromImage(
            image, scalefactor=1.0, size=(new_w, new_h),
            mean=(123.68, 116.78, 103.94), swapRB=True, crop=False
        )
        self.east_net.setInput(blob)
        scores, geometry = self.east_net.forward([
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"
        ])

        boxes = []
        conf_threshold = 0.3
        rows, cols = scores.shape[2:4]

        for y in range(rows):
            for x in range(cols):
                score = scores[0, 0, y, x]
                if score < conf_threshold:
                    continue

                offset_x = x * 4.0
                offset_y = y * 4.0
                angle = geometry[0, 4, y, x]
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)

                h_top = geometry[0, 0, y, x]
                h_right = geometry[0, 1, y, x]
                h_bottom = geometry[0, 2, y, x]
                h_left = geometry[0, 3, y, x]

                end_x = offset_x + (cos_a * h_right + sin_a * h_bottom)
                end_y = offset_y + (-sin_a * h_right + cos_a * h_bottom)
                start_x = offset_x - (cos_a * h_left + sin_a * h_top)
                start_y = offset_y - (-sin_a * h_left + cos_a * h_top)

                cx = (start_x + end_x) / 2.0 * scale_x
                cy = (start_y + end_y) / 2.0 * scale_y
                bw = (end_x - start_x) * scale_x
                bh = (end_y - start_y) * scale_y

                if bw > 5 and bh > 5:
                    boxes.append({
                        'bbox': [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2],
                        'center': (cx, cy),
                        'size': (bw, bh),
                        'angle': float(angle),
                        'score': float(score),
                    })

        return self._nms_boxes(boxes, iou_threshold=0.3)

    @staticmethod
    def _nms_boxes(boxes: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
        if not boxes:
            return boxes
        bboxes = np.array([b['bbox'] for b in boxes])
        scores_arr = np.array([b['score'] for b in boxes])
        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores_arr.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        return [boxes[i] for i in keep]

    def _detect_qr_codes(self, image: np.ndarray) -> List[Dict]:
        results = []
        for obj in pyzbar_decode(image):
            if obj.type == 'QRCODE':
                points = [(p.x, p.y) for p in obj.polygon]
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                results.append({
                    'data': obj.data.decode('utf-8', errors='ignore'),
                    'bbox': [min(xs), min(ys), max(xs), max(ys)],
                    'center': (float(np.mean(xs)), float(np.mean(ys))),
                    'polygon': points,
                })

        if not results:
            detector = cv2.QRCodeDetector()
            data, bbox_pts, _ = detector.detectAndDecode(image)
            if data and bbox_pts is not None:
                pts = bbox_pts.reshape(4, 2)
                xs, ys = pts[:, 0], pts[:, 1]
                results.append({
                    'data': data,
                    'bbox': [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))],
                    'center': (float(np.mean(xs)), float(np.mean(ys))),
                    'polygon': pts.tolist(),
                })

        return results

    def _detect_text_and_qr(self, keyframes: List[Dict]) -> List[Dict]:
        print("Этап 2: детекция текста (EAST) и QR-кодов...")
        for kf in tqdm(keyframes, desc="EAST+QR", unit="frame"):
            kf['text_boxes'] = self._detect_east(kf['image'])
            kf['qr_codes'] = self._detect_qr_codes(kf['image'])
        total_tb = sum(len(kf['text_boxes']) for kf in keyframes)
        total_qr = sum(len(kf['qr_codes']) for kf in keyframes)
        print(f"  Найдено текстовых bbox: {total_tb}, QR-кодов: {total_qr}")
        return keyframes

    # ================================================================
    #  ЭТАП 3: Генерация кандидатов в ценники
    # ================================================================
    @staticmethod
    def _has_price_pattern(text: str) -> bool:
        if re.search(r'\d{1,6}[.,\s]\d{2}\b', text):
            return True
        if re.search(r'\b\d{2,6}[.,]\d{2}\b', text):
            return True
        return False

    @staticmethod
    def _has_percent_pattern(text: str) -> bool:
        return bool(re.search(r'\d+\s*%', text))

    @staticmethod
    def _has_cyrillic(text: str) -> bool:
        return bool(re.search(r'[а-яА-ЯёЁ]', text))

    @staticmethod
    def _extract_price(text: str) -> str:
        m = re.search(r'(\d{1,6})[.,\s](\d{2})\b', text)
        if m:
            return m.group(1) + '.' + m.group(2)
        m = re.search(r'\b(\d{2,6})\b', text)
        if m:
            return m.group(1) + '.00'
        return ''

    def _score_cluster(self, boxes: List[Dict], has_qr: bool) -> float:
        score = 0.0
        if len(boxes) >= 3:
            score += 2.0
        elif len(boxes) >= 2:
            score += 1.0
        if has_qr:
            score += 3.0

        if boxes:
            areas = [b['size'][0] * b['size'][1] for b in boxes]
            area_std = np.std(areas) / (np.mean(areas) + 1e-6)
            if area_std < 1.0:
                score += 1.0

        if len(boxes) >= 2:
            centers_y = [b['center'][1] for b in boxes]
            centers_x = [b['center'][0] for b in boxes]
            y_range = max(centers_y) - min(centers_y)
            x_range = max(centers_x) - min(centers_x)
            if x_range > 0 and y_range > 0:
                aspect = max(x_range, y_range) / min(x_range, y_range)
                if aspect < 8.0:
                    score += 1.0

        return score

    def _cluster_text_boxes(self, boxes: List[Dict]) -> List[List[Dict]]:
        if len(boxes) < 2:
            return [boxes] if boxes else []

        heights = [b['size'][1] for b in boxes]
        median_h = np.median(heights) if heights else 20
        eps = median_h * self.dbscan_eps_factor

        centers = np.array([b['center'] for b in boxes])
        clustering = DBSCAN(eps=eps, min_samples=2).fit(centers)
        labels = clustering.labels_

        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            if label >= 0:
                clusters[label].append(boxes[i])

        noise = [boxes[i] for i, label in enumerate(labels) if label < 0]
        for box in noise:
            assigned = False
            bx, by = box['center']
            for label, cluster in clusters.items():
                for cb in cluster:
                    cx, cy = cb['center']
                    if np.sqrt((bx - cx) ** 2 + (by - cy) ** 2) < eps:
                        clusters[label].append(box)
                        assigned = True
                        break
                if assigned:
                    break

        merged = self._merge_vertically_aligned_clusters(list(clusters.values()), median_h)
        return merged

    @staticmethod
    def _merge_vertically_aligned_clusters(clusters: List[List[Dict]],
                                           median_h: float) -> List[List[Dict]]:
        if len(clusters) <= 1:
            return clusters

        cluster_bboxes = []
        for cluster in clusters:
            x1 = min(b['bbox'][0] for b in cluster)
            y1 = min(b['bbox'][1] for b in cluster)
            x2 = max(b['bbox'][2] for b in cluster)
            y2 = max(b['bbox'][3] for b in cluster)
            cluster_bboxes.append([x1, y1, x2, y2])

        n = len(clusters)
        merged_into = list(range(n))

        def find(x):
            while merged_into[x] != x:
                merged_into[x] = merged_into[merged_into[x]]
                x = merged_into[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                merged_into[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                ax1, ay1, ax2, ay2 = cluster_bboxes[i]
                bx1, by1, bx2, by2 = cluster_bboxes[j]

                overlap_x = max(0, min(ax2, bx2) - max(ax1, bx1))
                min_w = min(ax2 - ax1, bx2 - bx1)
                x_overlap_ratio = overlap_x / min_w if min_w > 0 else 0

                gap_y = max(0, max(ay1, by1) - min(ay2, by2))

                if x_overlap_ratio > 0.3 and gap_y < median_h * 5:
                    union(i, j)

        groups = defaultdict(list)
        for i in range(n):
            groups[find(i)].extend(clusters[i])

        return list(groups.values())

    @staticmethod
    def _bbox_for_boxes(boxes: List[Dict]) -> List[float]:
        if not boxes:
            return [0, 0, 0, 0]
        return [
            min(b['bbox'][0] for b in boxes),
            min(b['bbox'][1] for b in boxes),
            max(b['bbox'][2] for b in boxes),
            max(b['bbox'][3] for b in boxes),
        ]

    def _expand_bbox(self, bbox: List[float], image_shape: Tuple[int, ...],
                     ratio: float = 1.0) -> List[float]:
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        if ratio > 1:
            pad_x = max(w * ratio, w * self.qr_expand_ratio) - w
            pad_y = max(h * ratio, h * self.qr_expand_ratio) - h
        else:
            pad_x = w * self.horizontal_expand
            pad_y = h * self.vertical_expand
        img_h, img_w = image_shape[:2]
        return [
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(img_w, x2 + pad_x),
            min(img_h, y2 + pad_y),
        ]

    @staticmethod
    def _iou(a: List[float], b: List[float]) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    def _generate_candidates(self, keyframes: List[Dict]) -> List[Dict]:
        print("Этап 3: генерация кандидатов в ценники...")
        total_candidates = 0

        for kf in keyframes:
            image = kf['image']
            text_boxes = kf.get('text_boxes', [])
            qr_codes = kf.get('qr_codes', [])
            candidates = []

            qr_bbox_ids = set()

            for qr in qr_codes:
                expanded = self._expand_bbox(qr['bbox'], image.shape, ratio=self.qr_expand_ratio)
                nearby = [tb for tb in text_boxes
                          if expanded[0] <= tb['center'][0] <= expanded[2]
                          and expanded[1] <= tb['center'][1] <= expanded[3]]

                if nearby:
                    qr_as_box = {
                        'bbox': qr['bbox'],
                        'center': qr['center'],
                        'size': (qr['bbox'][2] - qr['bbox'][0], qr['bbox'][3] - qr['bbox'][1]),
                    }
                    cand_bbox = self._bbox_for_boxes(nearby + [qr_as_box])
                    score = self._score_cluster(nearby, has_qr=True)
                    candidates.append({
                        'bbox': cand_bbox,
                        'score': score,
                        'source': 'qr_anchor',
                        'boxes': nearby,
                        'qr_data': qr['data'],
                        'qr_bbox': qr['bbox'],
                    })
                    for tb in nearby:
                        qr_bbox_ids.add(id(tb))

            clusters = self._cluster_text_boxes(text_boxes)

            for cluster in clusters:
                cluster_ids = {id(b) for b in cluster}
                if cluster_ids & qr_bbox_ids and len(cluster) < len(text_boxes):
                    continue

                cand_bbox = self._bbox_for_boxes(cluster)
                has_nearby_qr = False
                qr_data = ''
                qr_bbox = None
                for qr in qr_codes:
                    qr_cx, qr_cy = qr['center']
                    if (cand_bbox[0] - 50 <= qr_cx <= cand_bbox[2] + 50 and
                            cand_bbox[1] - 50 <= qr_cy <= cand_bbox[3] + 50):
                        has_nearby_qr = True
                        qr_data = qr['data']
                        qr_bbox = qr['bbox']
                        break

                score = self._score_cluster(cluster, has_qr=has_nearby_qr)

                if score >= self.candidate_score_threshold:
                    candidates.append({
                        'bbox': cand_bbox,
                        'score': score,
                        'source': 'text_cluster',
                        'boxes': cluster,
                        'qr_data': qr_data,
                        'qr_bbox': qr_bbox,
                    })

            merged = []
            candidates.sort(key=lambda c: -c['score'])
            img_h, img_w = image.shape[:2]
            img_area = img_h * img_w
            for cand in candidates:
                bbox = cand['bbox']
                cand_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if cand_area < 500:
                    continue
                if cand_area > img_area * 0.5:
                    continue
                bw = bbox[2] - bbox[0]
                bh = bbox[3] - bbox[1]
                if bw < 80 or bh < 100:
                    continue
                aspect = bw / bh if bh > 0 else 0
                if aspect < 0.3 or aspect > 3.0:
                    continue
                is_dup = False
                for i, existing in enumerate(merged):
                    if self._iou(cand['bbox'], existing['bbox']) > 0.4:
                        if cand['score'] > existing['score']:
                            merged[i] = cand
                        is_dup = True
                        break
                if not is_dup:
                    merged.append(cand)

            merged = merged[:10]
            kf['candidates'] = merged
            total_candidates += len(merged)

        print(f"  Найдено кандидатов: {total_candidates}")
        return keyframes

    # ================================================================
    #  ЭТАП 4: Уточнение границ ценника
    # ================================================================
    def _refine_boundaries(self, keyframes: List[Dict]) -> List[Dict]:
        print("Этап 4: уточнение границ ценников...")
        for kf in keyframes:
            image = kf['image']
            for cand in kf.get('candidates', []):
                refined = self._find_rect_boundary(image, cand['bbox'])
                cand['bbox_refined'] = refined

                x1, y1, x2, y2 = [int(v) for v in refined]
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(image.shape[1], x2)
                y2 = min(image.shape[0], y2)

                if x2 - x1 < 20 or y2 - y1 < 20:
                    cand['crop'] = None
                    continue

                cand['crop'] = image[y1:y2, x1:x2].copy()
                cand['crop_bbox'] = [x1, y1, x2, y2]

        return keyframes

    def _find_rect_boundary(self, image: np.ndarray, text_bbox: List[float]) -> List[float]:
        x1, y1, x2, y2 = text_bbox
        img_h, img_w = image.shape[:2]

        pad_x = (x2 - x1) * self.horizontal_expand
        pad_y = (y2 - y1) * self.horizontal_expand
        search_x1 = max(0, int(x1 - pad_x * 2))
        search_y1 = max(0, int(y1 - pad_y * 2))
        search_x2 = min(img_w, int(x2 + pad_x * 2))
        search_y2 = min(img_h, int(y2 + pad_y * 2))

        roi = image[search_y1:search_y2, search_x1:search_x2]
        if roi.size == 0:
            return self._expand_bbox(text_bbox, image.shape)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        text_cx = (x1 + x2) / 2 - search_x1
        text_cy = (y1 + y2) / 2 - search_y1
        text_area = (x2 - x1) * (y2 - y1)

        best_contour = None
        best_score = -1

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < text_area * 0.3 or area > text_area * 10:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) < 4 or len(approx) > 6:
                continue
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            if rw < 30 or rh < 20:
                continue
            cx_cnt = rx + rw / 2
            cy_cnt = ry + rh / 2
            dist = np.sqrt((cx_cnt - text_cx) ** 2 + (cy_cnt - text_cy) ** 2)
            max_dist = np.sqrt(text_area)
            rectness = 4.0 / len(approx)
            proximity = 1.0 - min(dist / (max_dist + 1e-6), 1.0)
            coverage = min(area / (text_area + 1e-6), 3.0) / 3.0
            score = rectness * 0.3 + proximity * 0.4 + coverage * 0.3
            if score > best_score:
                best_score = score
                best_contour = approx

        if best_contour is not None and best_score > 0.2:
            rx, ry, rw, rh = cv2.boundingRect(best_contour)
            return [rx + search_x1, ry + search_y1, rx + rw + search_x1, ry + rh + search_y1]

        return self._expand_bbox(text_bbox, image.shape)

    # ================================================================
    #  ЭТАП 4b: Обрезка избыточной вертикали
    # ================================================================
    def _trim_vertical_excess(self, keyframes: List[Dict]) -> List[Dict]:
        print(f"Этап 4b: обрезка вертикали (метод={self.trim_method})...")
        trimmed_count = 0
        for kf in keyframes:
            image = kf['image']
            for cand in kf.get('candidates', []):
                crop = cand.get('crop')
                if crop is None:
                    continue
                h, w = crop.shape[:2]
                aspect = w / h if h > 0 else 999
                if aspect >= 0.6:
                    continue

                if self.trim_method == 'projection':
                    new_y2 = self._trim_by_projection(crop, cand)
                else:
                    new_y2 = self._trim_by_aspect(crop, cand)

                if new_y2 is not None and new_y2 < h:
                    bbox = cand.get('crop_bbox', [0, 0, w, h])
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    new_y2_abs = y1 + new_y2
                    new_y2_abs = min(new_y2_abs, image.shape[0])
                    cand['crop'] = image[y1:new_y2_abs, x1:x2].copy()
                    cand['crop_bbox'] = [x1, y1, x2, new_y2_abs]
                    trimmed_count += 1

        print(f"  Обрезано по вертикали: {trimmed_count} ценников")
        return keyframes

    @staticmethod
    def _trim_by_aspect(crop: np.ndarray, cand: Dict) -> Optional[int]:
        h, w = crop.shape[:2]
        max_h = int(w / 0.7)
        if h <= max_h:
            return None
        return max_h

    @staticmethod
    def _trim_by_projection(crop: np.ndarray, cand: Dict) -> Optional[int]:
        h, w = crop.shape[:2]
        if h < 10 or w < 10:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inv = 255 - binary

        projection = np.sum(inv, axis=1)
        max_proj = np.max(projection) if np.max(projection) > 0 else 1
        projection_norm = projection / max_proj

        text_boxes = cand.get('boxes', [])
        if text_boxes:
            bbox = cand.get('crop_bbox', [0, 0, w, h])
            offset_y = bbox[1]
            text_bottom = max(b['bbox'][3] for b in text_boxes) - offset_y
            search_start = min(int(text_bottom + (text_bottom * 0.2)), h - 1)
        else:
            search_start = h // 3

        gap_threshold = 0.03
        min_gap_height = max(3, h // 40)

        trim_y = None
        in_gap = False
        gap_start = search_start

        for y in range(search_start, h):
            if projection_norm[y] < gap_threshold:
                if not in_gap:
                    gap_start = y
                    in_gap = True
            else:
                if in_gap:
                    gap_len = y - gap_start
                    if gap_len >= min_gap_height:
                        trim_y = gap_start
                        break
                    in_gap = False

        if trim_y is not None:
            pad = max(5, int(h * 0.03))
            trim_y = min(trim_y + pad, h)

        max_h = int(w / 0.5)
        if trim_y is None or trim_y > max_h:
            trim_y = max_h

        return trim_y if trim_y < h else None

    # ================================================================
    #  ЭТАП 5: Дедупликация между кадрами
    # ================================================================
    def _deduplicate_across_frames(self, keyframes: List[Dict]) -> List[Dict]:
        print("Этап 5: дедупликация ценников между кадрами...")

        all_tags = []
        for kf in keyframes:
            for cand in kf.get('candidates', []):
                if cand.get('crop') is None:
                    continue
                all_tags.append({
                    'timestamp_ms': kf['timestamp_ms'],
                    'candidate': cand,
                    'image': kf['image'],
                })

        if not all_tags:
            print("  Нет валидных кандидатов для дедупликации")
            return []

        hashes = []
        for tag in all_tags:
            crop = tag['candidate']['crop']
            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            hashes.append(imagehash.phash(pil_img, hash_size=16))

        groups = []
        assigned = set()
        for i in range(len(all_tags)):
            if i in assigned:
                continue
            group = [i]
            assigned.add(i)
            for j in range(i + 1, len(all_tags)):
                if j in assigned:
                    continue
                if hashes[i] - hashes[j] <= 25:
                    group.append(j)
                    assigned.add(j)
            groups.append(group)

        result_tags = []
        for group in groups:
            best_idx = max(group, key=lambda idx: all_tags[idx]['candidate'].get('score', 0))
            result_tags.append(all_tags[best_idx])

        print(f"  Дедупликация: {len(all_tags)} -> {len(result_tags)} уникальных ценников")

        deduped = []
        for tag in result_tags:
            deduped.append({
                'timestamp_ms': tag['timestamp_ms'],
                'image': tag['image'],
                'price_tags': [tag['candidate']],
            })
        return deduped

    # ================================================================
    #  ЭТАП 6: Нормализация ориентации
    # ================================================================
    def _normalize_orientation(self, keyframes: List[Dict]) -> List[Dict]:
        print("Этап 6: нормализация ориентации...")
        for kf in keyframes:
            for tag in kf.get('price_tags', []):
                crop = tag.get('crop')
                if crop is None:
                    continue

                h, w = crop.shape[:2]
                text_boxes = tag.get('boxes', [])
                qr_bbox = tag.get('qr_bbox')

                if h > w * 1.5:
                    if qr_bbox:
                        qr_cx = (qr_bbox[0] + qr_bbox[2]) / 2
                        centers_x = [b['center'][0] for b in text_boxes] if text_boxes else [0]
                        text_cx = np.mean(centers_x)
                        if qr_cx < text_cx:
                            tag['crop'] = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        else:
                            tag['crop'] = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
                    else:
                        tag['crop'] = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)

                color = self._detect_color(tag.get('crop'))
                tag['color'] = color

                if color in ('red', 'yellow') and tag.get('crop') is not None:
                    tag['crop'] = self._orient_colored_tag(tag['crop'], color)

        return keyframes

    def _orient_colored_tag(self, crop: np.ndarray, color: str) -> np.ndarray:
        h, w = crop.shape[:2]
        if h < 4 or w < 4:
            return crop

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        if color == 'red':
            mask = cv2.bitwise_or(
                cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255])),
                cv2.inRange(hsv, np.array([160, 70, 50]), np.array([180, 255, 255]))
            )
        else:
            mask = cv2.inRange(hsv, np.array([20, 70, 50]), np.array([35, 255, 255]))

        top_half = mask[:h // 2, :]
        bottom_half = mask[h // 2:, :]
        top_score = cv2.countNonZero(top_half) / max(top_half.size, 1)
        bot_score = cv2.countNonZero(bottom_half) / max(bottom_half.size, 1)

        if top_score > bot_score:
            return cv2.rotate(crop, cv2.ROTATE_180)
        return crop

    # ================================================================
    #  ЭТАП 7: OCR на кропах
    # ================================================================
    def _run_ocr_on_crops(self, keyframes: List[Dict],
                          progress_callback: Optional[Callable] = None) -> List[Dict]:
        if self.ocr is None:
            print("Загружаю PaddleOCR (русский)...")
            self.ocr = PaddleOCR(lang='ru', use_textline_orientation=True)

        tags_to_process = []
        for kf in keyframes:
            for tag in kf.get('price_tags', []):
                if tag.get('crop') is not None:
                    tags_to_process.append(tag)
                else:
                    tag['ocr_text'] = []

        total = len(tags_to_process)
        print(f"Этап 7: OCR на {total} кропах...")

        for i, tag in enumerate(tqdm(tags_to_process, desc="OCR", unit="tag")):
            result = self.ocr.predict(tag['crop'])
            lines = []
            if result and len(result) > 0:
                res = result[0]
                if isinstance(res, dict) and 'rec_texts' in res:
                    rec_texts = res['rec_texts']
                    rec_scores = res.get('rec_scores', [])
                    dt_polys = res.get('dt_polys', [])
                    for j, text in enumerate(rec_texts):
                        conf = rec_scores[j] if j < len(rec_scores) else 0.0
                        bbox = dt_polys[j] if j < len(dt_polys) else [[0, 0], [0, 0], [0, 0], [0, 0]]
                        lines.append({'bbox': bbox, 'text': text, 'conf': conf})
                else:
                    for item in res:
                        if isinstance(item, (list, tuple)) and len(item) == 2:
                            bbox, (text, conf) = item
                            lines.append({'bbox': bbox, 'text': text, 'conf': conf})

            tag['ocr_text'] = lines
            if progress_callback and total > 0:
                progress_callback(f"OCR: {i + 1}/{total}", 0.6 + 0.3 * (i + 1) / total)

        return keyframes

    # ================================================================
    #  ЭТАП 8: Контентный парсинг полей
    # ================================================================
    @staticmethod
    def _line_center_y(line: Dict) -> float:
        bbox = line.get('bbox', [[0, 0]] * 4)
        if isinstance(bbox, (list, np.ndarray)) and len(bbox) > 0:
            if isinstance(bbox[0], (list, np.ndarray)):
                return sum(p[1] for p in bbox) / len(bbox)
        return 0.0

    @staticmethod
    def _line_center_x(line: Dict) -> float:
        bbox = line.get('bbox', [[0, 0]] * 4)
        if isinstance(bbox, (list, np.ndarray)) and len(bbox) > 0:
            if isinstance(bbox[0], (list, np.ndarray)):
                return sum(p[0] for p in bbox) / len(bbox)
        return 0.0

    @staticmethod
    def _line_height(line: Dict) -> float:
        bbox = line.get('bbox', [[0, 0]] * 4)
        if isinstance(bbox, (list, np.ndarray)) and len(bbox) > 0:
            if isinstance(bbox[0], (list, np.ndarray)):
                ys = [p[1] for p in bbox]
                return max(ys) - min(ys)
        return 0.0

    def _parse_fields_content_based(self, tag: Dict) -> Dict:
        ocr_lines = tag.get('ocr_text', [])
        color = tag.get('color', 'unknown')
        crop = tag.get('crop')
        h, w = crop.shape[:2] if crop is not None else (0, 0)

        all_texts = [ln['text'] for ln in ocr_lines]
        all_text_joined = ' '.join(all_texts)

        product_name = ''
        price_default = ''
        price_card = ''
        price_discount = ''
        discount_amount = ''
        print_datetime = ''
        id_sku = ''
        code = ''
        additional_info = ''
        special_symbols = ''

        non_numeric_lines = []
        price_lines = []
        for line in ocr_lines:
            text = line['text'].strip()
            if self._has_price_pattern(text):
                price_lines.append(line)
            if re.search(r'[а-яА-ЯёЁa-zA-Z]{2,}', text):
                non_numeric_lines.append(line)

        if non_numeric_lines:
            product_name = ' '.join(
                ln['text'].strip() for ln in non_numeric_lines
                if not self._has_price_pattern(ln['text'])
                and not self._has_percent_pattern(ln['text'])
            ).strip()
            product_name = re.sub(r'\s+', ' ', product_name)

        if not product_name and non_numeric_lines:
            product_name = ' '.join(ln['text'].strip() for ln in non_numeric_lines).strip()
            product_name = re.sub(r'\s+', ' ', product_name)

        dt_match = re.search(
            r'(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\s*(\d{1,2}:\d{2})?',
            all_text_joined
        )
        if dt_match:
            print_datetime = dt_match.group(0).strip()

        sku_match = re.search(r'\b(\d{6,12})\b', all_text_joined)
        if sku_match:
            num = int(sku_match.group(1))
            if num < 1000000:
                id_sku = sku_match.group(1)

        code_match = re.search(r'\b([A-ZА-Я]{1,3}-?\d{1,4})\b', all_text_joined)
        if code_match:
            code = code_match.group(1)

        percent_lines = [ln for ln in ocr_lines if self._has_percent_pattern(ln['text'])]
        if percent_lines:
            for line in percent_lines:
                pm = re.search(r'(-?\d+)\s*%', line['text'])
                if pm:
                    val = int(pm.group(1))
                    if abs(val) <= 100:
                        discount_amount = f'-{abs(val)}%'
                        break

        if h > 0 and w > 0 and color == 'red':
            split_y = int(h * 0.55)
            top_lines = [ln for ln in ocr_lines if self._line_center_y(ln) < split_y]
            bottom_lines = [ln for ln in ocr_lines if self._line_center_y(ln) >= split_y]

            right_top = [ln for ln in top_lines if self._line_center_x(ln) >= w / 2]
            for ln in right_top:
                price = self._extract_price(ln['text'])
                if price:
                    price_default = price
                    break

            if bottom_lines:
                right_bottom = [ln for ln in bottom_lines if self._line_center_x(ln) >= w / 2]
                best_price = None
                best_height = 0
                for ln in right_bottom:
                    lh = self._line_height(ln)
                    price = self._extract_price(ln['text'])
                    if price and lh > best_height:
                        best_height = lh
                        best_price = price
                if best_price:
                    price_discount = best_price

        elif h > 0 and w > 0 and color in ('yellow', 'white', 'unknown'):
            if price_lines:
                prices = []
                for ln in price_lines:
                    price = self._extract_price(ln['text'])
                    if price:
                        prices.append((price, self._line_height(ln), ln))
                if len(prices) >= 2:
                    prices.sort(key=lambda p: -p[1])
                    price_discount = prices[0][0]
                    price_default = prices[-1][0]
                elif len(prices) == 1:
                    price_default = prices[0][0]

        if not price_default and price_lines:
            for ln in price_lines:
                price = self._extract_price(ln['text'])
                if price:
                    price_default = price
                    break

        card_kw = re.search(r'(?:по\s*карт|карт[аоеы]|лен\.?\s*карт)', all_text_joined, re.IGNORECASE)
        if card_kw and price_lines:
            for ln in price_lines:
                text = ln['text'].lower()
                if 'карт' in text or 'card' in text:
                    price_card = self._extract_price(ln['text'])
                    break

        special_match = re.search(r'[★☆◆◇▲►◀◀●○◎□■∎]', all_text_joined)
        if special_match:
            special_symbols = special_match.group(0)

        remaining = []
        for ln in ocr_lines:
            text = ln['text'].strip()
            if text in (product_name, price_default, price_card, price_discount,
                        discount_amount, print_datetime, id_sku, code, special_symbols):
                continue
            if text and len(text) > 2:
                remaining.append(text)
        if remaining:
            additional_info = ' '.join(remaining[:3])

        if self.assume_99_kopecks:
            if price_default.endswith('.00'):
                price_default = price_default[:-3] + '.99'
            if price_discount.endswith('.00'):
                price_discount = price_discount[:-3] + '.99'
            if price_card.endswith('.00'):
                price_card = price_card[:-3] + '.99'

        if price_default and price_discount:
            try:
                if float(price_default) < float(price_discount):
                    price_default = ''
            except ValueError:
                pass

        return {
            'product_name': product_name,
            'price_default': price_default,
            'price_card': price_card,
            'price_discount': price_discount,
            'discount_amount': discount_amount,
            'id_sku': id_sku,
            'print_datetime': print_datetime,
            'code': code,
            'additional_info': additional_info,
            'special_symbols': special_symbols,
        }

    # ================================================================
    #  Определение цвета ценника
    # ================================================================
    @staticmethod
    def _detect_color(crop: np.ndarray) -> str:
        if crop is None or crop.size == 0:
            return 'unknown'

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 50, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red1, upper_red1),
            cv2.inRange(hsv, lower_red2, upper_red2)
        )

        lower_yellow = np.array([20, 70, 50])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        red_mask = cv2.bitwise_and(red_mask, cv2.bitwise_not(white_mask))
        yellow_mask = cv2.bitwise_and(yellow_mask, cv2.bitwise_not(white_mask))

        total_pixels = crop.shape[0] * crop.shape[1]
        red_pixels = cv2.countNonZero(red_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)

        min_color_ratio = 0.05

        if red_pixels / total_pixels > min_color_ratio:
            return 'red'
        elif yellow_pixels / total_pixels > min_color_ratio:
            return 'yellow'
        else:
            return 'white'

    # ================================================================
    #  Парсинг QR-полей
    # ================================================================
    @staticmethod
    def _parse_qr_fields(qr_data: str) -> Dict[str, str]:
        result = {
            'qr_code_barcode': '',
            'price1_qr': '', 'price2_qr': '', 'price3_qr': '', 'price4_qr': '',
            'wholesale_level_1_count': '', 'wholesale_level_1_price': '',
            'wholesale_level_2_count': '', 'wholesale_level_2_price': '',
            'action_price_qr': '', 'action_code_qr': '',
        }
        if not qr_data:
            return result

        key_map = {
            'barcode': 'qr_code_barcode',
            'price1': 'price1_qr', 'p1': 'price1_qr',
            'price2': 'price2_qr', 'p2': 'price2_qr',
            'price3': 'price3_qr', 'p3': 'price3_qr',
            'price4': 'price4_qr', 'p4': 'price4_qr',
            'wholesale_level_1_count': 'wholesale_level_1_count',
            'wholesale_level_1_price': 'wholesale_level_1_price',
            'wholesale_level_2_count': 'wholesale_level_2_count',
            'wholesale_level_2_price': 'wholesale_level_2_price',
            'actionprice': 'action_price_qr', 'ap': 'action_price_qr',
            'actioncode': 'action_code_qr', 'ac': 'action_code_qr',
        }

        for part in qr_data.split(';'):
            part = part.strip()
            if '=' not in part:
                continue
            key, value = part.split('=', 1)
            key_lower = key.strip().lower()
            value = value.strip()
            if key_lower in key_map:
                result[key_map[key_lower]] = value

        return result

    @staticmethod
    def _format_price(val):
        s = str(val).strip()
        if not s:
            return ''
        s = s.replace('.', ',')
        return s

    @staticmethod
    def _deduplicate_by_content(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        def _content_key(row):
            name = str(row.get('product_name', '')).strip().lower()
            name = re.sub(r'\s+', ' ', name)
            price_d = str(row.get('price_default', '')).strip()
            price_disc = str(row.get('price_discount', '')).strip()
            return f"{name}|{price_d}|{price_disc}"

        def _field_count(row):
            count = 0
            for col in ['product_name', 'price_default', 'price_card',
                        'price_discount', 'discount_amount', 'id_sku',
                        'print_datetime', 'code', 'additional_info', 'special_symbols']:
                v = row.get(col, '')
                if pd.notna(v) and str(v).strip() and str(v).strip() != 'нет':
                    count += 1
            return count

        before_filter = len(df)

        def _is_likely_tag(row):
            name = str(row.get('product_name', '')).strip()
            has_name = bool(name) and name != 'nan' and len(name) > 3
            price_d = str(row.get('price_default', '')).strip()
            price_disc = str(row.get('price_discount', '')).strip()
            has_price = bool(price_d) and price_d != 'nan' or bool(price_disc) and price_disc != 'nan'
            has_qr = bool(str(row.get('qr_code_barcode', '')).strip())
            return has_name or has_price or has_qr

        df = df[df.apply(_is_likely_tag, axis=1)].reset_index(drop=True)
        after_filter = len(df)
        if before_filter != after_filter:
            print(f"  Фильтрация: {before_filter} -> {after_filter} (убраны строки без имени/цены/QR)")

        df['_content_key'] = df.apply(_content_key, axis=1)
        df['_field_count'] = df.apply(_field_count, axis=1)

        df = df.sort_values('_field_count', ascending=False)
        deduped = df.drop_duplicates(subset=['_content_key'], keep='first')

        before = len(df)
        after = len(deduped)
        if before != after:
            print(f"  Контент-дедупликация: {before} -> {after} строк")

        deduped = deduped.drop(columns=['_content_key', '_field_count'])
        return deduped

    @staticmethod
    def _has_price_in_raw_text(raw_text: str) -> bool:
        if not raw_text:
            return False
        if re.search(r'\d{1,6}[.,\s]\d{2}\b', raw_text):
            return True
        if re.search(r'\b\d{2,6}\b', raw_text):
            nums = re.findall(r'\b(\d{2,6})\b', raw_text)
            for n in nums:
                val = int(n)
                if 10 <= val <= 99999:
                    return True
        return False

    def _filter_by_price(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if df.empty:
            return df, df

        def _has_any_price(row):
            for col in ['price_default', 'price_card', 'price_discount']:
                v = str(row.get(col, '')).strip()
                if v and v != 'nan' and v != '':
                    return True
            raw = str(row.get('raw_text', '')).strip()
            if self._has_price_in_raw_text(raw):
                return True
            return False

        mask = df.apply(_has_any_price, axis=1)
        kept = df[mask].reset_index(drop=True)
        filtered = df[~mask].reset_index(drop=True)
        print(f"  Фильтр по цене: {len(df)} -> {len(kept)} оставлено, {len(filtered)} отфильтровано")
        return kept, filtered

    # ================================================================
    #  ГЛАВНЫЙ МЕТОД
    # ================================================================
    def run_to_csv(self, video_path: str,
                   debug: bool = True,
                   debug_dir: str = 'debug_output_ocr',
                   csv_path: str = 'output_ocr.csv',
                   progress_callback: Optional[Callable] = None) -> List[Dict]:
        def _progress(message: str, fraction: float):
            print(f"[{fraction:.0%}] {message}")
            if progress_callback:
                progress_callback(message, fraction)

        _progress("Выбор ключевых кадров...", 0.0)
        keyframes = self._sample_keyframes(video_path)
        _progress(f"Отобрано {len(keyframes)} кадров", 0.15)

        if not keyframes:
            _progress("Нет ключевых кадров", 1.0)
            return []

        _progress("Детекция текста и QR...", 0.2)
        keyframes = self._detect_text_and_qr(keyframes)
        _progress(f"Текст+QR найдены", 0.3)

        _progress("Генерация кандидатов...", 0.35)
        keyframes = self._generate_candidates(keyframes)
        _progress("Кандидаты сгенерированы", 0.4)

        _progress("Уточнение границ...", 0.42)
        keyframes = self._refine_boundaries(keyframes)
        _progress("Границы уточнены", 0.44)

        _progress("Обрезка вертикали...", 0.45)
        keyframes = self._trim_vertical_excess(keyframes)
        _progress("Вертикаль обрезана", 0.47)

        _progress("Дедупликация...", 0.48)
        keyframes = self._deduplicate_across_frames(keyframes)
        _progress(f"Уникальных ценников: {len(keyframes)}", 0.5)

        if not keyframes:
            _progress("Ценники не найдены", 1.0)
            return []

        _progress("Нормализация ориентации...", 0.52)
        if self.simple_rotate:
            print("  Простой поворот: пропускаю нормализацию ориентации")
            for kf in keyframes:
                for tag in kf.get('price_tags', []):
                    crop = tag.get('crop')
                    if crop is not None:
                        tag['color'] = self._detect_color(crop)
        else:
            keyframes = self._normalize_orientation(keyframes)

        if debug:
            self._save_debug_crops(keyframes, debug_dir)

        _progress("Распознавание текста (OCR)...", 0.55)
        keyframes = self._run_ocr_on_crops(keyframes, progress_callback=progress_callback)

        records = []
        filename = os.path.basename(video_path)
        for kf in keyframes:
            ts = kf['timestamp_ms']
            for tag in kf.get('price_tags', []):
                crop_bbox = tag.get('crop_bbox', [0, 0, 0, 0])
                x_min = int(crop_bbox[0])
                y_min = int(crop_bbox[1])
                x_max = int(crop_bbox[2])
                y_max = int(crop_bbox[3])

                color = tag.get('color', 'unknown')

                ocr_lines = tag.get('ocr_text', [])
                raw_text = ' '.join([ln['text'] for ln in ocr_lines])

                fields = self._parse_fields_content_based(tag)

                qr_fields = self._parse_qr_fields(tag.get('qr_data', ''))

                warped_image_path = ''
                if debug and tag.get('crop') is not None:
                    crop = tag['crop']
                    crop_dir = os.path.join(debug_dir, 'crops')
                    os.makedirs(crop_dir, exist_ok=True)
                    fname = f"crop_{ts}_{x_min}_{y_min}.png"
                    warped_image_path = os.path.join(crop_dir, fname)
                    cv2.imwrite(warped_image_path, crop)

                record = {
                    'filename': filename,
                    'product_name': fields['product_name'],
                    'price_default': fields['price_default'],
                    'price_card': fields['price_card'],
                    'price_discount': fields['price_discount'],
                    'barcode': tag.get('qr_data', ''),
                    'discount_amount': fields['discount_amount'],
                    'id_sku': fields['id_sku'],
                    'print_datetime': fields['print_datetime'],
                    'code': fields['code'],
                    'additional_info': fields['additional_info'],
                    'color': color,
                    'special_symbols': fields['special_symbols'],
                    'frame_timestamp': ts,
                    'x_min': x_min,
                    'y_min': y_min,
                    'x_max': x_max,
                    'y_max': y_max,
                    'warped_image': warped_image_path,
                    'raw_text': raw_text,
                    **qr_fields,
                }
                records.append(record)

        df = pd.DataFrame(records)
        columns = [
            'filename', 'product_name', 'price_default', 'price_card', 'price_discount',
            'barcode', 'discount_amount', 'id_sku', 'print_datetime', 'code',
            'additional_info', 'color', 'special_symbols', 'frame_timestamp',
            'x_min', 'y_min', 'x_max', 'y_max', 'warped_image', 'raw_text',
            'qr_code_barcode',
            'price1_qr', 'price2_qr', 'price3_qr', 'price4_qr',
            'wholesale_level_1_count', 'wholesale_level_1_price',
            'wholesale_level_2_count', 'wholesale_level_2_price',
            'action_price_qr', 'action_code_qr',
        ]
        for col in columns:
            if col not in df.columns:
                df[col] = ''
        df = df[columns]

        price_cols = ['price_default', 'price_card', 'price_discount',
                      'price1_qr', 'price2_qr', 'price3_qr', 'price4_qr',
                      'wholesale_level_1_price', 'wholesale_level_2_price',
                      'action_price_qr']
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda v: self._format_price(v) if pd.notna(v) and v else v)

        df = self._deduplicate_by_content(df)

        df, df_filtered = self._filter_by_price(df)

        df.to_csv(csv_path, index=False, encoding='utf-8')

        filtered_csv_path = csv_path.replace('.csv', '_filtered_out.csv')
        df_filtered.to_csv(filtered_csv_path, index=False, encoding='utf-8')

        _progress(f"Готово: {len(df)} ценников ({len(df_filtered)} отфильтровано)", 1.0)
        return keyframes

    # ================================================================
    #  Отладка
    # ================================================================
    def _save_debug_crops(self, keyframes: List[Dict], output_dir: str):
        crops_dir = os.path.join(output_dir, 'crops')
        frames_dir = os.path.join(output_dir, 'frames')
        os.makedirs(crops_dir, exist_ok=True)
        os.makedirs(frames_dir, exist_ok=True)

        for kf in keyframes:
            ts = kf['timestamp_ms']
            image = kf.get('image')
            if image is not None:
                cv2.imwrite(os.path.join(frames_dir, f"frame_{ts:06d}.png"), image)

            for tag in kf.get('price_tags', []):
                crop = tag.get('crop')
                if crop is not None:
                    bbox = tag.get('crop_bbox', [0, 0, 0, 0])
                    cv2.imwrite(
                        os.path.join(crops_dir, f"crop_{ts}_{int(bbox[0])}_{int(bbox[1])}.png"),
                        crop
                    )

    # ================================================================
    #  HTML-отчёт
    # ================================================================
    @staticmethod
    def _build_html(df: pd.DataFrame, title: str) -> str:
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

        df = df.copy()
        df.insert(0, '#', range(1, len(df) + 1))
        df.insert(1, 'image', image_tags)

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
        <h2>{title}</h2>
        {table}
        </body>
        </html>
        """
        return html_template.replace('{title}', title).replace('{table}', df.to_html(escape=False, index=False))

    @staticmethod
    def generate_html_report(csv_path: str, html_path: str = 'report_ocr.html'):
        if not os.path.exists(csv_path):
            print(f"CSV файл не найден: {csv_path}")
            return
        df = pd.read_csv(csv_path)

        html = TextBasedPriceTagPipeline._build_html(df, 'Результаты распознавания ценников (OCR-пайплайн)')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"HTML-отчёт сохранён: {html_path}")

        filtered_csv_path = csv_path.replace('.csv', '_filtered_out.csv')
        if os.path.exists(filtered_csv_path):
            df_filtered = pd.read_csv(filtered_csv_path)
            if not df_filtered.empty:
                filtered_html_path = html_path.replace('.html', '_filtered_out.html')
                html_filtered = TextBasedPriceTagPipeline._build_html(
                    df_filtered,
                    f'Отфильтрованные строки (без распознанной цены) — {len(df_filtered)} шт.'
                )
                with open(filtered_html_path, 'w', encoding='utf-8') as f:
                    f.write(html_filtered)
                print(f"HTML-отчёт отфильтрованных: {filtered_html_path}")


# ================================================================
#  Пример использования
# ================================================================
if __name__ == "__main__":
    pipeline = TextBasedPriceTagPipeline(
        east_model_path='models/frozen_east_text_detection.pb',
        assume_99_kopecks=True,
    )
    pipeline.run_to_csv(
        video_path='videos/43_15.mp4',
        debug=True,
        debug_dir='debug_output_ocr',
        csv_path='result_ocr.csv',
    )
    pipeline.generate_html_report('result_ocr.csv', 'report_ocr.html')

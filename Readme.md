# Price Tag Detector OCR

Распознавание ценников с видео робота с помощью EAST + PaddleOCR (без обучения моделей).

## Быстрый старт (CPU)

```bash
python -m venv .venv
source .venv/bin/activate

# PyTorch (CPU)
pip install torch torchvision

# PaddlePaddle (CPU)
pip install paddlepaddle paddleocr

# Остальные зависимости
pip install -r requirements.txt

# Скачать EAST модель
mkdir -p models
curl -L -o models/frozen_east_text_detection.pb \
  https://raw.githubusercontent.com/oyyd/frozen_east_text_detection.pb/master/frozen_east_text_detection.pb

# Запуск
python web_app.py
```

Открыть http://localhost:8000

## Установка с GPU (NVIDIA CUDA)

Требования: NVIDIA GPU с драйвером, CUDA Toolkit 12.x.

```bash
python -m venv .venv
source .venv/bin/activate

# PyTorch с CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# PaddlePaddle GPU
python -m pip install paddlepaddle-gpu paddleocr

# Остальные зависимости
pip install -r requirements.txt

# Скачать EAST модель
mkdir -p models
curl -L -o models/frozen_east_text_detection.pb \
  https://raw.githubusercontent.com/oyyd/frozen_east_text_detection.pb/master/frozen_east_text_detection.pb

# Запуск
python web_app.py
```

При запуске выбрать "GPU" в выпадающем списке "Устройство" или оставить "Авто".

### Что ускоряет GPU

| Компонент | CPU | GPU |
|---|---|---|
| EAST (960px) | ~0.4с/кадр | ~0.05с/кадр (с OpenCV CUDA) |
| EAST (1920px) | ~1.5с/кадр | ~0.15с/кадр (с OpenCV CUDA) / ~1.5с/кадр (без) |
| PaddleOCR | ~1.7с/кроп | ~0.1с/кроп |
| MobileNetV3 (CNN дедуп) | ~0.1с/кроп | ~0.005с/кроп |

Без OpenCV с CUDA EAST работает на CPU даже при GPU-режиме. PaddleOCR и MobileNetV3 используют GPU через PaddlePaddle/PyTorch.

### OpenCV с CUDA (опционально, для максимальной скорости EAST)

Если нужна максимальная скорость EAST на GPU, соберите OpenCV из исходцов с CUDA:

```bash
# Установить зависимости
sudo apt-get install cmake build-essential libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev

# Скачать OpenCV
git clone --depth 1 -b 4.10.0 https://github.com/opencv/opencv.git
git clone --depth 1 -b 4.10.0 https://github.com/opencv/opencv_contrib.git

# Собрать
cd opencv && mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D WITH_CUDA=ON \
      -D CUDA_ARCH_BIN=8.6 \
      -D CUDA_FAST_MATH=ON \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D ENABLE_FAST_MATH=ON \
      -D PYTHON3_EXECUTABLE=$(which python3) \
      -D PYTHON3_INCLUDE_DIR=$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])") \
      -D PYTHON3_PACKAGES_PATH=$(python3 -c "from site import getsitepackages; print(getsitepackages()[0])") \
      -D BUILD_opencv_python3=ON \
      -D BUILD_EXAMPLES=OFF \
      ..

make -j$(nproc)
sudo make install
```

Замените `CUDA_ARCH_BIN=8.6` на архитектуру вашего GPU:
- RTX 3080/3090: 8.6
- RTX 4080/4090: 8.9
- GTX 1080/2080: 6.1/7.5

После сборки удалите `opencv-python` из pip: `pip uninstall opencv-python opencv-contrib-python`

## Режимы работы

- **CPU**: EAST 960px, PaddleOCR CPU, MobileNetV3 CPU — медленно, но работает везде
- **GPU**: EAST 1920px (больше текста найдено), PaddleOCR GPU, MobileNetV3 CUDA — быстро и качественно

## Папка результатов GPU

Положите результаты прогона на GPU в папку `gpu_results/`:

```
gpu_results/
├── result_ocr.csv
├── report_ocr.html
├── result_ocr_duplicates.html
├── result_ocr_filtered_out.csv
└── debug_output_ocr/
    └── crops/
```

## Пайплайн обработки

1. Выбор ключевых кадров (по резкости и разнообразию)
2. Детекция текста EAST + QR-кодов
3. Кластеризация текстовых боксов (DBSCAN)
4. Уточнение границ + PaddleOCR расширение кропов
5. **CNN-дедупликация** (MobileNetV3-Small, cosine similarity >= 0.75)
6. Нормализация ориентации
7. OCR (PaddleOCR на кропах)
7a. **Текстовая дедупликация** (difflib, word-level matching)
8. Контент-дедупликация (точное совпадение название+цена)
9. Фильтрация (только строки с ценой)

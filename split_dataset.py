import os
import random
import shutil
import argparse
from pathlib import Path

def clear_dir(path: Path):
    """Удаляет папку, если она существует, и создаёт заново пустую."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)

def split_dataset(
    source_images: Path,
    source_labels: Path,
    output_base: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
    extensions: tuple = ('.jpg', '.jpeg', '.png'),
):
    """
    Создаёт структуру:
        output_base/train_splitted/images, labels
        output_base/val_splitted/images, labels
    из файлов source_images и source_labels.
    """
    # 1. Собираем имена всех изображений (без расширения)
    image_stems = []
    for ext in extensions:
        for p in source_images.glob(f'*{ext}'):
            image_stems.append(p.stem)
        for p in source_images.glob(f'*{ext.upper()}'):
            image_stems.append(p.stem)

    if not image_stems:
        raise RuntimeError(f'В папке {source_images} не найдено изображений с расширениями {extensions}')

    # Убираем возможные дубли (если вдруг есть и .jpg и .JPG)
    image_stems = sorted(set(image_stems))

    # 2. Перемешиваем и делим
    random.seed(seed)
    random.shuffle(image_stems)
    split_idx = int(len(image_stems) * (1 - val_ratio))
    train_stems = image_stems[:split_idx]
    val_stems = image_stems[split_idx:]

    # 3. Очищаем целевые папки
    train_img_dir = output_base / 'train_splitted' / 'images'
    train_lbl_dir = output_base / 'train_splitted' / 'labels'
    val_img_dir = output_base / 'val_splitted' / 'images'
    val_lbl_dir = output_base / 'val_splitted' / 'labels'
    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        clear_dir(d)

    # 4. Копируем файлы (train)
    for stem in train_stems:
        # Ищем файл изображения (любое расширение)
        img_src = None
        for ext in extensions + tuple(e.upper() for e in extensions):
            candidate = source_images / f'{stem}{ext}'
            if candidate.exists():
                img_src = candidate
                break
        if not img_src:
            print(f'⚠️  Пропущен {stem}: изображение не найдено')
            continue
        lbl_src = source_labels / f'{stem}.txt'
        if not lbl_src.exists():
            print(f'⚠️  Пропущен {stem}: нет файла разметки')
            continue

        shutil.copy2(img_src, train_img_dir / img_src.name)
        shutil.copy2(lbl_src, train_lbl_dir / lbl_src.name)

    # 5. Копируем файлы (val)
    for stem in val_stems:
        img_src = None
        for ext in extensions + tuple(e.upper() for e in extensions):
            candidate = source_images / f'{stem}{ext}'
            if candidate.exists():
                img_src = candidate
                break
        if not img_src:
            continue
        lbl_src = source_labels / f'{stem}.txt'
        if not lbl_src.exists():
            continue
        shutil.copy2(img_src, val_img_dir / img_src.name)
        shutil.copy2(lbl_src, val_lbl_dir / lbl_src.name)

    print(f'✅ Готово: train = {len(train_stems)} изображений, val = {len(val_stems)} изображений')
    print(f'Пути для data.yaml:')
    print(f'  train: {train_img_dir}')
    print(f'  val:   {val_img_dir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Разделение датасета на train/val')
    parser.add_argument('--source', type=Path, required=True,
                        help='Путь к папке train (содержит images/ и labels/)')
    parser.add_argument('--output', type=Path, default=None,
                        help='Куда положить train_splitted и val_splitted (по умолчанию рядом с source)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Доля данных для валидации (0..1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Случайное зерно')
    args = parser.parse_args()

    source_dir = args.source
    # Ищем стандартное расположение: source/images и source/labels
    src_images = source_dir / 'images'
    src_labels = source_dir / 'labels'
    if not src_images.exists() or not src_labels.exists():
        raise FileNotFoundError(f'Ожидаются папки {src_images} и {src_labels}')

    output_base = args.output or (source_dir.parent)

    split_dataset(
        source_images=src_images,
        source_labels=src_labels,
        output_base=output_base,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
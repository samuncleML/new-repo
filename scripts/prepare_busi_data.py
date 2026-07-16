from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
source = root / 'BUSI'
target = root / 'data' / 'busi'
images_dir = target / 'images'
masks_dir = target / 'masks' / '0'
images_dir.mkdir(parents=True, exist_ok=True)
masks_dir.mkdir(parents=True, exist_ok=True)

for cls in ['benign', 'malignant', 'normal']:
    cls_dir = source / cls
    if not cls_dir.exists():
        continue
    for image_path in sorted(cls_dir.glob('*.png')):
        if image_path.name.endswith('_mask.png'):
            mask_name = image_path.name.replace('_mask.png', '.png')
            shutil.copy2(image_path, masks_dir / mask_name)
        elif '_mask' not in image_path.name:
            shutil.copy2(image_path, images_dir / image_path.name)

# Keep only pairs that have matching image and mask files with the same stem.
image_names = {p.stem for p in images_dir.glob('*.png')}
mask_names = {p.stem for p in masks_dir.glob('*.png')}
paired_names = sorted(image_names & mask_names)
for path in list(images_dir.glob('*.png')):
    if path.stem not in paired_names:
        path.unlink()
for path in list(masks_dir.glob('*.png')):
    if path.stem not in paired_names:
        path.unlink()

print(f'Prepared {len(list(images_dir.glob("*.png")))} images and {len(list(masks_dir.glob("*.png")))} masks in {target}')

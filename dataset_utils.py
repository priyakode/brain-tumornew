import os
import sys
import numpy as np
import cv2
from PIL import Image

def extract_samples_to_npy(samples_dir="samples", output_dir="brain_tumor_dataset", target_dim=128):
    """
    Extracts MRI images and target segmentation masks from composite sample PNG files in samples/
    Each sample PNG in samples/ has MRI image, ground truth mask, and prediction visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    images = []
    masks = []
    labels = []

    if not os.path.exists(samples_dir):
        print(f"Directory {samples_dir} not found.")
        return generate_synthetic_dataset(output_dir=output_dir, num_samples=50, target_dim=target_dim)

    sample_files = sorted([f for f in os.listdir(samples_dir) if f.endswith('.png')])
    if not sample_files:
        return generate_synthetic_dataset(output_dir=output_dir, num_samples=50, target_dim=target_dim)

    for i, fname in enumerate(sample_files):
        fpath = os.path.join(samples_dir, fname)
        try:
            img = Image.open(fpath).convert('RGB')
            arr = np.array(img)
            h, w, c = arr.shape

            # In sample images, left section (~1/3 of width) is MRI image, middle is mask/overlay
            mri_crop = arr[:, :int(w / 3), :]
            mri_gray = cv2.cvtColor(mri_crop, cv2.COLOR_RGB2GRAY)
            mri_resized = cv2.resize(mri_gray, (target_dim, target_dim), interpolation=cv2.INTER_AREA)

            # Middle crop contains the mask/overlay
            mask_crop = arr[:, int(w / 3):int(2 * w / 3), :]
            mask_gray = cv2.cvtColor(mask_crop, cv2.COLOR_RGB2GRAY)
            mask_resized = cv2.resize(mask_gray, (target_dim, target_dim), interpolation=cv2.INTER_AREA)

            # Threshold mask crop to produce binary tumor mask
            # Bright regions or tinted tumor regions in ground truth mask
            _, binary_mask = cv2.threshold(mask_resized, 180, 255, cv2.THRESH_BINARY_INV)

            images.append(mri_resized)
            masks.append((binary_mask > 0).astype(np.bool_))
            labels.append(1 if np.sum(binary_mask) > 10 else 0)
        except Exception as e:
            print(f"Error processing {fname}: {e}")

    # Augment samples to build a reasonable dataset for training if dataset is small
    if len(images) > 0 and len(images) < 50:
        orig_imgs = list(images)
        orig_masks = list(masks)
        orig_labels = list(labels)

        for aug_idx in range(4):
            for img_arr, mask_arr, lbl in zip(orig_imgs, orig_masks, orig_labels):
                if aug_idx == 0:
                    # Flip horizontally
                    aug_img = np.fliplr(img_arr)
                    aug_mask = np.fliplr(mask_arr)
                elif aug_idx == 1:
                    # Flip vertically
                    aug_img = np.flipud(img_arr)
                    aug_mask = np.flipud(mask_arr)
                elif aug_idx == 2:
                    # Rotate 90
                    aug_img = np.rot90(img_arr, 1)
                    aug_mask = np.rot90(mask_arr, 1)
                else:
                    # Brightness shift
                    aug_img = np.clip(img_arr.astype(np.float32) * 1.1, 0, 255).astype(np.uint8)
                    aug_mask = mask_arr

                images.append(aug_img)
                masks.append(aug_mask)
                labels.append(lbl)

    images_arr = np.array(images, dtype=np.uint8)
    masks_arr = np.array(masks, dtype=bool)
    labels_arr = np.array(labels, dtype=int)

    np.save(os.path.join(output_dir, 'images.npy'), images_arr)
    np.save(os.path.join(output_dir, 'masks.npy'), masks_arr)
    np.save(os.path.join(output_dir, 'labels.npy'), labels_arr)

    print(f"Saved dataset to {output_dir}:")
    print(f"  images.npy: {images_arr.shape}")
    print(f"  masks.npy: {masks_arr.shape}")
    print(f"  labels.npy: {labels_arr.shape}")
    return images_arr, masks_arr, labels_arr

def generate_synthetic_dataset(output_dir="brain_tumor_dataset", num_samples=100, target_dim=128):
    """Generates synthetic brain MRI scans with simulated tumors for zero-dependency robust testing/training."""
    os.makedirs(output_dir, exist_ok=True)
    images = []
    masks = []
    labels = []

    np.random.seed(42)
    for i in range(num_samples):
        # Create brain ellipse background
        img = np.zeros((target_dim, target_dim), dtype=np.uint8)
        center = (target_dim // 2, target_dim // 2)
        axes = (int(target_dim * 0.38), int(target_dim * 0.44))
        cv2.ellipse(img, center, axes, 0, 0, 360, 120, -1)

        # Add brain texture and brain structure noise
        noise = np.random.normal(0, 15, (target_dim, target_dim)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # Generate tumor in ~75% of samples
        has_tumor = np.random.rand() > 0.25
        mask = np.zeros((target_dim, target_dim), dtype=bool)

        if has_tumor:
            tx = int(center[0] + np.random.uniform(-target_dim * 0.15, target_dim * 0.15))
            ty = int(center[1] + np.random.uniform(-target_dim * 0.15, target_dim * 0.15))
            tradius = int(np.random.uniform(target_dim * 0.08, target_dim * 0.18))

            tumor_mask_single = np.zeros((target_dim, target_dim), dtype=np.uint8)
            cv2.circle(tumor_mask_single, (tx, ty), tradius, 255, -1)
            # Add irregular shape
            t_noise = np.random.normal(0, 10, (target_dim, target_dim)).astype(np.int16)
            tumor_mask_single = np.clip(tumor_mask_single.astype(np.int16) + t_noise, 0, 255).astype(np.uint8)

            # Brighten tumor region in MRI image
            img[tumor_mask_single > 100] = np.clip(img[tumor_mask_single > 100].astype(np.int16) + 110, 0, 255).astype(np.uint8)
            mask = tumor_mask_single > 100
            label = np.random.choice([1, 2, 3]) # meningioma, glioma, pituitary
        else:
            label = 0

        images.append(img)
        masks.append(mask)
        labels.append(label)

    images_arr = np.array(images, dtype=np.uint8)
    masks_arr = np.array(masks, dtype=bool)
    labels_arr = np.array(labels, dtype=int)

    np.save(os.path.join(output_dir, 'images.npy'), images_arr)
    np.save(os.path.join(output_dir, 'masks.npy'), masks_arr)
    np.save(os.path.join(output_dir, 'labels.npy'), labels_arr)

    print(f"Generated synthetic MRI dataset in {output_dir}:")
    print(f"  images.npy: {images_arr.shape}")
    print(f"  masks.npy: {masks_arr.shape}")
    print(f"  labels.npy: {labels_arr.shape}")
    return images_arr, masks_arr, labels_arr

def load_dataset(dataset_dir="brain_tumor_dataset"):
    img_path = os.path.join(dataset_dir, 'images.npy')
    mask_path = os.path.join(dataset_dir, 'masks.npy')
    lbl_path = os.path.join(dataset_dir, 'labels.npy')

    if os.path.exists(img_path) and os.path.exists(mask_path):
        images = np.load(img_path)
        masks = np.load(mask_path)
        labels = np.load(lbl_path) if os.path.exists(lbl_path) else np.ones(len(images))
        return images, masks, labels
    else:
        return extract_samples_to_npy(output_dir=dataset_dir)

if __name__ == '__main__':
    extract_samples_to_npy()

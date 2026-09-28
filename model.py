import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """PyTorch U-Net Architecture for Brain Tumor Segmentation"""
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (Downsampling)
        curr_in = in_channels
        for feature in features:
            self.downs.append(DoubleConv(curr_in, feature))
            curr_in = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder (Upsampling)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)

            concat_x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_x)

        return torch.sigmoid(self.final_conv(x))

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred_f = y_pred.contiguous().view(-1)
        y_true_f = y_true.contiguous().view(-1)
        intersection = (y_pred_f * y_true_f).sum()
        dice = (2. * intersection + self.smooth) / (y_pred_f.sum() + y_true_f.sum() + self.smooth)
        return 1.0 - dice

class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, y_pred, y_true):
        return self.bce(y_pred, y_true) + self.dice(y_pred, y_true)

def calculate_iou(pred_mask, true_mask):
    """Calculate Intersection over Union (IoU) metric."""
    intersection = np.logical_and(pred_mask > 0, true_mask > 0).sum()
    union = np.logical_or(pred_mask > 0, true_mask > 0).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float((intersection + 1e-7) / (union + 1e-7))

def calculate_dice(pred_mask, true_mask):
    """Calculate Dice Similarity Coefficient."""
    intersection = np.logical_and(pred_mask > 0, true_mask > 0).sum()
    total = (pred_mask > 0).sum() + (true_mask > 0).sum()
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    return float((2.0 * intersection + 1e-7) / (total + 1e-7))

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_or_create_model(weights_path="model_best_checkpoint.pth", device=None):
    if device is None:
        device = get_device()
    model = UNet(in_channels=1, out_channels=1).to(device)
    if os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location=device, weights_only=True)
            state_dict = {k: v.float() if torch.is_floating_point(v) else v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            model.eval()
            print(f"Loaded existing model weights from {weights_path}")
        except Exception as e:
            print(f"Failed to load weights from {weights_path}: {e}")
    else:
        model.eval()
    return model

def preprocess_image(image_input, target_size=(128, 128)):
    """Converts PIL Image, path or numpy array into (1, 1, H, W) normalized float tensor."""
    if isinstance(image_input, str):
        img = Image.open(image_input).convert('L')
        img_np = np.array(img)
    elif isinstance(image_input, Image.Image):
        img = image_input.convert('L')
        img_np = np.array(img)
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 3:
            if image_input.shape[2] == 4:
                img_np = cv2.cvtColor(image_input, cv2.COLOR_RGBA2GRAY)
            elif image_input.shape[2] == 3:
                img_np = cv2.cvtColor(image_input, cv2.COLOR_RGB2GRAY)
            else:
                img_np = image_input[:, :, 0]
        else:
            img_np = image_input
    else:
        raise ValueError("Unsupported image input type")

    orig_shape = img_np.shape
    resized = cv2.resize(img_np, target_size, interpolation=cv2.INTER_AREA)

    # Normalize to [0, 1]
    norm_img = resized.astype(np.float32)
    if norm_img.max() > 1.0:
        norm_img /= 255.0

    tensor = torch.from_numpy(norm_img).unsqueeze(0).unsqueeze(0) # (1, 1, 128, 128)
    return tensor, orig_shape, resized

def detect_tumor(prob_map, threshold=0.35, min_pixel_count=15):
    """Extract tumor detection statistics, bounding boxes, and contours from prediction probability map."""
    binary_mask = (prob_map >= threshold).astype(np.uint8)
    pixel_count = int(np.sum(binary_mask))
    total_pixels = prob_map.size
    area_percentage = float((pixel_count / total_pixels) * 100.0)

    has_tumor = pixel_count >= min_pixel_count
    max_confidence = float(np.max(prob_map)) if has_tumor else float(np.mean(prob_map))

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    if has_tumor and contours:
        for c in contours:
            if cv2.contourArea(c) >= min_pixel_count / 2:
                x, y, w, h = cv2.boundingRect(c)
                bounding_boxes.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h)})

    return {
        "has_tumor": has_tumor,
        "confidence": max_confidence,
        "pixel_count": pixel_count,
        "area_percentage": area_percentage,
        "binary_mask": binary_mask,
        "bounding_boxes": bounding_boxes,
        "contour_count": len(contours)
    }

def predict_mri_image(image_input, model=None, weights_path="model_best_checkpoint.pth", threshold=0.35, device=None):
    """Full end-to-end inference function."""
    if device is None:
        device = get_device()
    if model is None:
        model = load_or_create_model(weights_path=weights_path, device=device)

    model_device = next(model.parameters()).device
    model.eval()

    tensor, orig_shape, resized_gray = preprocess_image(image_input)
    tensor = tensor.to(model_device)

    with torch.no_grad():
        prob_tensor = model(tensor)
        prob_map = prob_tensor.squeeze().cpu().numpy() # (128, 128)

    detection = detect_tumor(prob_map, threshold=threshold)

    # Resize probability map & mask back to original resolution if requested
    prob_map_orig = cv2.resize(prob_map, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR)
    mask_orig = (prob_map_orig >= threshold).astype(np.uint8)

    result = {
        "has_tumor": detection["has_tumor"],
        "confidence": detection["confidence"],
        "area_percentage": detection["area_percentage"],
        "pixel_count": detection["pixel_count"],
        "bounding_boxes": detection["bounding_boxes"],
        "prob_map_128": prob_map,
        "binary_mask_128": detection["binary_mask"],
        "prob_map_orig": prob_map_orig,
        "binary_mask_orig": mask_orig,
        "resized_gray": resized_gray,
        "orig_shape": orig_shape
    }
    return result

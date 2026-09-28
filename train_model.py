import os
import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from model import UNet, BCEDiceLoss, calculate_iou, calculate_dice, get_device
from dataset_utils import load_dataset, extract_samples_to_npy

def train_unet(epochs=20, batch_size=8, lr=1e-3, save_path="model_best_checkpoint.pth"):
    device = get_device()
    print(f"Training U-Net using device: {device}")

    # Load dataset
    dataset_dir = "brain_tumor_dataset"
    if not os.path.exists(os.path.join(dataset_dir, 'images.npy')):
        print("Dataset not found. Preparing dataset from samples...")
        extract_samples_to_npy(samples_dir="samples", output_dir=dataset_dir)

    images, masks, labels = load_dataset(dataset_dir)
    print(f"Loaded dataset - Images: {images.shape}, Masks: {masks.shape}")

    # Preprocess tensors: (N, H, W) -> (N, 1, H, W) normalized float32
    X = torch.tensor(images, dtype=torch.float32).unsqueeze(1) / 255.0
    Y = torch.tensor(masks, dtype=torch.float32).unsqueeze(1)

    # Train/Validation split
    val_size = int(len(X) * 0.2)
    val_size = max(val_size, 1)
    indices = torch.randperm(len(X))

    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    train_dataset = TensorDataset(X[train_indices], Y[train_indices])
    val_dataset = TensorDataset(X[val_indices], Y[val_indices])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = UNet(in_channels=1, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = BCEDiceLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_dice': []}

    print(f"\nStarting U-Net Training for {epochs} Epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch_x)

        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        ious = []
        dices = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item() * len(batch_x)

                preds_np = (preds.cpu().numpy() >= 0.35).squeeze(1)
                targets_np = batch_y.cpu().numpy().squeeze(1)

                for p, t in zip(preds_np, targets_np):
                    ious.append(calculate_iou(p, t))
                    dices.append(calculate_dice(p, t))

        val_loss /= len(val_dataset)
        avg_iou = float(np.mean(ious)) if ious else 0.0
        avg_dice = float(np.mean(dices)) if dices else 0.0

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(avg_iou)
        history['val_dice'].append(avg_dice)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {avg_iou:.4f} | Val Dice: {avg_dice:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state_dict_half = {k: v.half() if torch.is_floating_point(v) else v for k, v in model.state_dict().items()}
            torch.save(state_dict_half, save_path)
            print(f"  --> Saved new best model checkpoint to {save_path} (Val Loss: {val_loss:.4f})")

    print("\nTraining completed successfully!")
    return model, history

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train U-Net on Brain Tumor Dataset")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=8, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    train_unet(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

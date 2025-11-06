"""
TrackNet Training Script

Train TrackNet model for tennis ball detection using annotated data.

Usage:
    python train_tracknet.py --video input.mp4 --annotations annotations.csv --epochs 50
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from tracknet_model import TrackNet, TrackNetLightning
from tracknet_dataset import TrackNetDataset


class TrackNetLoss(nn.Module):
    """
    Custom loss function for TrackNet.

    Combines MSE loss with optional focal loss to handle class imbalance
    (most pixels are background, few pixels are ball).
    """

    def __init__(self, use_focal=True, alpha=0.25, gamma=2.0):
        """
        Initialize loss function.

        Args:
            use_focal: Use focal loss to handle imbalance
            alpha: Focal loss alpha parameter
            gamma: Focal loss gamma parameter
        """
        super(TrackNetLoss, self).__init__()
        self.use_focal = use_focal
        self.alpha = alpha
        self.gamma = gamma
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        """
        Calculate loss.

        Args:
            pred: Predicted heatmap (batch, 1, H, W)
            target: Ground truth heatmap (batch, 1, H, W)

        Returns:
            Loss value
        """
        if self.use_focal:
            # Focal loss: downweight easy examples
            bce = nn.functional.binary_cross_entropy(pred, target, reduction='none')
            pt = torch.exp(-bce)  # pt is the probability of the correct class
            focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
            return focal_loss.mean()
        else:
            # Simple MSE loss
            return self.mse(pred, target)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training")

    for frames, heatmaps in progress_bar:
        frames = frames.to(device)
        heatmaps = heatmaps.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(frames)

        # Calculate loss
        loss = criterion(outputs, heatmaps)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track loss
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct_detections = 0
    total_samples = 0

    with torch.no_grad():
        for frames, heatmaps in tqdm(dataloader, desc="Validation"):
            frames = frames.to(device)
            heatmaps = heatmaps.to(device)

            # Forward pass
            outputs = model(frames)

            # Calculate loss
            loss = criterion(outputs, heatmaps)
            total_loss += loss.item()

            # Calculate accuracy (ball detected within threshold)
            for i in range(outputs.shape[0]):
                pred_max = outputs[i].max()
                target_max = heatmaps[i].max()

                if pred_max > 0.5 and target_max > 0.5:
                    # Find peak positions
                    pred_pos = torch.unravel_index(outputs[i].argmax(), outputs[i].shape)
                    target_pos = torch.unravel_index(heatmaps[i].argmax(), heatmaps[i].shape)

                    # Check if close (within 10 pixels)
                    distance = torch.sqrt(
                        (pred_pos[1] - target_pos[1]) ** 2 +
                        (pred_pos[2] - target_pos[2]) ** 2
                    )

                    if distance < 10:
                        correct_detections += 1

                total_samples += 1

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_detections / total_samples if total_samples > 0 else 0.0

    return avg_loss, accuracy


def train_tracknet(video_path, annotation_csv, output_dir="output/tracknet",
                  epochs=50, batch_size=4, learning_rate=1e-4,
                  model_type='full', val_split=0.2):
    """
    Train TrackNet model.

    Args:
        video_path: Path to training video
        annotation_csv: Path to ball annotations
        output_dir: Directory to save model checkpoints
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        model_type: 'full' or 'light'
        val_split: Validation set split ratio
    """
    # Setup
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Create dataset
    print("\nCreating dataset...")
    dataset = TrackNetDataset(
        video_path=video_path,
        annotation_csv=annotation_csv,
        width=512,
        height=288,
        sigma=5.0
    )

    # Split into train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Create model
    print(f"\nCreating {model_type} TrackNet model...")
    if model_type == 'light':
        model = TrackNetLightning()
    else:
        model = TrackNet()

    model = model.to(device)

    # Loss and optimizer
    criterion = TrackNetLoss(use_focal=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Update learning rate
        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.2%}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_acc
            }, output_dir / 'best_model.pth')
            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, output_dir / f'checkpoint_epoch_{epoch+1}.pth')

    # Save final model
    torch.save(model.state_dict(), output_dir / 'final_model.pth')

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png')
    print(f"\n✓ Training curves saved to {output_dir / 'training_curves.png'}")

    print("\n✓ Training complete!")
    print(f"Best model saved to: {output_dir / 'best_model.pth'}")
    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TrackNet for ball detection")
    parser.add_argument("--video", type=str, required=True, help="Path to training video")
    parser.add_argument("--annotations", type=str, required=True, help="Path to annotations CSV")
    parser.add_argument("--output", type=str, default="output/tracknet", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--model", type=str, default='full', choices=['full', 'light'],
                       help="Model type")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")

    args = parser.parse_args()

    train_tracknet(
        video_path=args.video,
        annotation_csv=args.annotations,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        model_type=args.model,
        val_split=args.val_split
    )

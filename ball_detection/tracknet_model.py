"""
TrackNet: Tennis Ball Detection and Tracking

Based on "TrackNet: A Deep Learning Network for Tracking High-speed and Tiny Objects in Sports"
by Yu-Chuan Huang et al. (2019)

TrackNet uses temporal information from 3 consecutive frames to detect the ball position
and output a heatmap showing the probability of ball location.

Architecture:
- Input: 3 consecutive RGB frames (H x W x 9 channels)
- Backbone: VGG16-based encoder
- Decoder: Upsampling layers (U-Net style)
- Output: Heatmap (H x W x 1) where peak = ball location

Key Innovation:
- Temporal context helps handle motion blur
- Heatmap output is more robust than bounding boxes for tiny objects
- Works even when ball is partially occluded or blurred
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrackNet(nn.Module):
    """
    TrackNet model for tennis ball detection.

    Takes 3 consecutive frames and outputs a heatmap of ball location.
    Uses VGG16-inspired encoder with U-Net style decoder.
    """

    def __init__(self, input_channels=9, output_channels=1, dropout_rate=0.2):
        """
        Initialize TrackNet model.

        Args:
            input_channels: Number of input channels (3 frames * 3 RGB = 9)
            output_channels: Number of output channels (1 for heatmap)
            dropout_rate: Dropout probability for regularization
        """
        super(TrackNet, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.dropout_rate = dropout_rate

        # Encoder (VGG16-style with batch norm)
        # Block 1
        self.conv1_1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder (Upsampling with skip connections)
        # Upsample 1 (512 -> 256, concat with x4(512) = 768 total)
        self.upsample1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1_1 = nn.Conv2d(768, 256, kernel_size=3, padding=1)  # 256+512=768 from concat
        self.bn_up1_1 = nn.BatchNorm2d(256)
        self.conv_up1_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn_up1_2 = nn.BatchNorm2d(256)

        # Upsample 2 (256 -> 128, concat with x3(256) = 384 total)
        self.upsample2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2_1 = nn.Conv2d(384, 128, kernel_size=3, padding=1)  # 128+256=384 from concat
        self.bn_up2_1 = nn.BatchNorm2d(128)
        self.conv_up2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn_up2_2 = nn.BatchNorm2d(128)

        # Upsample 3 (128 -> 64, concat with x2(128) = 192 total)
        self.upsample3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3_1 = nn.Conv2d(192, 64, kernel_size=3, padding=1)  # 64+128=192 from concat
        self.bn_up3_1 = nn.BatchNorm2d(64)
        self.conv_up3_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn_up3_2 = nn.BatchNorm2d(64)

        # Upsample 4 (64 -> 64, concat with x1(64) = 128 total)
        self.upsample4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv_up4_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)  # 64+64=128 from concat
        self.bn_up4_1 = nn.BatchNorm2d(64)
        self.conv_up4_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn_up4_2 = nn.BatchNorm2d(64)

        # Final output layer
        self.output = nn.Conv2d(64, output_channels, kernel_size=1)

        # Dropout for regularization
        self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        """
        Forward pass through TrackNet.

        Args:
            x: Input tensor of shape (batch, 9, H, W) - 3 stacked RGB frames

        Returns:
            Heatmap tensor of shape (batch, 1, H, W) - ball location probability
        """
        # Encoder
        # Block 1
        x1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x1 = F.relu(self.bn1_2(self.conv1_2(x1)))
        p1 = self.pool1(x1)

        # Block 2
        x2 = F.relu(self.bn2_1(self.conv2_1(p1)))
        x2 = F.relu(self.bn2_2(self.conv2_2(x2)))
        p2 = self.pool2(x2)

        # Block 3
        x3 = F.relu(self.bn3_1(self.conv3_1(p2)))
        x3 = F.relu(self.bn3_2(self.conv3_2(x3)))
        x3 = F.relu(self.bn3_3(self.conv3_3(x3)))
        p3 = self.pool3(x3)

        # Block 4 (bottleneck)
        x4 = F.relu(self.bn4_1(self.conv4_1(p3)))
        x4 = F.relu(self.bn4_2(self.conv4_2(x4)))
        x4 = F.relu(self.bn4_3(self.conv4_3(x4)))
        x4 = self.dropout(x4)
        p4 = self.pool4(x4)

        # Decoder with skip connections (U-Net style)
        # Upsample 1
        up1 = self.upsample1(p4)
        merge1 = torch.cat([up1, x4], dim=1)  # Skip connection from encoder
        d1 = F.relu(self.bn_up1_1(self.conv_up1_1(merge1)))
        d1 = F.relu(self.bn_up1_2(self.conv_up1_2(d1)))
        d1 = self.dropout(d1)

        # Upsample 2
        up2 = self.upsample2(d1)
        merge2 = torch.cat([up2, x3], dim=1)
        d2 = F.relu(self.bn_up2_1(self.conv_up2_1(merge2)))
        d2 = F.relu(self.bn_up2_2(self.conv_up2_2(d2)))
        d2 = self.dropout(d2)

        # Upsample 3
        up3 = self.upsample3(d2)
        merge3 = torch.cat([up3, x2], dim=1)
        d3 = F.relu(self.bn_up3_1(self.conv_up3_1(merge3)))
        d3 = F.relu(self.bn_up3_2(self.conv_up3_2(d3)))

        # Upsample 4
        up4 = self.upsample4(d3)
        merge4 = torch.cat([up4, x1], dim=1)
        d4 = F.relu(self.bn_up4_1(self.conv_up4_1(merge4)))
        d4 = F.relu(self.bn_up4_2(self.conv_up4_2(d4)))

        # Output layer (sigmoid for heatmap probability)
        output = torch.sigmoid(self.output(d4))

        return output

    def predict_ball_position(self, heatmap, threshold=0.5):
        """
        Extract ball position from output heatmap.

        Args:
            heatmap: Output heatmap (batch, 1, H, W) or (1, H, W)
            threshold: Confidence threshold for detection

        Returns:
            List of (x, y, confidence) tuples, one per batch item
        """
        if heatmap.dim() == 3:
            heatmap = heatmap.unsqueeze(0)

        batch_size = heatmap.shape[0]
        positions = []

        for i in range(batch_size):
            h = heatmap[i, 0].detach().cpu().numpy()

            # Find maximum value
            max_val = h.max()

            if max_val > threshold:
                # Find position of maximum
                max_pos = h.argmax()
                y = max_pos // h.shape[1]
                x = max_pos % h.shape[1]
                positions.append((int(x), int(y), float(max_val)))
            else:
                # No ball detected
                positions.append(None)

        return positions if len(positions) > 1 else positions[0]


class TrackNetLightning(TrackNet):
    """
    Lighter version of TrackNet with fewer parameters.
    Suitable for faster training and inference with slightly lower accuracy.
    """

    def __init__(self, input_channels=9, output_channels=1, dropout_rate=0.2):
        # Override parent's __init__ to use smaller architecture
        nn.Module.__init__(self)

        self.input_channels = input_channels
        self.output_channels = output_channels

        # Lighter encoder
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up1 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn_up1 = nn.BatchNorm2d(128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_up2 = nn.BatchNorm2d(64)

        self.up3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv_up3 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn_up3 = nn.BatchNorm2d(32)

        self.output = nn.Conv2d(32, output_channels, 1)
        self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        # Encoder
        e1 = F.relu(self.bn1(self.conv1(x)))
        p1 = self.pool1(e1)

        e2 = F.relu(self.bn2(self.conv2(p1)))
        p2 = self.pool2(e2)

        e3 = F.relu(self.bn3(self.conv3(p2)))
        p3 = self.pool3(e3)

        e4 = F.relu(self.bn4(self.conv4(p3)))
        e4 = self.dropout(e4)

        # Decoder
        d1 = self.up1(e4)
        d1 = torch.cat([d1, e3], dim=1)
        d1 = F.relu(self.bn_up1(self.conv_up1(d1)))

        d2 = self.up2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = F.relu(self.bn_up2(self.conv_up2(d2)))

        d3 = self.up3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        d3 = F.relu(self.bn_up3(self.conv_up3(d3)))

        output = torch.sigmoid(self.output(d3))
        return output


def create_tracknet(model_type='full', pretrained=False):
    """
    Factory function to create TrackNet model.

    Args:
        model_type: 'full' for TrackNet or 'light' for TrackNetLightning
        pretrained: Load pretrained weights if available

    Returns:
        TrackNet model instance
    """
    if model_type == 'light':
        model = TrackNetLightning()
    else:
        model = TrackNet()

    if pretrained:
        # TODO: Load pretrained weights
        print("Warning: Pretrained weights not yet available")

    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing TrackNet model...")

    # Create model
    model = TrackNet()

    # Test forward pass
    batch_size = 2
    height, width = 288, 512  # Standard resolution
    dummy_input = torch.randn(batch_size, 9, height, width)

    print(f"\nInput shape: {dummy_input.shape}")

    # Forward pass
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    # Test position prediction
    positions = model.predict_ball_position(output)
    print(f"\nPredicted ball positions:")
    for i, pos in enumerate(positions):
        if pos:
            print(f"  Batch {i}: x={pos[0]}, y={pos[1]}, conf={pos[2]:.3f}")
        else:
            print(f"  Batch {i}: No ball detected")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")

    # Test light model
    print("\n" + "="*60)
    print("Testing TrackNetLightning model...")
    light_model = TrackNetLightning()
    output_light = light_model(dummy_input)
    print(f"Output shape: {output_light.shape}")

    light_params = sum(p.numel() for p in light_model.parameters())
    print(f"Light model parameters: {light_params:,}")
    print(f"Light model size: ~{light_params * 4 / 1024 / 1024:.1f} MB")
    print(f"Parameter reduction: {100 * (total_params - light_params) / total_params:.1f}%")

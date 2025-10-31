from typing import List, Literal, Tuple
from dataclasses import field
from pydantic.dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch import optim
from torch.nn import CrossEntropyLoss
from torchmetrics.classification import Accuracy
from tqdm import tqdm

PoolType = Literal["max", "avg", "none"]

@dataclass
class CNNArchConfig:
    """
    Configuration for a modular CNN architecture.
    """
    num_conv_layers: int
    num_filters: int 
    kernel_size: int 
    pooling: PoolType
    in_channels: int = 1          # 1 for MNIST (grayscale)
    num_classes: int = 10         # digits 0–9

@dataclass
class AugmentationConfig:
    """
    Configuration for data augmentations.
    Set values to 0.0 to disable augmentation.
    """
    # Geometric transforms
    rotation_deg: float               
    translate_xy: Tuple[float, float] 
    shear_deg: float  

@dataclass
class RegularizationConfig:
    """
    Config for regularization techniques.
    """
    # Model-level regularization
    dropout: float            # dropout rate, 0.0 to disable
    weight_decay: float           # L2 regularization term (Adam/SGD)
    use_early_stopping: bool 
    patience: int              # epochs to wait for improvement
    batchnorm: bool = True           # whether to include BatchNorm layers


# ============================================
# DATA PROCESSING STEP 
# ============================================

# Define image transformations 
def getTrainTransforms(augmentation_config: AugmentationConfig):
    """
    Define the transformations applied to training images.
    Includes conversion to tensor, normalization, and augmentations based on the config.
    """
    return transforms.Compose([
        transforms.RandomAffine(
            degrees=augmentation_config.rotation_deg,
            translate=augmentation_config.translate_xy,
            shear=augmentation_config.shear_deg
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # Standard MNIST normalization
    ])

def getTestTransform():
    """
    Define transformations for test and validation datasets.
    No augmentation, just tensor conversion and normalization.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

# Split training dataset into train/validation
def getValidationSplit(train_percentage, dataset):
    """
    Splits the dataset into training and validation sets.

    Args:
        train_percentage: fraction of data used for training (e.g., 0.8)
        dataset: the dataset to split

    Returns:
        train_data, val_data
    """
    seed = 42  # reproducibility
    generator = torch.Generator().manual_seed(seed)

    train_size = int(train_percentage * len(dataset))
    val_size = len(dataset) - train_size

    train_data, val_data = random_split(dataset, [train_size, val_size], generator=generator)
    return train_data, val_data


# Load MNIST dataset and create DataLoaders
def getDataFromTorchVision(augmentation_config: AugmentationConfig, batch_size=60):
    """
    Loads the MNIST dataset and returns train, validation, and test DataLoaders.

    Args:
        batch_size: number of samples per batch

    Returns:
        train_loader, val_loader, test_loader
    """
    # Download MNIST dataset (if not already present)
    train_dataset = torchvision.datasets.MNIST(
        root="dataset/",
        download=True,
        train=True,
        transform=getTrainTransforms(augmentation_config=augmentation_config)
    )

    test_dataset = torchvision.datasets.MNIST(
        root="dataset/",
        download=True,
        train=False,
        transform=getTestTransform()
    )

    # Split into training (80%) and validation (20%)
    train_data, val_data = getValidationSplit(0.8, train_dataset)

    # Create DataLoaders to batch and shuffle data efficiently
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# Visualize a batch of training images (for sanity check)
def imshow(img):
    """
    Utility function to display a batch of images using Matplotlib.
    """
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")
    plt.show()


# ============================================
# MODEL DEFINITION
# ============================================
class ModularCNN(nn.Module):
    """
    A modular CNN controlled by architecture and regularization configs.
    """
    def __init__(self, arch_cfg: CNNArchConfig, reg_cfg: RegularizationConfig):
        super().__init__()

        self.arch_cfg = arch_cfg
        self.reg_cfg = reg_cfg

        layers = []
        in_channels = arch_cfg.in_channels

        # Build convolutional layers dynamically
        for i in range(arch_cfg.num_conv_layers):
            block = []  # temp list for each layer's components

            # Convolution
            block.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=arch_cfg.num_filters,
                    kernel_size=arch_cfg.kernel_size,
                    padding=arch_cfg.kernel_size // 2
                )
            )

            # Optional Batch Normalization
            if reg_cfg.batchnorm:
                block.append(nn.BatchNorm2d(arch_cfg.num_filters))

            # Activation
            block.append(nn.ReLU())

            # Optional pooling
            if arch_cfg.pooling == "max":
                block.append(nn.MaxPool2d(2))
            elif arch_cfg.pooling == "avg":
                block.append(nn.AvgPool2d(2))
            # no pooling if "none"

            # Optional Dropout
            if reg_cfg.dropout > 0:
                block.append(nn.Dropout(p=reg_cfg.dropout))

            # Combine into one block and add to overall layer list
            layers.append(nn.Sequential(*block))

            # Update input channels for the next layer
            in_channels = arch_cfg.num_filters

        # Combine all conv blocks into one Sequential
        self.conv_layers = nn.Sequential(*layers)

        # Compute flattened size for fully connected layer
        num_pools = arch_cfg.num_conv_layers if arch_cfg.pooling != "none" else 0
        side_length = 28 // (2 ** num_pools)
        flattened_size = arch_cfg.num_filters * side_length * side_length

        # Fully connected classification layer
        self.fc = nn.Linear(flattened_size, arch_cfg.num_classes)

    # FORWARD PASS
    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc(x)
        return x


# ================================
# TRAIN / EVAL / TEST 
# ================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    One full pass over the training set.
    """
    model.train()  # enable dropout, batchnorm updates
    running_loss = 0.0

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        # forward
        logits = model(images)
        loss = criterion(logits, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Run validation: returns (avg_loss, accuracy).
    BatchNorm runs in inference mode, Dropout disabled.
    """
    model.eval()

    total_loss = 0.0
    acc_metric = Accuracy(task="multiclass", num_classes=model.arch_cfg.num_classes).to(device)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)
        acc_metric.update(preds, labels)

    avg_loss = total_loss / len(loader.dataset)
    avg_acc = acc_metric.compute().item()
    return avg_loss, avg_acc


# Final test accuracy
@torch.no_grad()
def test_accuracy(model, loader, device):
    model.eval()
    acc_metric = Accuracy(task="multiclass", num_classes=model.arch_cfg.num_classes).to(device)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)
        acc_metric.update(preds, labels)

    return acc_metric.compute().item()


def evaluate_model(arch_cfg, aug_cfg, reg_cfg, num_epochs=5, batch_size=64):
    """
    Trains a CNN with given architecture, augmentation, and regularization configs,
    and returns the validation accuracy after num_epochs.
    """

    # Prepare data loaders
    train_loader, val_loader, test_loader = getDataFromTorchVision(
        augmentation_config=aug_cfg, batch_size=batch_size
    )

    # Initialize model, loss, optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ModularCNN(arch_cfg, reg_cfg).to(device)

    criterion = CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=reg_cfg.weight_decay
    )

    # Training loop (with optional early stopping)
    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Track improvements
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping check
        if reg_cfg.use_early_stopping and epochs_no_improve >= reg_cfg.patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Restore best weights (if early stopping)
    if best_state is not None:
        model.load_state_dict(best_state)

    # Return best validation accuracy
    return best_val_acc


















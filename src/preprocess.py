#!/usr/bin/env python3
"""
Data preprocessing module for ATL-WGAN experiments.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10_data(batch_size=64, download=True):
    """
    Load and preprocess CIFAR-10 dataset.
    
    Args:
        batch_size (int): Batch size for dataloaders
        download (bool): Whether to download the dataset
        
    Returns:
        dataloader (DataLoader): DataLoader for the dataset
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = datasets.CIFAR10(root='./data', download=download, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"CIFAR-10 dataset loaded with {len(dataset)} images")
    return dataloader

if __name__ == "__main__":
    # Test data loading
    dataloader = get_cifar10_data()
    print(f"Example batch shape: {next(iter(dataloader))[0].shape}")

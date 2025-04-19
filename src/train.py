#!/usr/bin/env python3
"""
Training module for ATL-WGAN experiments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)
# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

class AdaptiveEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AdaptiveEncoder, self).__init__()
        self.fc = nn.Linear(input_dim, latent_dim)
        # A learnable mask parameter to adapt latent active dimensions
        self.latent_mask = nn.Parameter(torch.ones(latent_dim))
    
    def forward(self, x):
        z = self.fc(x)
        # Use sigmoid of the mask to modulate each latent unit
        z = z * torch.sigmoid(self.latent_mask)
        return z

class Tokenizer(nn.Module):
    def __init__(self, in_channels, token_dim, num_tokens):
        super(Tokenizer, self).__init__()
        # Use a kernel size 1 convolution to map the feature maps into tokens
        self.conv = nn.Conv2d(in_channels, token_dim * num_tokens, kernel_size=1)
        self.num_tokens = num_tokens
        self.token_dim = token_dim

    def forward(self, x):
        # Input shape: (B, C, H, W)
        tokens = self.conv(x)  # Shape: (B, token_dim*num_tokens, H, W)
        B, _, H, W = tokens.shape
        # Reshape to (B, token_dim, T) where T = num_tokens * H * W, then transpose to get (B, T, token_dim)
        tokens = tokens.view(B, self.token_dim, self.num_tokens * H * W)
        tokens = tokens.transpose(1, 2)
        return tokens

class TokenBasedGenerator(nn.Module):
    def __init__(self, noise_dim, output_channels, token_dim, num_tokens):
        super(TokenBasedGenerator, self).__init__()
        self.noise_proj = nn.Linear(noise_dim, 256*4*4)
        self.initial_conv = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.tokenizer = Tokenizer(in_channels=128, token_dim=token_dim, num_tokens=num_tokens)
        # Transformer to process token sequences
        self.transformer = nn.Transformer(d_model=token_dim, nhead=4, num_encoder_layers=2)
        self.final_fc = nn.Linear(token_dim, output_channels*8*8)  # project token pooled features to image patch

    def forward(self, noise):
        # noise: (B, noise_dim)
        x = self.noise_proj(noise).view(-1, 256, 4, 4)  # reshape to feature map
        x = self.initial_conv(x)  # upsample: (B, 128, 8, 8)
        tokens = self.tokenizer(x)  # (B, T, token_dim)
        # Transformer expects shape (T, B, token_dim)
        tokens = tokens.transpose(0, 1)
        processed_tokens = self.transformer(tokens, tokens)
        processed_tokens = processed_tokens.transpose(0, 1)  # (B, T, token_dim)
        # Pool across tokens and generate image patch
        pooled_tokens = processed_tokens.mean(dim=1)
        out = self.final_fc(pooled_tokens)
        out = out.view(-1, 3, 8, 8)  # Assume output image has 3 channels and size 8x8
        return out

class CNNGenerator(nn.Module):
    def __init__(self, noise_dim, output_channels):
        super(CNNGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 256*4*4),
            nn.ReLU(),
            # Reshape to (B, 256, 4, 4) then a series of transpose convolutions:
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.Tanh()
        )
    def forward(self, noise):
        return self.net(noise)

class JointEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, token_dim, num_tokens):
        super(JointEncoder, self).__init__()
        self.shared_fc = nn.Linear(input_dim, 512)
        self.latent_fc = nn.Linear(512, latent_dim)
        self.token_fc = nn.Linear(512, token_dim * num_tokens)
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        # Include learnable latent mask as before
        self.latent_mask = nn.Parameter(torch.ones(latent_dim))
        
    def forward(self, x):
        shared = F.relu(self.shared_fc(x))
        latent = self.latent_fc(shared)
        latent = latent * torch.sigmoid(self.latent_mask)
        tokens = self.token_fc(shared)
        # Reshape to (B, num_tokens, token_dim)
        tokens = tokens.view(-1, self.num_tokens, self.token_dim)
        return latent, tokens

class ResolutionGeneralizationGenerator(nn.Module):
    def __init__(self, latent_dim, token_dim, num_tokens, output_channels):
        super(ResolutionGeneralizationGenerator, self).__init__()
        self.latent_proj = nn.Linear(latent_dim, 256*4*4)
        self.initial_conv = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        # Transformer layer on tokens
        self.transformer = nn.Transformer(d_model=token_dim, nhead=4, num_encoder_layers=2)
        self.final_fc = nn.Linear(token_dim, output_channels*8*8)
    
    def forward(self, latent, tokens, target_resolution=(8,8)):
        # Process latent into feature maps
        x = self.latent_proj(latent).view(-1, 256, 4, 4)
        x = self.initial_conv(x)  # (B, 128, 8, 8)
        
        # Adjust tokens if target resolution differs
        B, T, token_dim = tokens.size()
        # For the demo, use adaptive pooling on the token dimension (simulate grid adjustment)
        tokens = tokens.transpose(1,2)  # (B, token_dim, T)
        new_token_length = target_resolution[0]*target_resolution[1]
        tokens = F.adaptive_avg_pool1d(tokens, new_token_length)
        tokens = tokens.transpose(1,2)  # (B, new_T, token_dim)
        
        # Process tokens via transformer. Transformer expects (T, B, token_dim)
        tokens_seq = tokens.transpose(0, 1)
        processed_tokens = self.transformer(tokens_seq, tokens_seq)
        processed_tokens = processed_tokens.transpose(0,1).mean(dim=1)
        out = self.final_fc(processed_tokens)
        out = out.view(-1, 3, 8, 8)
        return out

def train_latent_adaptation(dataloader, num_epochs=3, batch_limit=20):
    """
    Train and compare adaptive vs fixed latent encoder models.
    
    Args:
        dataloader: DataLoader for the dataset
        num_epochs: Number of epochs to train
        batch_limit: Maximum number of batches per epoch (for testing/demo)
        
    Returns:
        adaptive_active_dims_history: List of active dimensions over epochs
        fixed_active_dims_history: List of active dimensions over epochs
    """
    print("Starting Experiment 1: Validation of Latent Environment Adaptation")
    
    input_dim = 3 * 32 * 32  # flattened image
    latent_dim = 128  # large latent space; adaptive method will select fewer dimensions effectively
    
    # Instantiate two encoders: one with adaptive latent mask, one with fixed mask
    encoder_adaptive = AdaptiveEncoder(input_dim=input_dim, latent_dim=latent_dim)
    encoder_fixed = AdaptiveEncoder(input_dim=input_dim, latent_dim=latent_dim)
    # Freeze latent_mask for fixed encoder
    encoder_fixed.latent_mask.requires_grad = False

    optimizer_adaptive = optim.Adam(encoder_adaptive.parameters(), lr=1e-4)
    optimizer_fixed = optim.Adam(encoder_fixed.parameters(), lr=1e-4)
    
    adaptive_active_dims_history = []
    fixed_active_dims_history = []
    
    # Training loop
    for epoch in range(num_epochs):
        encoder_adaptive.train()
        encoder_fixed.train()
        batch_count = 0
        for batch, _ in dataloader:
            batch = batch.view(batch.size(0), -1)  # flatten image
            # For adaptive encoder
            optimizer_adaptive.zero_grad()
            z_adaptive = encoder_adaptive(batch)
            loss_adaptive = (z_adaptive ** 2).mean()
            loss_adaptive.backward()
            optimizer_adaptive.step()
            
            # For fixed encoder
            optimizer_fixed.zero_grad()
            z_fixed = encoder_fixed(batch)
            loss_fixed = (z_fixed ** 2).mean()
            loss_fixed.backward()
            optimizer_fixed.step()

            batch_count += 1
            if batch_count >= batch_limit:  # limit batches per epoch for quick testing
                break

        with torch.no_grad():
            active_dims_adaptive = (torch.sigmoid(encoder_adaptive.latent_mask) > 0.5).sum().item()
            active_dims_fixed = (torch.sigmoid(encoder_fixed.latent_mask) > 0.5).sum().item()
            adaptive_active_dims_history.append(active_dims_adaptive)
            fixed_active_dims_history.append(active_dims_fixed)
            print(f"Epoch {epoch+1}: Adaptive encoder active dims = {active_dims_adaptive} | Fixed encoder active dims = {active_dims_fixed}")
    
    # Save models
    torch.save(encoder_adaptive.state_dict(), "models/adaptive_encoder.pth")
    torch.save(encoder_fixed.state_dict(), "models/fixed_encoder.pth")
    
    return adaptive_active_dims_history, fixed_active_dims_history

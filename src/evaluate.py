#!/usr/bin/env python3
"""
Evaluation module for ATL-WGAN experiments.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

def evaluate_latent_adaptation(adaptive_history, fixed_history):
    """
    Evaluate and plot results of latent adaptation experiment.
    
    Args:
        adaptive_history: List of active dimensions for adaptive encoder
        fixed_history: List of active dimensions for fixed encoder
    """
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(np.arange(1, len(adaptive_history)+1), adaptive_history, marker='o', label='Adaptive Encoder')
    plt.plot(np.arange(1, len(fixed_history)+1), fixed_history, marker='x', label='Fixed Encoder')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Count of Active Dimensions", fontsize=12)
    plt.legend(fontsize=10)
    plt.title("Latent Active Dimensions Evolution", fontsize=14)
    plt.grid(True, alpha=0.3)
    filename = "logs/latent_dimension_adaptation_pair1.pdf"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Experiment 1 evaluation completed; plot saved as {filename}")

def evaluate_token_vs_cnn(resolutions, fid_scores_token, fid_scores_cnn):
    """
    Evaluate and plot results of token-based vs CNN generator experiment.
    
    Args:
        resolutions: List of resolutions used
        fid_scores_token: FID scores for token-based generator
        fid_scores_cnn: FID scores for CNN generator
    """
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(resolutions, fid_scores_token, marker='o', label='Token-based Generator')
    plt.plot(resolutions, fid_scores_cnn, marker='x', label='CNN-based Generator')
    plt.xlabel("Resolution (square image side)", fontsize=12)
    plt.ylabel("Simulated FID Score", fontsize=12)
    plt.title("FID Score vs. Image Resolution", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    filename = "logs/fid_resolution_token_vs_cnn_pair1.pdf"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Experiment 2 evaluation completed; FID vs. resolution plot saved as {filename}")

def evaluate_joint_training(latent_active_history, gen_imgs, target_resolutions):
    """
    Evaluate and plot results of joint latent-and-token training experiment.
    
    Args:
        latent_active_history: List of active dimensions over epochs
        gen_imgs: List of generated images at different resolutions
        target_resolutions: List of target resolutions used
    """
    # Plot latent active dimension estimates over epochs
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(np.arange(1, len(latent_active_history)+1), latent_active_history, marker='o')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Count of Active Dimensions (Joint Encoder)", fontsize=12)
    plt.title("Joint Latent Active Dimensions Evolution", fontsize=14)
    plt.grid(True, alpha=0.3)
    filename = "logs/joint_latent_active_dimensions_pair1.pdf"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    
    # Plot generated images at different resolutions
    fig, axes = plt.subplots(1, len(target_resolutions), figsize=(12, 4), dpi=300)
    for idx, (res, img) in enumerate(zip(target_resolutions, gen_imgs)):
        # Normalize image for plotting
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        axes[idx].imshow(img)
        axes[idx].set_title(f"Resolution {res[0]}x{res[1]}")
        axes[idx].axis('off')
    plt.suptitle("Joint Training: Inference at Various Resolutions", fontsize=14)
    filename = "logs/joint_inference_resolutions_pair1.pdf"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Experiment 3 evaluation completed; plots saved as '{filename}' and 'logs/joint_latent_active_dimensions_pair1.pdf'")

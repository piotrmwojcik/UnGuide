#!/usr/bin/env python3
"""
Weights & Biases logging utilities.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import wandb
from torchvision.transforms.functional import to_pil_image


class WandbLogger:
    """Handles all W&B logging operations for training."""
    
    def __init__(self, enabled: bool = False):
        """
        Initialize W&B logger.
        
        Args:
            enabled: Whether W&B logging is enabled
        """
        self.enabled = enabled
        self._initialized = False
    
    def init_tracker(self, config: Dict[str, Any], project_name: str = "UnGuide"):
        """
        Initialize W&B tracker with configuration.
        
        Args:
            config: Training configuration dictionary
            project_name: W&B project name
        """
        if not self.enabled:
            return
        
        wandb.init(project=project_name, config=config)
        self._initialized = True
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log scalar metrics to W&B.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Optional step/iteration number
        """
        if not self.enabled or not self._initialized:
            return
        
        wandb.log(metrics, step=step)
    
    def log_image(
        self,
        image: torch.Tensor,
        caption: str,
        step: Optional[int] = None,
        key: str = "image"
    ):
        """
        Log a single image to W&B.
        
        Args:
            image: Image tensor [C, H, W] in range [0, 1]
            caption: Image caption
            step: Optional step/iteration number
            key: W&B logging key
        """
        if not self.enabled or not self._initialized:
            return
        
        pil_image = to_pil_image(image.cpu())
        wandb.log({key: wandb.Image(pil_image, caption=caption)}, step=step)
    
    def log_images(
        self,
        images: torch.Tensor,
        captions: List[str],
        step: Optional[int] = None,
        key: str = "images"
    ):
        """
        Log multiple images to W&B.
        
        Args:
            images: Image tensor [B, C, H, W] in range [0, 1]
            captions: List of captions for each image
            step: Optional step/iteration number
            key: W&B logging key
        """
        if not self.enabled or not self._initialized:
            return
        
        wandb_images = [
            wandb.Image(to_pil_image(img.cpu()), caption=cap)
            for img, cap in zip(images, captions)
        ]
        wandb.log({key: wandb_images}, step=step)
    
    def log_generated_image(
        self,
        image: torch.Tensor,
        prompt: str,
        step: int,
        prefix: str = ""
    ):
        """
        Log a generated image with prompt.
        
        Args:
            image: Image tensor [C, H, W] in range [0, 1]
            prompt: Generation prompt
            step: Training step/iteration
            prefix: Optional prefix for the logging key
        """
        if not self.enabled or not self._initialized:
            return
        
        key = f"{prefix}_image" if prefix else "generated_image"
        self.log_image(image, caption=prompt, step=step, key=key)
    
    def log_comparison_images(
        self,
        images_dict: Dict[str, torch.Tensor],
        step: int,
        prompt: str = ""
    ):
        """
        Log multiple images for comparison (e.g., original vs unlearned).
        
        Args:
            images_dict: Dictionary mapping labels to image tensors [C, H, W]
            step: Training step/iteration
            prompt: Optional prompt used for generation
        """
        if not self.enabled or not self._initialized:
            return
        
        log_dict = {}
        for label, img in images_dict.items():
            caption = f"{label}: {prompt}" if prompt else label
            pil_img = to_pil_image(img.cpu())
            log_dict[f"comparison/{label}"] = wandb.Image(pil_img, caption=caption)
        
        wandb.log(log_dict, step=step)
    
    def log_gradient_stats(
        self,
        grad_dict: Dict[str, torch.Tensor],
        step: int,
        prefix: str = "gradients"
    ):
        """
        Log gradient statistics.
        
        Args:
            grad_dict: Dictionary of gradient tensors
            step: Training step/iteration
            prefix: Prefix for logging keys
        """
        if not self.enabled or not self._initialized:
            return
        
        stats = {}
        for name, grad in grad_dict.items():
            if grad is not None:
                stats[f"{prefix}/{name}/norm"] = grad.norm().item()
                stats[f"{prefix}/{name}/mean"] = grad.mean().item()
                stats[f"{prefix}/{name}/std"] = grad.std().item()
        
        self.log_metrics(stats, step=step)
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """
        Log hyperparameters as config.
        
        Args:
            params: Dictionary of hyperparameters
        """
        if not self.enabled or not self._initialized:
            return
        
        wandb.config.update(params)
    
    def save_artifact(
        self,
        file_path: str,
        artifact_name: str,
        artifact_type: str = "model",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save a file as a W&B artifact.
        
        Args:
            file_path: Path to file to save
            artifact_name: Name for the artifact
            artifact_type: Type of artifact (model, dataset, etc.)
            metadata: Optional metadata dictionary
        """
        if not self.enabled or not self._initialized:
            return
        
        artifact = wandb.Artifact(artifact_name, type=artifact_type, metadata=metadata)
        artifact.add_file(file_path)
        wandb.log_artifact(artifact)
    
    def finish(self):
        """Finish the W&B run."""
        if self.enabled and self._initialized:
            wandb.finish()
            self._initialized = False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()

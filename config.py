#!/usr/bin/env python3
"""
Configuration management for Hyperlora training.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any


@dataclass
class TrainingConfig:
    """Configuration for Hyperlora training."""
    
    # Model paths
    config_path: str = "./configs/stable-diffusion/v1-inference.yaml"
    ckpt_path: str = "./models/sd-v1-4.ckpt"
    
    # LoRA/HyperLoRA settings
    lora_rank: int = 1
    lora_alpha: float = 8.0
    target_modules: List[str] = field(default_factory=lambda: ["attn2.to_k", "attn2.to_v"])
    clip_size: int = 768
    
    # Training hyperparameters
    iterations: int = 200
    gradient_accumulation_steps: int = 1
    lr: float = 3e-5
    image_size: int = 512
    ddim_steps: int = 50
    ddim_eta: float = 0.0
    start_guidance: float = 9.0
    negative_guidance: float = 2.0
    seed: Optional[int] = None
    internal_lr: float = 1e-4
    batch_size: int = 1
    
    # Logging / tracking
    use_wandb: bool = False
    log_from: int = 0
    logging_dir: str = "logs"
    mixed_precision: Optional[str] = None
    
    # Output / data
    output_dir: str = "output"
    data_dir: str = "data10"
    neutral_concepts_file: str = "assets/neutral_concepts.json"
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainingConfig":
        """
        Create configuration from parsed command-line arguments.
        
        Args:
            args: Parsed argparse.Namespace
            
        Returns:
            TrainingConfig instance
        """
        return cls(
            # Model paths
            config_path=args.config_path,
            ckpt_path=args.ckpt_path,
            
            # LoRA/HyperLoRA
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            clip_size=args.clip_size,
            
            # Training hyperparameters
            iterations=args.iterations,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            lr=args.lr,
            image_size=args.image_size,
            ddim_steps=args.ddim_steps,
            ddim_eta=args.ddim_eta,
            start_guidance=args.start_guidance,
            negative_guidance=args.negative_guidance,
            seed=args.seed,
            internal_lr=args.internal_lr,
            batch_size=args.batch_size,
            
            # Logging / tracking
            use_wandb=args.use_wandb,
            log_from=args.log_from,
            logging_dir=args.logging_dir,
            mixed_precision=args.mixed_precision,
            
            # Output / data
            output_dir=args.output_dir,
            data_dir=args.data_dir,
            neutral_concepts_file=args.neutral_concepts_file,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of config
        """
        return {
            # Model paths
            "config_path": self.config_path,
            "ckpt_path": self.ckpt_path,
            
            # LoRA/HyperLoRA
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules,
            "clip_size": self.clip_size,
            
            # Training hyperparameters
            "iterations": self.iterations,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "lr": self.lr,
            "image_size": self.image_size,
            "ddim_steps": self.ddim_steps,
            "ddim_eta": self.ddim_eta,
            "start_guidance": self.start_guidance,
            "negative_guidance": self.negative_guidance,
            "seed": self.seed,
            "internal_lr": self.internal_lr,
            "batch_size": self.batch_size,
            
            # Logging / tracking
            "use_wandb": self.use_wandb,
            "log_from": self.log_from,
            "logging_dir": self.logging_dir,
            "mixed_precision": self.mixed_precision,
            
            # Output / data
            "output_dir": self.output_dir,
            "data_dir": self.data_dir,
            "neutral_concepts_file": self.neutral_concepts_file,
        }
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid
        """
        if self.lora_rank < 1:
            raise ValueError(f"lora_rank must be >= 1, got {self.lora_rank}")
        
        if self.lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be > 0, got {self.lora_alpha}")
        
        if self.iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {self.iterations}")
        
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        
        if self.image_size < 1:
            raise ValueError(f"image_size must be >= 1, got {self.image_size}")
        
        if self.ddim_steps < 1:
            raise ValueError(f"ddim_steps must be >= 1, got {self.ddim_steps}")
        
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        
        if self.mixed_precision not in [None, "no", "fp16", "bf16"]:
            raise ValueError(f"mixed_precision must be one of [None, 'no', 'fp16', 'bf16'], got {self.mixed_precision}")
        
        # Validate paths exist
        if not Path(self.neutral_concepts_file).exists():
            raise ValueError(f"neutral_concepts_file does not exist: {self.neutral_concepts_file}")


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for Hyperlora training.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="LoRA/HyperLoRA Fine-tuning for Stable Diffusion (Accelerate)"
    )

    # Model paths
    parser.add_argument(
        "--config_path",
        type=str,
        default="./configs/stable-diffusion/v1-inference.yaml",
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./models/sd-v1-4.ckpt",
        help="Path to model checkpoint",
    )

    # LoRA/HyperLoRA
    parser.add_argument(
        "--lora_rank", 
        type=int, 
        default=1, 
        help="LoRA rank parameter"
    )
    parser.add_argument(
        "--lora_alpha", 
        type=float, 
        default=8, 
        help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=["attn2.to_k", "attn2.to_v"],
        help="Target modules for LoRA injection",
    )
    parser.add_argument(
        "--clip_size", 
        type=int, 
        default=768, 
        help="CLIP embedding size"
    )

    # Optimization/Training
    parser.add_argument(
        "--iterations", 
        type=int, 
        default=200, 
        help="Number of training iterations"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", 
        type=int, 
        default=1,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=3e-5, 
        help="Learning rate"
    )
    parser.add_argument(
        "--image_size", 
        type=int, 
        default=512, 
        help="Image size for training"
    )
    parser.add_argument(
        "--ddim_steps", 
        type=int, 
        default=50, 
        help="DDIM sampling steps"
    )
    parser.add_argument(
        "--ddim_eta", 
        type=float, 
        default=0.0, 
        help="DDIM eta"
    )
    parser.add_argument(
        "--start_guidance", 
        type=float, 
        default=9.0, 
        help="Starting guidance scale"
    )
    parser.add_argument(
        "--negative_guidance", 
        type=float, 
        default=2.0, 
        help="Negative guidance scale"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="Random seed"
    )
    parser.add_argument(
        "--internal_lr", 
        type=float, 
        default=1e-4, 
        help="Simulated lr for hypernetwork"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )

    # Logging / tracking
    parser.add_argument(
        "--use-wandb", 
        action="store_true", 
        dest="use_wandb",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--log_from", 
        type=int, 
        default=0, 
        help="Log debug images from iteration"
    )
    parser.add_argument(
        "--logging_dir", 
        type=str, 
        default="logs",
        help="Base logging directory (used by Accelerate trackers)."
    )
    parser.add_argument(
        "--mixed_precision", 
        type=str, 
        default=None, 
        choices=["no", "fp16", "bf16"],
        help="Override Accelerate mixed precision (fp16/bf16)."
    )

    # Output / data
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="output", 
        help="Directory to save models"
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data10", 
        help="Directory with prompt json files"
    )
    parser.add_argument(
        "--neutral_concepts_file",
        type=str,
        default="assets/neutral_concepts.json",
        help="Path to JSON file with neutral concept lists for regularization training",
    )

    return parser.parse_args()

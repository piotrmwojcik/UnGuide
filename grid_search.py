#!/usr/bin/env python3
import yaml
import subprocess
import itertools
import os
import argparse
import random
from pathlib import Path

def run_grid_search():
    parser = argparse.ArgumentParser(description="Grid search for UnGuide training")
    parser.add_argument("--base_config", type=str, default="configs/celebrity/train_celebrity_100.yaml", help="Base config file to start from")
    parser.add_argument("--output_dir", type=str, default="grid_search_configs_celebrity_100", help="Directory to save generated configs")
    parser.add_argument("--launch", action="store_true", help="Actually launch the training jobs sequentially")
    parser.add_argument("--slurm", action="store_true", help="Generate SLURM submission scripts instead of running directly")
    parser.add_argument("--random", action="store_true", help="Perform randomized grid search instead of exhaustive")
    parser.add_argument("--num_iterations", type=int, default=100, help="Number of random combinations to sample if --random is used")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random sampling")
    args = parser.parse_args()

    if args.random:
        random.seed(args.seed)

    # Define the grid
    grid = {
        "remove_weight": [5.0, 10.0, 20.0, 50.0],
        "retain_weight": [1.0,2.0, 5.0, 10.0],
        "rank": [1, 2, 4, 8],
        "lora_alpha": [1.0e-06, 1.0e-05, 1.0e-04, 1.0e-03],
        "hyper_train_steps": [150, 300, 600, 1000],
        "max_train_steps": [1000, 2000, 3000, 4000],
    }

    # Load base config
    with open(args.base_config, 'r') as f:
        full_config = yaml.safe_load(f)
    
    # Get the first key (the config name)
    config_name_base = list(full_config.keys())[0]
    base_params = full_config[config_name_base]

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("grid_search_slurm", exist_ok=True) if args.slurm else None
    os.makedirs("logs", exist_ok=True) if args.slurm else None

    # Generate combinations
    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    if args.random:
        if args.num_iterations < len(combinations):
            print(f"Sampling {args.num_iterations} random combinations out of {len(combinations)} total.")
            combinations = random.sample(combinations, args.num_iterations)
        else:
            print(f"Num iterations ({args.num_iterations}) >= total combinations ({len(combinations)}). Running all.")

    print(f"Generated {len(combinations)} combinations to run.")

    for i, combo in enumerate(combinations):
        # Create a unique name for this run
        combo_str = "_".join([f"{k}{v}" for k, v in combo.items()])
        run_name = f"GS_{i}_{combo_str}"
        
        # Build the new config
        params = base_params.copy()
        params.update(combo)
        
        # Update output paths to be unique
        params["output_dir"] = os.path.join("output", "grid_search", run_name)
        params["final_save_path"] = os.path.join(params["output_dir"], "model")
        
        # Save the new config
        new_config_path = os.path.join(args.output_dir, f"{run_name}.yaml")
        with open(new_config_path, 'w') as f:
            yaml.dump({run_name: params}, f)
        
        if args.slurm:
            slurm_path = os.path.join("grid_search_slurm", f"submit_{run_name}.sh")
            with open(slurm_path, 'w') as f:
                f.write(f"#!/bin/bash\n")
                f.write(f"#SBATCH --job-name={run_name}\n")
                f.write(f"#SBATCH --time=0-8:00:00\n")
                f.write(f"#SBATCH --nodes=1\n")
                f.write(f"#SBATCH --account=plgroomagine-gpu-a100\n")
                f.write(f"#SBATCH --partition=plgrid-gpu-a100\n")
                f.write(f"#SBATCH --gres=gpu:1\n")
                f.write(f"#SBATCH --cpus-per-task=16\n")
                f.write(f"#SBATCH --mem=100G\n")
                f.write(f"#SBATCH --output=logs/slurm_{run_name}.out\n")
                f.write(f"#SBATCH --error=logs/slurm_{run_name}.err\n\n")
                f.write(f"source .venv/bin/activate\n")
                f.write(f"accelerate launch --num_processes 1 train_simple.py --config {new_config_path}\n")
            print(f"Created SLURM script: {slurm_path}")
        
        elif args.launch:
            print(f"[{i+1}/{len(combinations)}] Launching: {run_name}")
            cmd = [
                "accelerate", "launch",
                "--num_processes", "1",
                "train_simple.py",
                "--config", new_config_path
            ]
            subprocess.run(cmd)
        else:
            print(f"Created config: {new_config_path}")

    if args.slurm:
        print("\nTo submit all jobs, you can use:")
        print("for f in grid_search_slurm/*.sh; do sbatch $f; done")
    elif not args.launch:
        print("\nConfigs generated. Use --launch to run them sequentially or --slurm to generate submission scripts.")

if __name__ == "__main__":
    run_grid_search()

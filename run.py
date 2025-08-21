#!/usr/bin/env python3
"""
Cross-platform launcher script for pokegym
Automatically detects the platform and runs the appropriate command
"""
import os
import sys
import platform
import subprocess

def main():
    # Environment variables for debugging (uncomment if needed)
    debug_env = {
        # "TORCH_LOGS": "+dynamo",
        # "TORCHDYNAMO_EXTENDED_DEBUG_CPP": "1",
        # "TORCHDYNAMO_EXTENDED_DEBUG_GUARD_ADDED": "1941*s0*s1 < 2147483648",
        # "TORCH_COMPILE_DEBUG": "1",
    }
    
    # Set environment variables
    env = os.environ.copy()
    env.update(debug_env)
    
    # Main command
    cmd = [
        sys.executable, "demo.py", 
        "--env", "pokemon_red", 
        "--mode", "train", 
        "--vec", "multiprocessing", 
        "--track", 
        "--wandb-entity", "leanke"
    ]
    
    print(f"Running on {platform.system()} {platform.machine()}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the command
        result = subprocess.run(cmd, env=env, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
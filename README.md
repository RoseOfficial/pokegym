# Pokegym

Pokemon Red Gymnasium environment for reinforcement learning

## Installation

1. Clone the repo to your local machine and install it.
2. Fork the repo and clone your fork to your local machine.

```sh
pip install -e . 
```

### Platform Support

This environment supports:
- **Linux** (Primary platform)
- **macOS** 
- **Windows** (Full compatibility added)

All path handling, compilation flags, and system calls have been made cross-platform compatible.

### Running

#### Cross-platform (Recommended)
```sh
python run.py
```

#### Platform-specific
**Linux/macOS:**
```sh
./run.sh
```

**Windows:**
```cmd
run.bat
```

### Structure

/envs: Contains environment wrappers for customizing and extending the base environment.

/pokegym: Holds the core environment files. Modify these files to alter the environment's behavior.

/config: Configuration files for setting parameters and environment settings.

## Powered by Pufferlib

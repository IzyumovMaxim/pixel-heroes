import yaml
import os
from pathlib import Path

from paths import CONFIGS_DIR

def load_config(config_name: str):
    with open(CONFIGS_DIR / f"{config_name}.yaml", "r") as f:
        return yaml.safe_load(f)

def save_config(config: dict, config_name: str, save_dir: Path = CONFIGS_DIR):
    os.makedirs(save_dir, exist_ok=True)
    with open(save_dir / f"{config_name}.yaml", "w") as f:
        yaml.safe_dump(config, f)
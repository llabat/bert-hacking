from pathlib import Path
import yaml


def load_config(config_path):
    """
    Load a YAML configuration file.
    config_path : str | Path
        Path to the YAML config file.
    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config
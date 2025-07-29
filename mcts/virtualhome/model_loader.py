import argparse
import yaml
import json
import os

def load_model_config(config_path: str, model_name: str) -> dict:
    """
    Load the model configuration for a given model from a YAML or JSON file.

    Args:
        config_path: Path to the config file (YAML or JSON). Can be absolute or relative to this module.
        model_name: Key of the model in the config file.

    Returns:
        A dict containing 'loading_params', 'goal_sample_params', and 'sampling_params'.

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If the model_name is not found or required sections are missing.
        ValueError: If the file format is unsupported.
    """
    # Resolve relative paths against this module's directory
    if not os.path.isabs(config_path):
        base_dir = os.path.dirname(__file__)
        config_path = os.path.join(base_dir, config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    _, ext = os.path.splitext(config_path)
    with open(config_path, 'r') as f:
        if ext.lower() in ('.yaml', '.yml'):
            cfg = yaml.safe_load(f)
        elif ext.lower() == '.json':
            cfg = json.load(f)
        else:
            raise ValueError("Unsupported config file format. Use .yaml, .yml, or .json")

    if model_name not in cfg:
        raise KeyError(f"Model '{model_name}' not found in config file.")

    model_cfg = cfg[model_name]
    required_keys = ['loading_params', 'goal_sample_params', 'sampling_params']
    missing = [k for k in required_keys if k not in model_cfg]
    if missing:
        raise KeyError(f"Missing required config sections for '{model_name}': {missing}")

    return model_cfg


def load_model(model_name: str, config_path: str):
    """
    Factory that loads an HF LocalLLM and returns the model plus its goal and sampling params.

    Args:
        model_name: Key of the model in the config file (and HF repo ID).
        config_path: Path to the YAML/JSON config file.

    Returns:
        llm: An instance of LocalLLM.
        goal_sample_params: Dict of hyperparameters for goal interpretation.
        sampling_params: Dict of hyperparameters for planning sampling.
    """
    model_cfg = load_model_config(config_path, model_name)
    from hf_llm import LocalLLM
    llm = LocalLLM(**model_cfg['loading_params'])
    return llm, model_cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load model configuration")
    parser.add_argument("--config", "-c", required=True,
                        help="Path to model config file (YAML or JSON)")
    parser.add_argument("--model", "-m", required=True,
                        help="Model key to load from config file")
    parser.add_argument("--format", "-f", choices=['json', 'yaml'],
                        help="Output format: json or yaml (defaults to printing to stdout)")
    args = parser.parse_args()

    try:
        cfg = load_model_config(args.config, args.model)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

    if args.format == 'json':
        print(json.dumps(cfg, indent=2))
    elif args.format == 'yaml':
        print(yaml.safe_dump(cfg, sort_keys=False))
    else:
        import pprint
        pprint.pprint(cfg)

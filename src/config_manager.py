#!/usr/bin/env python3
"""
Training Configuration Manager

Manages reusable training configurations stored as JSON files in ~/abliterate/configs/.
Configs contain abliteration settings (excluding model_path/output_path which are
selected at runtime).
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Schema version for future migrations
CONFIG_SCHEMA_VERSION = "1.0"

# Fields that should NOT be saved in configs (set at runtime)
EXCLUDED_FIELDS = {"model_path", "output_path", "harmful_prompts", "harmless_prompts"}

# All saveable settings fields with their defaults
DEFAULT_SETTINGS = {
    "num_prompts": 30,
    "direction_multiplier": 1.0,
    "norm_preservation": True,
    "filter_prompts": True,
    "device": "cuda",
    "dtype": "float16",
    # Winsorization
    "use_winsorization": False,
    "winsorize_percentile": 0.995,
    # Magnitude clipping
    "use_magnitude_clipping": False,
    "magnitude_clip_percentile": 0.99,
    # Numerical stability
    "use_welford_mean": True,
    "use_float64_subtraction": True,
    # Null-space
    "use_null_space": False,
    "preservation_prompts_path": None,
    "null_space_rank_ratio": 0.95,
    # Adaptive weighting
    "use_adaptive_weighting": False,
    # Projected abliteration (orthogonalize against harmless)
    "use_projected_refusal": True,
    # Biprojection
    "use_biprojection": False,
    "use_per_neuron_norm": False,
    "target_layer_types": None,
    "use_harmless_boundary": False,
    "harmless_clamp_ratio": 0.1,
    "num_measurement_layers": 2,
    "intervention_range": [0.25, 0.95],
}


# ==============================================================================
# Directory Management
# ==============================================================================


def get_configs_dir() -> Path:
    """
    Return the path to the configs directory (~/abliterate/job_configs/).
    Creates the directory if it doesn't exist.
    """
    configs_dir = Path.home() / "abliterate" / "job_configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    return configs_dir


def get_config_file_path(config_name: str) -> Path:
    """
    Return the full path to a config file.

    Args:
        config_name: Name of the config (without .json extension)

    Returns:
        Path to the config file
    """
    sanitized = sanitize_config_name(config_name)
    return get_configs_dir() / f"{sanitized}.json"


def sanitize_config_name(name: str) -> str:
    """
    Sanitize a config name for use as a filename.

    - Converts to lowercase
    - Replaces spaces and special chars with hyphens
    - Removes consecutive hyphens
    - Strips leading/trailing hyphens

    Args:
        name: Raw config name from user

    Returns:
        Sanitized filename-safe string
    """
    # Convert to lowercase
    sanitized = name.lower()
    # Replace spaces and common separators with hyphens
    sanitized = re.sub(r"[\s_]+", "-", sanitized)
    # Remove any characters that aren't alphanumeric or hyphens
    sanitized = re.sub(r"[^a-z0-9\-]", "", sanitized)
    # Remove consecutive hyphens
    sanitized = re.sub(r"-+", "-", sanitized)
    # Strip leading/trailing hyphens
    sanitized = sanitized.strip("-")
    # Ensure we have something
    if not sanitized:
        sanitized = "config"
    return sanitized


# ==============================================================================
# CRUD Operations
# ==============================================================================


def list_configs() -> list[dict]:
    """
    List all saved training configs with their metadata.

    Returns:
        List of dicts with keys: name, description, created_at, updated_at, file_path
        Sorted by updated_at (most recent first)
    """
    configs_dir = get_configs_dir()
    configs = []

    for config_file in configs_dir.glob("*.json"):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            metadata = data.get("metadata", {})
            configs.append({
                "name": metadata.get("name", config_file.stem),
                "description": metadata.get("description", ""),
                "created_at": metadata.get("created_at", ""),
                "updated_at": metadata.get("updated_at", ""),
                "file_path": str(config_file),
                "filename": config_file.stem,
            })
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load config {config_file}: {e}")
            # Include in list but mark as corrupted
            configs.append({
                "name": config_file.stem,
                "description": "[Error loading config]",
                "created_at": "",
                "updated_at": "",
                "file_path": str(config_file),
                "filename": config_file.stem,
                "error": str(e),
            })

    # Sort by updated_at, most recent first
    configs.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return configs


def load_training_config(config_name: str) -> Optional[dict]:
    """
    Load a training config by name.

    Args:
        config_name: Name of the config to load

    Returns:
        Full config dict with metadata and settings, or None if not found
    """
    config_path = get_config_file_path(config_name)

    if not config_path.exists():
        logger.warning(f"Config not found: {config_name}")
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate structure
        if "metadata" not in data or "settings" not in data:
            logger.warning(f"Invalid config structure: {config_name}")
            return None

        return data
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to load config {config_name}: {e}")
        return None


def save_training_config(
    config_name: str,
    settings: dict,
    description: str = "",
    overwrite: bool = False,
) -> tuple[bool, str]:
    """
    Save settings as a new training config.

    Args:
        config_name: Name for the config
        settings: Dict of abliteration settings
        description: Optional description
        overwrite: If True, overwrite existing config

    Returns:
        Tuple of (success: bool, message: str)
    """
    config_path = get_config_file_path(config_name)
    sanitized_name = sanitize_config_name(config_name)

    if config_path.exists() and not overwrite:
        return False, f"Config '{sanitized_name}' already exists. Use overwrite=True to replace."

    # Clean settings - remove excluded fields and apply defaults for missing
    clean_settings = {}
    for key, default_value in DEFAULT_SETTINGS.items():
        if key in settings:
            clean_settings[key] = settings[key]
        else:
            clean_settings[key] = default_value

    now = datetime.now(timezone.utc).isoformat()

    config_data = {
        "metadata": {
            "name": config_name,  # Keep original name for display
            "description": description,
            "created_at": now,
            "updated_at": now,
            "version": CONFIG_SCHEMA_VERSION,
        },
        "settings": clean_settings,
    }

    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"Saved config: {config_path}")
        return True, f"Config saved as '{sanitized_name}'"
    except IOError as e:
        logger.error(f"Failed to save config: {e}")
        return False, f"Failed to save config: {e}"


def update_training_config(
    config_name: str,
    settings: Optional[dict] = None,
    description: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Update an existing training config's settings and/or description.

    Args:
        config_name: Name of the config to update
        settings: New settings (if None, keeps existing)
        description: New description (if None, keeps existing)

    Returns:
        Tuple of (success: bool, message: str)
    """
    existing = load_training_config(config_name)
    if existing is None:
        return False, f"Config '{config_name}' not found"

    # Update metadata
    if description is not None:
        existing["metadata"]["description"] = description
    existing["metadata"]["updated_at"] = datetime.now(timezone.utc).isoformat()

    # Update settings if provided
    if settings is not None:
        clean_settings = {}
        for key, default_value in DEFAULT_SETTINGS.items():
            if key in settings:
                clean_settings[key] = settings[key]
            elif key in existing["settings"]:
                clean_settings[key] = existing["settings"][key]
            else:
                clean_settings[key] = default_value
        existing["settings"] = clean_settings

    config_path = get_config_file_path(config_name)

    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)

        logger.info(f"Updated config: {config_path}")
        return True, f"Config '{config_name}' updated"
    except IOError as e:
        logger.error(f"Failed to update config: {e}")
        return False, f"Failed to update config: {e}"


def delete_training_config(config_name: str) -> tuple[bool, str]:
    """
    Delete a training config.

    Args:
        config_name: Name of the config to delete

    Returns:
        Tuple of (success: bool, message: str)
    """
    config_path = get_config_file_path(config_name)

    if not config_path.exists():
        return False, f"Config '{config_name}' not found"

    try:
        config_path.unlink()
        logger.info(f"Deleted config: {config_path}")
        return True, f"Config '{config_name}' deleted"
    except IOError as e:
        logger.error(f"Failed to delete config: {e}")
        return False, f"Failed to delete config: {e}"


def config_exists(config_name: str) -> bool:
    """
    Check if a config with this name exists.

    Args:
        config_name: Name of the config

    Returns:
        True if config exists
    """
    return get_config_file_path(config_name).exists()


# ==============================================================================
# Conversion Helpers
# ==============================================================================


def settings_from_abliteration_config(config: dict) -> dict:
    """
    Extract saveable settings from a runtime abliteration config dict.

    Removes model_path, output_path, and other runtime-only fields.

    Args:
        config: Runtime config dict from get_abliteration_config()

    Returns:
        Dict containing only saveable settings
    """
    settings = {}
    for key in DEFAULT_SETTINGS:
        if key in config:
            settings[key] = config[key]
    return settings


def apply_config_to_runtime(
    loaded_config: dict,
    model_path: str,
    output_path: str,
) -> dict:
    """
    Merge loaded config settings with runtime model/output paths.

    Args:
        loaded_config: Config dict loaded from file (has metadata and settings)
        model_path: Runtime model path
        output_path: Runtime output path

    Returns:
        Complete config dict ready for run_abliteration()
    """
    # Start with loaded settings
    runtime_config = dict(loaded_config.get("settings", {}))

    # Add runtime paths
    runtime_config["model_path"] = model_path
    runtime_config["output_path"] = output_path

    return runtime_config


# ==============================================================================
# Validation
# ==============================================================================


def validate_config_settings(settings: dict) -> tuple[bool, list[str]]:
    """
    Validate config settings.

    Args:
        settings: Settings dict to validate

    Returns:
        Tuple of (is_valid: bool, errors: list of error messages)
    """
    errors = []

    # Validate numeric ranges
    if "num_prompts" in settings:
        if not isinstance(settings["num_prompts"], int) or settings["num_prompts"] < 1:
            errors.append("num_prompts must be a positive integer")

    if "direction_multiplier" in settings:
        val = settings["direction_multiplier"]
        if not isinstance(val, (int, float)) or val < 0 or val > 2:
            errors.append("direction_multiplier must be between 0 and 2")

    if "winsorize_percentile" in settings:
        val = settings["winsorize_percentile"]
        if not isinstance(val, (int, float)) or val < 0.5 or val > 1:
            errors.append("winsorize_percentile must be between 0.5 and 1")

    if "magnitude_clip_percentile" in settings:
        val = settings["magnitude_clip_percentile"]
        if not isinstance(val, (int, float)) or val < 0.5 or val > 1:
            errors.append("magnitude_clip_percentile must be between 0.5 and 1")

    if "null_space_rank_ratio" in settings:
        val = settings["null_space_rank_ratio"]
        if not isinstance(val, (int, float)) or val < 0.5 or val > 1:
            errors.append("null_space_rank_ratio must be between 0.5 and 1")

    if "harmless_clamp_ratio" in settings:
        val = settings["harmless_clamp_ratio"]
        if not isinstance(val, (int, float)) or val < 0 or val > 1:
            errors.append("harmless_clamp_ratio must be between 0 and 1")

    if "intervention_range" in settings:
        val = settings["intervention_range"]
        if not isinstance(val, (list, tuple)) or len(val) != 2:
            errors.append("intervention_range must be a list of two floats")
        elif val[0] >= val[1] or val[0] < 0 or val[1] > 1:
            errors.append("intervention_range must be [start, end] where 0 <= start < end <= 1")

    # Validate device
    if "device" in settings:
        if settings["device"] not in ("cuda", "cpu", "mps"):
            errors.append("device must be 'cuda', 'cpu', or 'mps'")

    # Validate dtype
    if "dtype" in settings:
        if settings["dtype"] not in ("float16", "bfloat16", "float32"):
            errors.append("dtype must be 'float16', 'bfloat16', or 'float32'")

    return len(errors) == 0, errors


def get_default_settings() -> dict:
    """Return a copy of the default settings."""
    return dict(DEFAULT_SETTINGS)

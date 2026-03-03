#!/usr/bin/env python3
import argparse
import gc
import json
import logging
import os
import random
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model_utils import load_model_and_tokenizer
from src.cli_components import get_versioned_path, get_prompts_path
from utils.refusal_detector import LogLikelihoodRefusalDetector, RefusalDetectorConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Mapping of torch.dtype to string for JSON serialization
DTYPE_TO_STRING = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.bool: "bool",
}


def make_json_serializable(obj):
    """Recursively convert non-JSON-serializable objects to serializable types.

    Handles torch.dtype, torch.Tensor, Path objects, and nested dicts/lists.
    """
    if isinstance(obj, torch.dtype):
        return DTYPE_TO_STRING.get(obj, str(obj))
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        # Handle dataclass-like objects that might have dtype attributes
        return make_json_serializable(vars(obj))
    return obj


def _clean_config_dtypes_recursive(obj, visited=None) -> None:
    """Recursively clean torch.dtype objects from a config object in-place.

    Args:
        obj: The object to clean (config, quantization_config, etc.)
        visited: Set of visited object ids to avoid infinite recursion
    """
    if visited is None:
        visited = set()

    # Avoid infinite recursion on circular references
    obj_id = id(obj)
    if obj_id in visited:
        return
    visited.add(obj_id)

    # Skip basic types that can't contain dtype attributes
    if obj is None or isinstance(obj, (str, int, float, bool, bytes)):
        return

    # Handle dict-like objects
    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            if isinstance(value, torch.dtype):
                obj[key] = DTYPE_TO_STRING.get(value, str(value))
            else:
                _clean_config_dtypes_recursive(value, visited)
        return

    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        for item in obj:
            _clean_config_dtypes_recursive(item, visited)
        return

    # For objects with attributes, iterate over all attributes
    if hasattr(obj, '__dict__'):
        for attr_name in list(vars(obj).keys()):
            try:
                value = getattr(obj, attr_name)
                if isinstance(value, torch.dtype):
                    setattr(obj, attr_name, DTYPE_TO_STRING.get(value, str(value)))
                elif not callable(value) and not attr_name.startswith('_'):
                    _clean_config_dtypes_recursive(value, visited)
            except (AttributeError, TypeError):
                # Some attributes might be properties that raise on access
                pass


def clean_model_config_for_save(model) -> None:
    """Clean the model's config to ensure it can be serialized to JSON.

    Some models (especially FP8 quantized models like Mistral3) have config
    attributes that contain torch.dtype objects which can't be JSON serialized.
    This function converts those to strings in-place by recursively traversing
    all config attributes.
    """
    if not hasattr(model, 'config'):
        return

    config = model.config

    # Recursively clean all dtype objects in the entire config tree
    _clean_config_dtypes_recursive(config)

    # Also explicitly handle quantization_config via to_dict() for completeness
    # Some quantization configs have special serialization logic
    if hasattr(config, 'quantization_config') and config.quantization_config is not None:
        qconfig = config.quantization_config
        if hasattr(qconfig, 'to_dict'):
            try:
                qdict = qconfig.to_dict()
                # Convert any dtype objects to strings and update attributes
                for key, value in list(qdict.items()):
                    if isinstance(value, torch.dtype):
                        cleaned_value = DTYPE_TO_STRING.get(value, str(value))
                        if hasattr(qconfig, key):
                            setattr(qconfig, key, cleaned_value)
            except Exception as e:
                logger.debug(f"Could not clean quantization_config via to_dict: {e}")


def _needs_manual_save(model) -> bool:
    """Check if model needs manual saving due to irreversible weight conversions.

    Detects FP8 dequantized models and other cases where transformers'
    save_pretrained would fail with NotImplementedError.
    """
    # Check for quantization config indicators
    if hasattr(model, 'config') and hasattr(model.config, 'quantization_config'):
        qconfig = model.config.quantization_config
        if qconfig is not None:
            # Check various FP8/dequantization indicators
            is_dequantized = getattr(qconfig, 'dequantize', False)
            quant_method = str(getattr(qconfig, 'quant_method', '')).lower()
            if is_dequantized or 'fp8' in quant_method:
                return True

    # Check for hf_quantizer (indicates quantization was applied during load)
    if hasattr(model, 'hf_quantizer') and model.hf_quantizer is not None:
        return True

    # Check for weight conversion hooks/operations
    if hasattr(model, '_hf_hook') and model._hf_hook is not None:
        return True

    # Check for Mistral3 models which typically need special handling
    model_type = getattr(getattr(model, 'config', None), 'model_type', '')
    if 'mistral3' in str(model_type).lower():
        return True

    return False


def _save_model_manual(model, tokenizer, output_path: Path, target_dtype: torch.dtype = torch.float16) -> None:
    """Manually save model using safetensors, bypassing transformers' weight conversion.

    This function is specifically designed to handle FP8 dequantized models and ensure
    the saved model is compatible with GGUF conversion tools like llama.cpp.

    Key operations:
    1. Filter out FP8 scale tensors (_scale_inv, _scale) that break GGUF conversion
    2. Convert bfloat16 weights to float16 for broader GGUF compatibility
    3. Clear quantization config that no longer applies after dequantization

    Args:
        model: The model to save
        tokenizer: The tokenizer to save
        output_path: Directory to save to
        target_dtype: Target dtype for weights (default: float16 for GGUF compatibility)
    """
    from safetensors.torch import save_file

    logger.info("Using manual save to bypass weight conversion")

    # Clear quantization-related config
    if hasattr(model, 'config'):
        if hasattr(model.config, 'quantization_config'):
            model.config.quantization_config = None
        if hasattr(model.config, '_pre_quantization_dtype'):
            delattr(model.config, '_pre_quantization_dtype')

    # Get state dict and save with safetensors
    raw_state_dict = model.state_dict()

    # FP8 scale tensor patterns that must be filtered out for GGUF compatibility
    # These tensors cause "ValueError: Can not map tensor" errors in convert_hf_to_gguf
    fp8_scale_patterns = ('_scale_inv', '_scale', '.fp8_scale')

    # Filter out FP8 scale tensors and convert dtypes
    state_dict = {}
    filtered_count = 0
    converted_count = 0

    for key, tensor in raw_state_dict.items():
        # Skip FP8 scale tensors - they break GGUF conversion
        if any(pattern in key for pattern in fp8_scale_patterns):
            filtered_count += 1
            continue

        # Convert bfloat16 to float16 for GGUF compatibility
        # llama.cpp's convert_hf_to_gguf downcasts bf16 to fp16 which can cause artifacts
        # By doing the conversion here with proper rounding, we get cleaner results
        if tensor.dtype == torch.bfloat16 and target_dtype == torch.float16:
            tensor = tensor.to(torch.float16)
            converted_count += 1
        elif tensor.dtype != target_dtype and tensor.is_floating_point():
            # Convert other floating point tensors to target dtype
            tensor = tensor.to(target_dtype)
            converted_count += 1

        state_dict[key] = tensor

    if filtered_count > 0:
        logger.info(f"Filtered out {filtered_count} FP8 scale tensors for GGUF compatibility")
    if converted_count > 0:
        logger.info(f"Converted {converted_count} tensors to {target_dtype} for GGUF compatibility")

    # Split into shards if needed (max 5GB per shard)
    max_shard_size = 5 * 1024 * 1024 * 1024  # 5GB
    total_size = sum(t.numel() * t.element_size() for t in state_dict.values())

    if total_size > max_shard_size:
        # Shard the model
        logger.info(f"Model size ({total_size / 1e9:.2f}GB) exceeds shard limit, splitting...")
        current_shard = {}
        current_size = 0
        shard_idx = 1
        weight_map = {}

        for key, tensor in state_dict.items():
            tensor_size = tensor.numel() * tensor.element_size()
            if current_size + tensor_size > max_shard_size and current_shard:
                # Save current shard
                shard_name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
                save_file(current_shard, output_path / shard_name)
                for k in current_shard:
                    weight_map[k] = shard_name
                current_shard = {}
                current_size = 0
                shard_idx += 1

            current_shard[key] = tensor
            current_size += tensor_size

        # Save last shard
        if current_shard:
            shard_name = f"model-{shard_idx:05d}-of-{shard_idx:05d}.safetensors"
            save_file(current_shard, output_path / shard_name)
            for k in current_shard:
                weight_map[k] = shard_name

        # Rename shards with correct total count and save index
        total_shards = shard_idx
        for i in range(1, total_shards + 1):
            old_name = output_path / f"model-{i:05d}-of-XXXXX.safetensors"
            new_name = output_path / f"model-{i:05d}-of-{total_shards:05d}.safetensors"
            if old_name.exists():
                old_name.rename(new_name)
                # Update weight map
                for k, v in weight_map.items():
                    if v == f"model-{i:05d}-of-XXXXX.safetensors":
                        weight_map[k] = f"model-{i:05d}-of-{total_shards:05d}.safetensors"

        # Save index file
        index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        with open(output_path / "model.safetensors.index.json", "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
    else:
        # Single file save
        save_file(state_dict, output_path / "model.safetensors")

    # Save config manually
    model.config.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)


def _untie_shared_weights(model) -> None:
    """Clone shared/tied weights to break memory sharing for safetensors save.

    Models with tied weights (e.g., embed_tokens and lm_head sharing memory)
    cause safetensors to fail. This function detects and clones shared weights
    dynamically by checking data pointers.
    """
    state_dict = model.state_dict()
    cloned_any = False

    # Build a map of data pointers to tensor names
    ptr_to_names: dict[int, list[str]] = {}
    for name, tensor in state_dict.items():
        ptr = tensor.data_ptr()
        if ptr not in ptr_to_names:
            ptr_to_names[ptr] = []
        ptr_to_names[ptr].append(name)

    # Find groups of tensors that share memory
    for ptr, names in ptr_to_names.items():
        if len(names) > 1:
            logger.info(f"Found shared weights: {names}")
            # Clone all but the first tensor in each group
            for name in names[1:]:
                logger.info(f"  Cloning: {name}")
                # Get the original tensor and clone it
                original_tensor = state_dict[name]
                cloned_tensor = original_tensor.clone()

                # Find the module and replace its weight/bias
                parts = name.rsplit(".", 1)  # Split off the param name (weight/bias)
                if len(parts) == 2:
                    module_path, param_name = parts
                    module = model
                    for part in module_path.split("."):
                        module = getattr(module, part, None)
                        if module is None:
                            break

                    if module is not None and hasattr(module, param_name):
                        setattr(module, param_name, torch.nn.Parameter(cloned_tensor))
                        cloned_any = True

    if cloned_any:
        # Also set tie_word_embeddings to False in config if it exists
        if hasattr(model.config, "tie_word_embeddings"):
            model.config.tie_word_embeddings = False
        # Handle nested configs (VL models)
        if hasattr(model.config, "text_config") and hasattr(model.config.text_config, "tie_word_embeddings"):
            model.config.text_config.tie_word_embeddings = False


def save_model_safe(model, tokenizer, output_path: Path, target_dtype: torch.dtype = torch.float16) -> None:
    """Save model and tokenizer, handling FP8 dequantized models specially.

    For dequantized FP8 models (loaded with FineGrainedFP8Config(dequantize=True)),
    transformers' save_pretrained fails with NotImplementedError when trying to
    reverse the dequantization transform. This function bypasses that by saving
    the state dict directly with safetensors.

    For GGUF compatibility, weights are converted to float16 (the default) since
    llama.cpp's converter can have issues with bfloat16 weights.

    Args:
        model: The model to save
        tokenizer: The tokenizer to save
        output_path: Directory to save to (must exist)
        target_dtype: Target dtype for weights (default: float16 for GGUF compatibility)
    """
    # Clean model config first
    clean_model_config_for_save(model)

    # Check if this model needs manual saving
    if _needs_manual_save(model):
        _save_model_manual(model, tokenizer, output_path, target_dtype=target_dtype)
        # Update config.json to reflect the actual weight dtype
        _update_config_dtype(output_path, target_dtype)
        return

    # Try normal save, fall back to manual if it fails
    try:
        model.save_pretrained(output_path, safe_serialization=True, max_shard_size="5GB")
        tokenizer.save_pretrained(output_path)
    except (NotImplementedError, RuntimeError) as e:
        error_msg = str(e)
        if "share memory" in error_msg or "shared" in error_msg.lower():
            logger.warning("Model has tied/shared weights, using safe clone before save...")
            # Clone tied weights to break sharing
            _untie_shared_weights(model)
            try:
                model.save_pretrained(output_path, safe_serialization=True, max_shard_size="5GB")
                tokenizer.save_pretrained(output_path)
                return
            except Exception as e2:
                logger.warning(f"Save after untying still failed: {e2}")
        else:
            logger.warning(f"save_pretrained failed: {e}")
        logger.info("Falling back to manual save...")
        _save_model_manual(model, tokenizer, output_path, target_dtype=target_dtype)
        _update_config_dtype(output_path, target_dtype)


def _update_config_dtype(output_path: Path, target_dtype: torch.dtype) -> None:
    """Update config.json to reflect the actual weight dtype after conversion.

    This is important for GGUF conversion tools that read torch_dtype from config.

    Args:
        output_path: Path to the model directory
        target_dtype: The dtype the weights were converted to
    """
    config_path = output_path / "config.json"
    if not config_path.exists():
        return

    dtype_str = DTYPE_TO_STRING.get(target_dtype, "float16")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        config["torch_dtype"] = dtype_str

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        logger.debug(f"Updated config.json torch_dtype to {dtype_str}")
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Could not update config.json dtype: {e}")


def get_package_root() -> Path:
    """Get the root directory of the abliteration package.

    This allows the CLI to be invoked from anywhere and still find
    the prompts directory correctly on both Windows and Unix systems.
    """
    return Path(__file__).resolve().parent.parent


def get_default_prompts_path(filename: str) -> str:
    """Get the absolute path to a prompts file.

    Checks user prompts directory (~/.abliterate/prompts/) first,
    then falls back to the package prompts directory.

    Args:
        filename: Name of the prompts file (e.g., 'harmful.txt')

    Returns:
        Absolute path to the prompts file as a string
    """
    return str(get_prompts_path(filename))


def copy_vision_files(source_path: Path, dest_path: Path) -> list[str]:
    """Copy vision-related files from source model to destination.

    This is needed for Vision-Language (VL) models so that the abliterated
    model can still be converted to GGUF with mmproj support.

    Args:
        source_path: Path to the original model directory
        dest_path: Path to the abliterated model directory

    Returns:
        List of filenames that were copied
    """
    # Files needed for VL model vision encoder conversion
    vision_files = [
        "preprocessor_config.json",
        "processor_config.json",
        "image_processor_config.json",
        "chat_template.json",
        # Vision encoder weights (if stored separately)
        "vision_encoder.safetensors",
        "vision_model.safetensors",
        "visual.safetensors",
        # Qwen VL specific
        "mrope_sections.txt",
    ]

    # Also copy any files with "vision" or "image" in the name
    vision_patterns = ["*vision*", "*image*", "*visual*", "*processor*", "*pixtral*"]

    copied_files = []
    source_path = Path(source_path)
    dest_path = Path(dest_path)

    # Copy specific vision files
    for filename in vision_files:
        src_file = source_path / filename
        if src_file.exists():
            dst_file = dest_path / filename
            if not dst_file.exists():
                shutil.copy2(src_file, dst_file)
                copied_files.append(filename)
                logger.debug(f"Copied vision file: {filename}")

    # Copy files matching vision patterns
    for pattern in vision_patterns:
        for src_file in source_path.glob(pattern):
            if src_file.is_file():
                dst_file = dest_path / src_file.name
                if not dst_file.exists():
                    shutil.copy2(src_file, dst_file)
                    copied_files.append(src_file.name)
                    logger.debug(f"Copied vision file: {src_file.name}")

    return copied_files


def copy_essential_model_files(source_path: Path, dest_path: Path) -> list[str]:
    """Copy essential model files that save_pretrained doesn't handle.

    This includes tokenizer vocabulary files, generation config, and other
    model-specific files that aren't part of the core model/tokenizer save.

    Args:
        source_path: Path to the original model directory
        dest_path: Path to the abliterated model directory

    Returns:
        List of filenames that were copied
    """
    # Essential files that should be copied if present
    essential_files = [
        # Generation config
        "generation_config.json",
        # Custom tokenizer files (various model families)
        "tekken.json",           # Ministral/Mistral tokenizer
        "merges.txt",            # BPE merges file
        "vocab.json",            # Vocabulary file
        "added_tokens.json",     # Additional tokens
        "special_tokens_map.json",  # Special token mappings
        # Model params
        "params.json",           # Mistral-style params
        # Chat templates (if not already saved by tokenizer)
        "chat_template.jinja",   # Jinja2 chat template
        "chat_template.json",    # JSON chat template
        # Other config files
        "preprocessor_config.json",
    ]

    copied_files = []
    source_path = Path(source_path)
    dest_path = Path(dest_path)

    for filename in essential_files:
        src_file = source_path / filename
        if src_file.exists():
            dst_file = dest_path / filename
            if not dst_file.exists():
                shutil.copy2(src_file, dst_file)
                copied_files.append(filename)
                logger.debug(f"Copied essential file: {filename}")

    return copied_files


def preserve_model_config(source_path: Path, dest_path: Path) -> None:
    """Preserve original model config.json fields after save_pretrained.

    When transformers saves a model, it may not preserve all config fields,
    especially for newer model architectures or custom fields. This function
    reads the original config.json and merges any missing fields into the
    saved config.json.

    Note: quantization_config is intentionally NOT preserved because models
    that were dequantized during loading (e.g., FP8 models with dequantize=True)
    no longer have the scale factors that the quantization config expects.
    Preserving it would cause loading errors when the model is later loaded.

    Args:
        source_path: Path to the original model directory
        dest_path: Path to the saved model directory
    """
    source_config_path = source_path / "config.json"
    dest_config_path = dest_path / "config.json"

    if not source_config_path.exists() or not dest_config_path.exists():
        logger.debug("Cannot preserve config: source or dest config.json not found")
        return

    try:
        # Read original config
        with open(source_config_path, "r", encoding="utf-8") as f:
            original_config = json.load(f)

        # Read saved config
        with open(dest_config_path, "r", encoding="utf-8") as f:
            saved_config = json.load(f)

        # Track fields that were preserved
        preserved_fields = []

        # Fields to NOT preserve - these are invalidated by dequantization
        # or other transformations during abliteration
        # These fields cause issues with GGUF conversion if present
        skip_fields = {
            "quantization_config",  # FP8 scale factors no longer exist after dequantization
            "_pre_quantization_dtype",  # Internal quantization state
            "_name_or_path",  # Can cause confusion with converted models
        }

        # Also check for and remove any nested quantization-related configs
        # that might confuse GGUF converters
        quant_related_keys = [k for k in saved_config.keys()
                              if 'quant' in k.lower() or 'fp8' in k.lower()]
        for key in quant_related_keys:
            skip_fields.add(key)

        # Remove fields that should not be present after dequantization
        removed_fields = []
        for field in skip_fields:
            if field in saved_config:
                del saved_config[field]
                removed_fields.append(field)

        # Merge missing fields from original to saved
        # Preserve original values - don't overwrite what transformers saved
        # unless the field is completely missing
        for key, value in original_config.items():
            if key not in saved_config and key not in skip_fields:
                saved_config[key] = value
                preserved_fields.append(key)

        # Also ensure architectures matches original (transformers might change it)
        if "architectures" in original_config:
            if saved_config.get("architectures") != original_config["architectures"]:
                saved_config["architectures"] = original_config["architectures"]
                if "architectures" not in preserved_fields:
                    preserved_fields.append("architectures")

        # Write merged config back
        if preserved_fields or removed_fields:
            with open(dest_config_path, "w", encoding="utf-8") as f:
                json.dump(saved_config, f, indent=2)
            if removed_fields:
                logger.info(f"Removed invalidated config fields: {removed_fields}")
            if preserved_fields:
                logger.info(f"Preserved {len(preserved_fields)} config fields from original: {preserved_fields}")

    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to preserve model config: {e}")


def _has_vision_weights(model_path: Path) -> bool:
    """Check if a model has actual vision encoder weights in its files.

    This is the definitive test for VL capability - checking for actual
    vision encoder weights rather than just architecture names.

    Args:
        model_path: Path to the model directory

    Returns:
        True if vision encoder weights are found
    """
    import struct

    vision_prefixes = [
        "vision_tower", "vision_model", "visual_encoder",
        "vision_encoder", "image_encoder", "vit.", "visual.",
        "model.vision_tower", "model.vision_model",
    ]

    # Check safetensors index for vision-related weight names
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            for weight_name in weight_map.keys():
                weight_lower = weight_name.lower()
                if any(prefix in weight_lower for prefix in vision_prefixes):
                    return True
        except (json.JSONDecodeError, IOError):
            pass

    # Check single safetensors file header
    single_safetensors = model_path / "model.safetensors"
    if single_safetensors.exists():
        try:
            with open(single_safetensors, "rb") as f:
                header_size = struct.unpack("<Q", f.read(8))[0]
                header_json = f.read(header_size).decode("utf-8")
                header = json.loads(header_json)
                for weight_name in header.keys():
                    if weight_name == "__metadata__":
                        continue
                    weight_lower = weight_name.lower()
                    if any(prefix in weight_lower for prefix in vision_prefixes):
                        return True
        except (struct.error, json.JSONDecodeError, IOError):
            pass

    return False


def is_vision_model(model_path: Path) -> bool:
    """Check if a model is a Vision-Language model.

    Uses a two-stage check:
    1. First checks architecture/config hints for VL capability
    2. Then verifies actual vision weights exist (to avoid false positives
       for models like Ministral 3B that use VL-capable architectures
       but don't have vision weights)

    Args:
        model_path: Path to the model directory

    Returns:
        True if the model has actual vision capability
    """
    config_path = model_path / "config.json"

    # First check if architecture/config suggests VL capability
    is_vl_architecture = False

    if not config_path.exists():
        # Fall back to name-based detection
        model_name_lower = model_path.name.lower()
        is_vl_architecture = any(kw in model_name_lower for kw in ["vl", "vision", "llava", "visual", "pixtral"])
    else:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            # Check for VL architectures
            vl_architectures = [
                "Qwen2VLForConditionalGeneration",
                "Qwen2_5_VLForConditionalGeneration",
                "Qwen3VLForConditionalGeneration",
                "LlavaForConditionalGeneration",
                "LlavaNextForConditionalGeneration",
                "MllamaForConditionalGeneration",
                "InternVLChatModel",
                "PaliGemmaForConditionalGeneration",
                "Idefics2ForConditionalGeneration",
                "MiniCPMV",
                "Phi3VForCausalLM",
                "Mistral3ForConditionalGeneration",
            ]

            architectures = config.get("architectures", [])
            for arch in architectures:
                if arch in vl_architectures:
                    is_vl_architecture = True
                    break

            # Check for vision_config
            if "vision_config" in config or "visual" in config:
                is_vl_architecture = True

            # Check model_type
            model_type = config.get("model_type", "").lower()
            if any(kw in model_type for kw in ["vl", "vision", "llava"]):
                is_vl_architecture = True

        except (json.JSONDecodeError, IOError):
            pass

    # If architecture suggests VL, verify by checking for actual vision weights
    # This prevents false positives for models like Ministral 3B
    if is_vl_architecture:
        return _has_vision_weights(model_path)

    return False


# Data Classes


@dataclass
class AbliterationConfig:
    """Configuration for the abliteration process."""

    model_path: str
    output_path: str
    harmful_prompts_path: str = field(default_factory=lambda: get_default_prompts_path("harmful.txt"))
    harmless_prompts_path: str = field(default_factory=lambda: get_default_prompts_path("harmless.txt"))
    num_prompts: Optional[int] = None  # Number of prompts to sample (None = use all)
    harmful_prompts: list[str] = field(default_factory=list)
    harmless_prompts: list[str] = field(default_factory=list)
    target_layers: Optional[list[int]] = None  # None = all layers
    extraction_layer_indices: Optional[list[int]] = None  # Layers to extract directions from
    use_mean_direction: bool = True  # Average directions across extraction layers
    normalize_directions: bool = True
    norm_preservation: bool = True  # Rescale weights to preserve Frobenius norm
    direction_multiplier: float = 1.0  # Scale the ablation strength
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32  # Computation dtype
    save_directions: bool = True  # Save computed refusal directions
    load_directions_path: Optional[str] = None  # Load pre-computed directions
    batch_size: int = 4
    max_new_tokens: int = 1  # We only need the first forward pass
    token_position: str = "last"  # Which token position to extract: "last", "mean", or int
    filter_harmful_prompts: bool = True  # Filter out harmful prompts that aren't actually refused
    refusal_test_max_tokens: int = 50  # Max tokens to generate when testing for refusal (legacy)
    refusal_test_batch_size: int = 16  # Batch size for refusal testing (larger = faster but more VRAM)
    refusal_threshold: float = -7.0  # Log-likelihood threshold for refusal detection (higher = more likely to refuse)
    refusal_keywords: list[str] = field(default_factory=lambda: [
        "I cannot", "I can't", "I'm not able", "I am not able",
        "I won't", "I will not", "I'm unable", "I am unable",
        "Sorry", "I apologize", "As an AI", "As a language model",
        "I'm an AI", "I am an AI", "not appropriate", "against my",
        "ethical", "illegal", "harmful", "dangerous", "unethical",
        "I must refuse", "I have to refuse", "cannot assist", "can't assist",
        "cannot help", "can't help", "cannot provide", "can't provide",
        "not able to help", "not able to assist", "not able to provide",
    ])

    # Advanced options: Winsorization (clips outlier activations)
    use_winsorization: bool = False  # Enable per-dimension Winsorization preprocessing
    winsorize_percentile: float = 0.995  # Clip values above this percentile

    # Advanced options: Global magnitude clipping
    use_magnitude_clipping: bool = False  # Enable global magnitude clipping (alternative to Winsorization)
    magnitude_clip_percentile: float = 0.99  # Clip to this percentile of absolute values

    # Advanced options: Numerical stability (from llm-abliteration)
    use_welford_mean: bool = True  # Use Welford's algorithm for stable mean computation
    use_float64_subtraction: bool = True  # Use float64 for mean difference (handles high cosine similarity)

    # Advanced options: Null-space constraints (preserves model capabilities)
    use_null_space: bool = False  # Enable null-space constrained abliteration
    preservation_prompts_path: Optional[str] = None  # Path to preservation prompts file
    null_space_rank_ratio: float = 0.95  # SVD rank ratio for null-space computation
    null_space_regularization: float = 1e-4  # Tikhonov regularization for numerical stability

    # Advanced options: Adaptive layer weighting
    use_adaptive_weighting: bool = False  # Enable per-layer adaptive weighting
    adaptive_position_center: float = 0.6  # Center of Gaussian position weighting (0-1)
    adaptive_position_sigma: float = 0.2  # Width of Gaussian position weighting

    # Advanced options: Projected abliteration
    use_projected_refusal: bool = True  # Orthogonalize refusal direction against harmless direction (recommended)

    # Advanced options: Biprojection mode
    use_biprojection: bool = False  # Enable biprojection (measure at high-quality layers, apply across range)
    use_per_neuron_norm: bool = False  # Use per-neuron norm preservation instead of Frobenius

    # Biprojection configuration
    measurement_layers: Optional[list[int]] = None  # Layers to measure refusal direction (auto if None)
    intervention_layers: Optional[list[int]] = None  # Layers to apply ablation (auto if None)
    num_measurement_layers: int = 2  # How many top-quality layers for measurement
    intervention_range: tuple[float, float] = (0.25, 0.95)  # Depth range for intervention as fraction

    # Layer type targeting
    target_layer_types: Optional[list[str]] = None  # e.g., ['o_proj', 'down_proj'], None = all types

    # Harmless direction boundary clamping
    use_harmless_boundary: bool = False  # Clamp ablation to preserve harmless direction
    harmless_clamp_ratio: float = 0.1  # How much to clamp toward harmless (0.1 = 10%)

    # Quality-based layer selection
    use_quality_selection: bool = False  # Use SNR-based layer quality scoring
    min_quality_threshold: float = 0.0  # Skip layers below this quality score

    # Layer target map integration (data-driven per-layer weighting)
    layer_target_map_path: Optional[str] = None  # Path to layer_target_map.json
    per_layer_multipliers: Optional[dict[int, float]] = None  # Direct multipliers per layer
    exclude_layers: Optional[list[int]] = None  # Layers to skip entirely
    layer_targeting_mode: str = "none"  # "none", "adaptive", "target_map"

    # Unmapped layer behavior (when using target map)
    unmapped_layer_behavior: str = "skip"  # "skip" or "default"
    unmapped_layer_multiplier: float = 1.0  # Multiplier for layers not in map (when behavior="default")

    # Dynamic layer targeting: extract from ALL layers and apply per-layer directions/projectors
    dynamic_layer_targeting: bool = False  # When True: extract all layers, use per-layer directions

    # Hybrid architecture strategy (Qwen3.5, etc.)
    hybrid_strategy: str = "auto"  # "auto" (detect+apply), "uniform" (current behavior)
    hybrid_full_attn_weight: float = 1.0  # Ablation multiplier for full attention layers
    hybrid_linear_attn_weight: float = 0.4  # Ablation multiplier for linear attention layers
    hybrid_skip_recurrent_proj: bool = True  # Skip in_proj_a, in_proj_b during ablation
    hybrid_skip_state_proj: bool = False  # Also skip in_proj_qkv, in_proj_z (more conservative)

    # KL divergence monitoring
    use_kl_monitoring: bool = False
    kl_reference_prompts_path: Optional[str] = None  # default: preservation.txt
    kl_num_reference_prompts: int = 50
    kl_top_k: int = 200
    kl_batch_size: int = 4

    # KL auto-tune (binary search over direction_multiplier)
    use_kl_auto_tune: bool = False
    kl_threshold: float = 0.5  # max mean KL (nats)
    kl_search_min: float = 0.1
    kl_search_max: float = 2.0
    kl_search_tolerance: float = 0.01
    kl_max_search_iterations: int = 15


@dataclass
class RefusalDirections:
    """Container for computed refusal directions."""

    directions: dict[int, torch.Tensor]  # layer_idx -> direction vector
    mean_direction: Optional[torch.Tensor] = None
    metadata: dict = field(default_factory=dict)

    # Biprojection support
    harmless_directions: Optional[dict[int, torch.Tensor]] = None  # layer_idx -> harmless mean
    quality_scores: Optional[dict[int, dict[str, float]]] = None  # layer_idx -> {snr, cos_sim, quality}
    biprojected_direction: Optional[torch.Tensor] = None  # Combined direction from measurement layers

    def save(self, path: str):
        """Save directions to disk."""
        save_dict = {
            "directions": {k: v.cpu() for k, v in self.directions.items()},
            "mean_direction": self.mean_direction.cpu() if self.mean_direction is not None else None,
            "metadata": self.metadata,
            # Biprojection fields
            "harmless_directions": {k: v.cpu() for k, v in self.harmless_directions.items()} if self.harmless_directions else None,
            "quality_scores": self.quality_scores,
            "biprojected_direction": self.biprojected_direction.cpu() if self.biprojected_direction is not None else None,
        }
        torch.save(save_dict, path)
        logger.info(f"Saved refusal directions to {path}")

    @classmethod
    def load(cls, path: str) -> "RefusalDirections":
        """Load directions from disk."""
        data = torch.load(path, map_location="cpu", weights_only=True)
        return cls(
            directions=data["directions"],
            mean_direction=data["mean_direction"],
            metadata=data.get("metadata", {}),
            # Biprojection fields (with backwards compatibility)
            harmless_directions=data.get("harmless_directions"),
            quality_scores=data.get("quality_scores"),
            biprojected_direction=data.get("biprojected_direction"),
        )


# Hybrid Architecture Detection


@dataclass
class HybridArchitectureInfo:
    """Information about hybrid attention architectures (e.g., Qwen3.5)."""

    is_hybrid: bool
    layer_types: list[str]  # "full_attention" or "linear_attention" per layer
    full_attention_indices: list[int]
    linear_attention_indices: list[int]
    full_attention_interval: int  # e.g. 4 for Qwen3.5 (every 4th layer is full attention)


def detect_hybrid_architecture(model_path: str) -> HybridArchitectureInfo:
    """
    Detect hybrid attention architecture from model config.json.

    Reads layer_types from config.json to identify models with mixed
    full attention and linear attention layers (e.g., Qwen3.5 with
    GatedDeltaNet linear attention).

    Args:
        model_path: Path to model directory containing config.json

    Returns:
        HybridArchitectureInfo with detected architecture details.
        Returns is_hybrid=False if no hybrid architecture detected.
    """
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return HybridArchitectureInfo(
            is_hybrid=False, layer_types=[], full_attention_indices=[],
            linear_attention_indices=[], full_attention_interval=0,
        )

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (json.JSONDecodeError, OSError):
        return HybridArchitectureInfo(
            is_hybrid=False, layer_types=[], full_attention_indices=[],
            linear_attention_indices=[], full_attention_interval=0,
        )

    # Look for layer_types in text_config (VL models) or top-level
    layer_types_raw = None
    if "text_config" in config and "layer_types" in config["text_config"]:
        layer_types_raw = config["text_config"]["layer_types"]
    elif "layer_types" in config:
        layer_types_raw = config["layer_types"]

    if not layer_types_raw or not isinstance(layer_types_raw, list):
        return HybridArchitectureInfo(
            is_hybrid=False, layer_types=[], full_attention_indices=[],
            linear_attention_indices=[], full_attention_interval=0,
        )

    # Normalize layer type names
    # Common patterns: "full_attention"/"sliding_window"/"global" vs "linear_attention"/"gated_deltanet"
    layer_types = []
    full_indices = []
    linear_indices = []

    full_attn_names = {"full_attention", "sliding_window", "global", "sdpa"}
    linear_attn_names = {"linear_attention", "gated_deltanet", "linear"}

    for i, lt in enumerate(layer_types_raw):
        lt_lower = lt.lower().strip()
        if lt_lower in full_attn_names:
            layer_types.append("full_attention")
            full_indices.append(i)
        elif lt_lower in linear_attn_names:
            layer_types.append("linear_attention")
            linear_indices.append(i)
        else:
            # Unknown type - treat as linear attention (conservative)
            layer_types.append("linear_attention")
            linear_indices.append(i)
            logger.warning(f"Unknown layer type '{lt}' at index {i}, treating as linear_attention")

    # Check if it's actually hybrid (has both types)
    is_hybrid = len(full_indices) > 0 and len(linear_indices) > 0

    # Compute interval (distance between full attention layers)
    interval = 0
    if len(full_indices) >= 2:
        intervals = [full_indices[i + 1] - full_indices[i] for i in range(len(full_indices) - 1)]
        # Use the most common interval
        from collections import Counter
        interval = Counter(intervals).most_common(1)[0][0]
    elif len(full_indices) == 1:
        interval = len(layer_types_raw)  # Only one full attention layer

    return HybridArchitectureInfo(
        is_hybrid=is_hybrid,
        layer_types=layer_types,
        full_attention_indices=full_indices,
        linear_attention_indices=linear_indices,
        full_attention_interval=interval,
    )


# Activation Extraction


class ActivationExtractor:
    """Extracts hidden state activations from transformer layers."""

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, config: AbliterationConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.activations: dict[int, list[torch.Tensor]] = {}
        self.hooks = []
        # Welford accumulators for streaming mean computation
        self.welford_accumulators: dict[int, WelfordMeanAccumulator] = {}
        self.use_welford = config.use_welford_mean

    def _get_layers(self):
        """Get the transformer layers from the model."""
        # Handle different model architectures
        if hasattr(self.model, "model"):
            # Vision-Language models (Qwen2-VL, Qwen3-VL, LLaVA, etc.)
            # Structure: model.model.model.layers (VL wrapper -> multimodal -> text -> layers)
            if hasattr(self.model.model, "model") and hasattr(self.model.model.model, "layers"):
                return self.model.model.model.layers
            # VL MoE models, Mistral3, Qwen3.5 (Qwen3-VL-MoE, Qwen2-VL-MoE, Mistral3, Qwen3_5, etc.)
            # Structure: model.model.language_model.layers
            if hasattr(self.model.model, "language_model") and hasattr(self.model.model.language_model, "layers"):
                return self.model.model.language_model.layers
            # Standard text models (Llama, Qwen, etc.)
            # Structure: model.model.layers
            if hasattr(self.model.model, "layers"):
                return self.model.model.layers
            elif hasattr(self.model.model, "decoder") and hasattr(self.model.model.decoder, "layers"):
                return self.model.model.decoder.layers
        # VL models with language_model attribute (some LLaVA variants, InternVL, GLM4v, etc.)
        if hasattr(self.model, "language_model"):
            if hasattr(self.model.language_model, "model") and hasattr(self.model.language_model.model, "layers"):
                return self.model.language_model.model.layers
            # GLM4v and similar: language_model.transformer.layers
            if hasattr(self.model.language_model, "transformer") and hasattr(self.model.language_model.transformer, "layers"):
                return self.model.language_model.transformer.layers
            if hasattr(self.model.language_model, "layers"):
                return self.model.language_model.layers
        if hasattr(self.model, "transformer"):
            if hasattr(self.model.transformer, "h"):
                return self.model.transformer.h
            elif hasattr(self.model.transformer, "layers"):
                return self.model.transformer.layers
        if hasattr(self.model, "gpt_neox") and hasattr(self.model.gpt_neox, "layers"):
            return self.model.gpt_neox.layers
        # GptOss and similar: may have backbone or encoder wrapper
        if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "layers"):
            return self.model.backbone.layers
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layers"):
            return self.model.encoder.layers
        # Models with layers directly on the wrapper (no inner model)
        if hasattr(self.model, "layers"):
            return self.model.layers

        # Debug: print model structure to help identify the correct path
        structure_info = self._get_model_structure_info()
        raise ValueError(
            f"Could not find layers in model architecture: {type(self.model)}\n"
            f"Model structure:\n{structure_info}\n"
            f"Please add support for this architecture in ActivationExtractor._get_layers()"
        )

    def _get_model_structure_info(self, max_depth: int = 3) -> str:
        """Get a string representation of the model's top-level structure for debugging."""
        lines = []

        def explore(obj, prefix="", depth=0):
            if depth >= max_depth:
                return
            for name in dir(obj):
                if name.startswith("_"):
                    continue
                try:
                    attr = getattr(obj, name)
                    if isinstance(attr, torch.nn.ModuleList):
                        lines.append(f"{prefix}{name}: ModuleList[{len(attr)}]")
                    elif isinstance(attr, torch.nn.Module):
                        lines.append(f"{prefix}{name}: {type(attr).__name__}")
                        explore(attr, prefix + "  ", depth + 1)
                except Exception:
                    pass

        explore(self.model)
        return "\n".join(lines[:50])  # Limit output

    def _create_hook(self, layer_idx: int, hidden_dim: Optional[int] = None):
        """Create a forward hook for a specific layer.

        Args:
            layer_idx: Index of the layer to hook
            hidden_dim: Hidden dimension size (required for Welford mode)
        """

        def hook(module, input, output):
            # Handle different output formats
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Extract based on token position config
            if self.config.token_position == "last":
                # Get the last non-padding token
                extracted = hidden_states[:, -1, :]
            elif self.config.token_position == "mean":
                extracted = hidden_states.mean(dim=1)
            elif isinstance(self.config.token_position, int):
                extracted = hidden_states[:, self.config.token_position, :]
            else:
                extracted = hidden_states[:, -1, :]

            # Always store raw activations for quality scores, harmless boundary, etc.
            if layer_idx not in self.activations:
                self.activations[layer_idx] = []
            self.activations[layer_idx].append(extracted.detach().cpu())

            # Additionally update Welford accumulators for more stable mean computation
            if self.use_welford:
                if layer_idx not in self.welford_accumulators:
                    dim = extracted.shape[-1]
                    self.welford_accumulators[layer_idx] = WelfordMeanAccumulator(
                        hidden_dim=dim, device="cpu", dtype=torch.float32
                    )
                self.welford_accumulators[layer_idx].update(extracted.detach().cpu())

        return hook

    def register_hooks(self, layer_indices: Optional[list[int]] = None):
        """Register forward hooks on specified layers."""
        layers = self._get_layers()
        num_layers = len(layers)

        if layer_indices is None:
            # Default to middle-to-later layers where refusal is typically encoded
            layer_indices = list(range(num_layers // 4, 3 * num_layers // 4))

        for idx in layer_indices:
            if 0 <= idx < num_layers:
                hook = layers[idx].register_forward_hook(self._create_hook(idx))
                self.hooks.append(hook)

        logger.info(f"Registered hooks on layers: {layer_indices}")
        return layer_indices

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear_activations(self):
        """Clear stored activations and Welford accumulators."""
        self.activations = {}
        self.welford_accumulators = {}

    @torch.no_grad()
    def extract_activations(self, prompts: list[str]) -> dict[int, torch.Tensor]:
        """Extract activations for a list of prompts.

        Returns:
            Dict mapping layer_idx to concatenated activations [num_prompts, hidden_dim].
            If use_welford is enabled, means are computed via Welford's algorithm but
            full activations are still returned for quality score computation.
        """
        self.clear_activations()
        self.model.eval()

        for i in tqdm(range(0, len(prompts), self.config.batch_size), desc="Extracting activations"):
            batch_prompts = prompts[i : i + self.config.batch_size]

            # Tokenize with chat template if available
            if hasattr(self.tokenizer, "apply_chat_template"):
                formatted_prompts = []
                for prompt in batch_prompts:
                    messages = [{"role": "user", "content": prompt}]
                    formatted = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    formatted_prompts.append(formatted)
            else:
                formatted_prompts = batch_prompts

            # Use longer max_length to accommodate models with long system prompts
            # (e.g., Ministral-3 has ~500 token system prompt)
            # Also set padding_side to left for proper batch handling
            original_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = "left"
            inputs = self.tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.config.device)
            self.tokenizer.padding_side = original_padding_side

            # Forward pass to trigger hooks
            _ = self.model(**inputs)

        # Concatenate all activations for each layer
        result = {}
        for layer_idx, acts in self.activations.items():
            result[layer_idx] = torch.cat(acts, dim=0)

        return result

    def get_welford_means(self) -> dict[int, torch.Tensor]:
        """Get means computed via Welford's algorithm during extraction.

        Only available if use_welford=True was set during extraction.

        Returns:
            Dict mapping layer_idx to mean activation [hidden_dim]
        """
        return {
            layer_idx: acc.get_mean()
            for layer_idx, acc in self.welford_accumulators.items()
        }


# Adaptive Layer Weighting


def compute_adaptive_layer_weights(
    num_layers: int,
    center: float = 0.6,
    sigma: float = 0.2,
    min_weight: float = 0.1,
) -> dict[int, float]:
    """
    Compute adaptive weights for each layer using a Gaussian distribution.

    Research shows refusal behavior is more concentrated in middle-to-later layers.
    This function assigns higher weights to layers around the specified center
    position, allowing the ablation to be stronger where it matters most.

    Args:
        num_layers: Total number of layers in the model
        center: Center of the Gaussian as fraction of depth (0-1)
        sigma: Standard deviation of the Gaussian as fraction of depth
        min_weight: Minimum weight for any layer (to ensure all layers get some ablation)

    Returns:
        Dictionary mapping layer indices to weights (0.0 - 1.0)
    """
    import math

    weights = {}
    center_layer = center * (num_layers - 1)
    sigma_layers = sigma * num_layers

    for i in range(num_layers):
        # Gaussian weight centered at center_layer
        gaussian_weight = math.exp(-0.5 * ((i - center_layer) / sigma_layers) ** 2)
        # Apply minimum weight floor
        weights[i] = max(gaussian_weight, min_weight)

    # Normalize so max weight is 1.0
    max_weight = max(weights.values())
    if max_weight > 0:
        weights = {k: v / max_weight for k, v in weights.items()}

    return weights


# Layer Target Map Loading and Validation


def load_layer_target_map(path: str) -> dict:
    """
    Load and parse a layer target map from a JSON file.

    The layer target map provides data-driven per-layer abliteration weights
    based on feature distribution analysis. It identifies which layers to
    target aggressively, protect, or exclude entirely.

    Args:
        path: Path to the layer_target_map.json file

    Returns:
        Dictionary containing:
            - per_layer_multipliers: {int: float} mapping layer index to multiplier
            - exclude_layers: list[int] of layers to skip
            - target_layer_indices: list[int] of layers to abliterate
            - recommended_center_ratio: float for Gaussian fallback
            - recommended_sigma_ratio: float for Gaussian fallback
            - metadata: dict with layer_stats, aggressive_layers, protected_layers
    """
    with open(path, "r", encoding="utf-8") as f:
        raw_config = json.load(f)

    # Convert string keys to int for layer_multipliers
    per_layer_multipliers = {}
    if "layer_multipliers" in raw_config:
        per_layer_multipliers = {
            int(k): float(v) for k, v in raw_config["layer_multipliers"].items()
        }
    elif "abliteration_config" in raw_config and "per_layer_multipliers" in raw_config["abliteration_config"]:
        per_layer_multipliers = {
            int(k): float(v) for k, v in raw_config["abliteration_config"]["per_layer_multipliers"].items()
        }

    # Extract exclude_layers
    exclude_layers = raw_config.get("excluded_layers", [])
    if not exclude_layers and "abliteration_config" in raw_config:
        exclude_layers = raw_config["abliteration_config"].get("exclude_layers", [])

    # Extract target layer indices
    target_layer_indices = raw_config.get("target_layer_indices", [])

    # Extract Gaussian parameters for fallback
    recommended_center_ratio = raw_config.get("recommended_center_ratio", 0.6)
    recommended_sigma_ratio = raw_config.get("recommended_sigma_ratio", 0.2)

    # Extract metadata
    layer_stats = {}
    if "layer_stats" in raw_config:
        layer_stats = {int(k): v for k, v in raw_config["layer_stats"].items()}

    aggressive_layers = raw_config.get("aggressive_layers", [])
    protected_layers = raw_config.get("protected_layers", [])

    result = {
        "per_layer_multipliers": per_layer_multipliers,
        "exclude_layers": exclude_layers,
        "target_layer_indices": target_layer_indices,
        "recommended_center_ratio": recommended_center_ratio,
        "recommended_sigma_ratio": recommended_sigma_ratio,
        "metadata": {
            "layer_stats": layer_stats,
            "aggressive_layers": aggressive_layers,
            "protected_layers": protected_layers,
            "version": raw_config.get("version", "unknown"),
            "generated_at": raw_config.get("generated_at"),
            "analysis_params": raw_config.get("analysis_params", {}),
        },
    }

    logger.info(f"Loaded layer target map from {path}")
    logger.info(f"  - Target layers: {len(target_layer_indices)}")
    logger.info(f"  - Excluded layers: {len(exclude_layers)}")
    logger.info(f"  - Aggressive layers: {len(aggressive_layers)}")
    logger.info(f"  - Protected layers: {len(protected_layers)}")

    return result


def validate_layer_target_map(target_map: dict, num_layers: int) -> list[str]:
    """
    Validate a layer target map against a model's architecture.

    Checks that layer indices are valid and warns about gaps in coverage.

    Args:
        target_map: Parsed layer target map from load_layer_target_map()
        num_layers: Total number of transformer layers in the model

    Returns:
        List of warning messages (empty if no issues found)
    """
    warnings = []

    # Check layer indices don't exceed model layers
    per_layer_multipliers = target_map.get("per_layer_multipliers", {})
    for layer_idx in per_layer_multipliers.keys():
        if layer_idx >= num_layers:
            warnings.append(
                f"Layer {layer_idx} in target map exceeds model layers ({num_layers})"
            )

    for layer_idx in target_map.get("exclude_layers", []):
        if layer_idx >= num_layers:
            warnings.append(
                f"Excluded layer {layer_idx} exceeds model layers ({num_layers})"
            )

    # Check for layers not covered by map
    covered_layers = set(per_layer_multipliers.keys())
    covered_layers.update(target_map.get("exclude_layers", []))

    # All layers that should be in the map
    expected_layers = set(range(num_layers))
    missing_layers = expected_layers - covered_layers

    if missing_layers:
        # This is informational - depends on unmapped_layer_behavior setting
        missing_sorted = sorted(missing_layers)
        if len(missing_sorted) <= 10:
            warnings.append(f"Layers not in target map: {missing_sorted}")
        else:
            warnings.append(
                f"Layers not in target map: {missing_sorted[:5]}...{missing_sorted[-5:]} "
                f"({len(missing_sorted)} total)"
            )

    return warnings


# Winsorization for Outlier-Robust Direction Computation


def winsorize_activations(
    activations: torch.Tensor,
    percentile: float = 0.995,
) -> torch.Tensor:
    """
    Apply per-dimension Winsorization to clip extreme activation values.

    Critical for models like Gemma 3 where high-magnitude outliers
    can obscure the true refusal direction. Computes thresholds
    independently for each dimension.

    Args:
        activations: [num_samples, hidden_dim] activation tensor
        percentile: Clip values above this percentile (default: 0.995)

    Returns:
        Winsorized activations with outliers clipped per-dimension
    """
    # Compute per-dimension thresholds based on absolute values
    abs_acts = activations.abs()
    thresholds = torch.quantile(abs_acts.float(), percentile, dim=0)

    # Clip to thresholds (both positive and negative)
    clipped = torch.clamp(activations, -thresholds, thresholds)

    return clipped


def magnitude_clip_activations(
    activations: torch.Tensor,
    percentile: float = 0.99,
) -> torch.Tensor:
    """
    Apply global magnitude clipping.

    Unlike per-dimension Winsorization, this clips based on the global
    magnitude of each activation vector. This can be more effective when
    outliers are concentrated in specific samples rather than dimensions.

    Args:
        activations: [num_samples, hidden_dim] activation tensor
        percentile: Clip components to this percentile of absolute values (0.0 to 1.0)

    Returns:
        Clipped activations with extreme values clamped
    """
    original_dtype = activations.dtype
    activations_float = activations.float()

    # Compute global threshold across all values
    abs_activations = torch.abs(activations_float)
    threshold = torch.quantile(abs_activations.flatten(), percentile)

    # Symmetric clipping
    clipped = torch.clamp(activations_float, min=-threshold, max=threshold)

    return clipped.to(original_dtype)


# Welford's Online Algorithm for Numerically Stable Mean Computation


class WelfordMeanAccumulator:
    """
    Welford's online algorithm for numerically stable streaming mean computation.

    This is more numerically stable than the naive sum-and-divide approach,
    especially for large numbers of samples or when values have high variance.

    Reference: Welford, B. P. (1962). "Note on a method for calculating
    corrected sums of squares and products"
    """

    def __init__(self, hidden_dim: int, device: str = "cpu", dtype: torch.dtype = torch.float32):
        """
        Initialize the accumulator.

        Args:
            hidden_dim: Dimension of the activation vectors
            device: Device to store running statistics
            dtype: Data type for accumulation (recommend float32 for stability)
        """
        self.count = 0
        self.mean = torch.zeros(hidden_dim, device=device, dtype=dtype)
        self.hidden_dim = hidden_dim
        self.device = device
        self.dtype = dtype

    def update(self, batch: torch.Tensor) -> None:
        """
        Update running mean with a batch of samples.

        Uses the batch Welford update formula for efficiency:
            new_count = count + batch_size
            delta = batch_mean - mean
            mean += delta * batch_size / new_count

        Args:
            batch: [batch_size, hidden_dim] tensor of new samples
        """
        batch = batch.to(device=self.device, dtype=self.dtype)
        batch_size = batch.shape[0]

        if batch_size == 0:
            return

        batch_mean = batch.mean(dim=0)
        new_count = self.count + batch_size

        # Welford's update formula for batch
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (batch_size / new_count)
        self.count = new_count

    def get_mean(self) -> torch.Tensor:
        """Get the current running mean."""
        return self.mean

    def get_count(self) -> int:
        """Get the current sample count."""
        return self.count

    def reset(self) -> None:
        """Reset the accumulator."""
        self.count = 0
        self.mean = torch.zeros(self.hidden_dim, device=self.device, dtype=self.dtype)


def compute_mean_welford(
    activations_list: list[torch.Tensor],
    hidden_dim: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute mean using Welford's online algorithm for numerical stability.

    Args:
        activations_list: List of [batch_size, hidden_dim] tensors
        hidden_dim: Dimension of activation vectors
        device: Device for computation

    Returns:
        Mean activation vector [hidden_dim]
    """
    accumulator = WelfordMeanAccumulator(hidden_dim, device=device, dtype=torch.float32)

    for batch in activations_list:
        accumulator.update(batch)

    return accumulator.get_mean()


def compute_refusal_direction_float64(
    harmful_mean: torch.Tensor,
    harmless_mean: torch.Tensor,
) -> torch.Tensor:
    """
    Compute refusal direction using float64 for numerical stability.

    When harmful and harmless means have high cosine similarity (common in
    well-trained models), the difference can suffer from catastrophic
    cancellation in float32. Using float64 for the subtraction preserves
    precision in the small differences.

    Args:
        harmful_mean: Mean activation for harmful prompts [hidden_dim]
        harmless_mean: Mean activation for harmless prompts [hidden_dim]

    Returns:
        Refusal direction in float32 [hidden_dim]
    """
    # Perform subtraction in float64 to avoid catastrophic cancellation
    direction = (harmful_mean.double() - harmless_mean.double()).float()
    return direction


def orthogonalize_against_harmless(
    refusal_dir: torch.Tensor,
    harmless_mean: torch.Tensor,
) -> torch.Tensor:
    """
    Orthogonalize refusal direction against harmless direction.

    The raw refusal direction r = harmful_mean - harmless_mean contains two components:
        - r_parallel: Component aligned with harmless direction (general helpfulness magnitude)
        - r_perpendicular: Component orthogonal to harmless (mechanistically-specific refusal)

    Only the perpendicular component should be ablated. The parallel component represents
    variation in "how helpful" the model is, which is a confound that shouldn't be removed.

    Mathematical formulation:
        harmless_normalized = harmless_mean / ||harmless_mean||
        r_parallel = (r · harmless_normalized) * harmless_normalized
        r_perpendicular = r - r_parallel

    Args:
        refusal_dir: Raw refusal direction [hidden_dim]
        harmless_mean: Mean activation for harmless prompts [hidden_dim]

    Returns:
        Orthogonalized refusal direction (perpendicular to harmless) [hidden_dim]
    """
    # Work in float32 for consistency
    refusal_float = refusal_dir.float()
    harmless_float = harmless_mean.float()

    # Normalize harmless direction to unit vector
    harmless_normalized = F.normalize(harmless_float, dim=0)

    # Compute projection onto harmless direction
    projection_scalar = refusal_float @ harmless_normalized

    # Remove parallel component (keep only perpendicular)
    refusal_orthogonal = refusal_float - projection_scalar * harmless_normalized

    return refusal_orthogonal


# Biprojection: SNR-Based Layer Quality Scoring


def compute_direction_quality_scores(
    harmful_activations: dict[int, torch.Tensor],
    harmless_activations: dict[int, torch.Tensor],
    hybrid_info: Optional["HybridArchitectureInfo"] = None,
) -> dict[int, dict[str, float]]:
    """
    Compute SNR-based quality scores for refusal directions at each layer.

    Args:
        harmful_activations: Per-layer harmful prompt activations {layer_idx: tensor}
        harmless_activations: Per-layer harmless prompt activations {layer_idx: tensor}
        hybrid_info: Optional hybrid architecture info for layer type metadata

    Returns:
        Dict mapping layer_idx -> {snr, cos_sim, quality, refusal_norm, harmful_norm, harmless_norm, layer_type}
    """
    quality_scores = {}

    common_layers = set(harmful_activations.keys()) & set(harmless_activations.keys())

    for layer_idx in common_layers:
        harmful = harmful_activations[layer_idx].float()
        harmless = harmless_activations[layer_idx].float()

        # Compute means
        harmful_mean = harmful.mean(dim=0)
        harmless_mean = harmless.mean(dim=0)

        # Compute norms
        harmful_norm = harmful_mean.norm().item()
        harmless_norm = harmless_mean.norm().item()

        # Refusal direction (mean difference)
        refusal_dir = harmful_mean - harmless_mean
        refusal_norm = refusal_dir.norm().item()

        max_norm = max(harmful_norm, harmless_norm)
        snr = refusal_norm / (max_norm + 1e-8)

        # Cosine similarity between harmful and harmless means
        cos_sim = F.cosine_similarity(
            harmful_mean.unsqueeze(0),
            harmless_mean.unsqueeze(0)
        ).item()

        quality = snr * (1 - cos_sim)

        # Determine layer type from hybrid info
        layer_type = "unknown"
        if hybrid_info and hybrid_info.is_hybrid and layer_idx < len(hybrid_info.layer_types):
            layer_type = hybrid_info.layer_types[layer_idx]

        quality_scores[layer_idx] = {
            "snr": snr,
            "cos_sim": cos_sim,
            "quality": quality,
            "refusal_norm": refusal_norm,
            "harmful_norm": harmful_norm,
            "harmless_norm": harmless_norm,
            "layer_type": layer_type,
        }

        logger.debug(
            f"Layer {layer_idx}: SNR={snr:.3f}, cos_sim={cos_sim:.3f}, quality={quality:.3f}"
            + (f", type={layer_type}" if layer_type != "unknown" else "")
        )

    return quality_scores


def select_biprojection_layers(
    quality_scores: dict[int, dict[str, float]],
    num_layers: int,
    num_measurement_layers: int = 2,
    intervention_range: tuple[float, float] = (0.25, 0.95),
) -> tuple[list[int], list[int]]:
    """
    Select measurement and intervention layers for biprojection.

    Strategy:
        1. Measurement layers: Top N layers by quality score (where refusal is clearest)
        2. Intervention layers: Range from 25% to 95% of model depth

    Args:
        quality_scores: Per-layer quality scores from compute_direction_quality_scores()
        num_layers: Total number of transformer layers
        num_measurement_layers: How many top-quality layers to use for measurement (default: 2)
        intervention_range: (start, end) as fraction of depth for intervention

    Returns:
        (measurement_layers, intervention_layers)
    """
    # Sort layers by quality score (descending)
    sorted_layers = sorted(
        quality_scores.items(),
        key=lambda x: x[1]["quality"],
        reverse=True
    )

    # Top layers for measurement
    measurement_layers = [layer_idx for layer_idx, _ in sorted_layers[:num_measurement_layers]]

    # Intervention range
    start_layer = int(intervention_range[0] * num_layers)
    end_layer = int(intervention_range[1] * num_layers)
    intervention_layers = list(range(start_layer, end_layer + 1))

    logger.info(f"Biprojection measurement layers (top quality): {measurement_layers}")
    logger.info(f"Biprojection intervention layers: {start_layer}-{end_layer} ({len(intervention_layers)} layers)")

    return measurement_layers, intervention_layers


# Refusal Direction Computation


def compute_refusal_directions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: AbliterationConfig,
) -> RefusalDirections:
    """
    Compute refusal directions from contrastive prompt pairs.

    The refusal direction is computed as the mean difference between
    harmful and harmless prompt activations at each layer.
    """
    logger.info("Computing refusal directions from contrastive prompts...")

    extractor = ActivationExtractor(model, tokenizer, config)

    # Determine which layers to extract from
    layers = extractor._get_layers()
    num_layers = len(layers)

    # Detect hybrid architecture for extraction strategy
    hybrid_info = getattr(config, "_hybrid_info", None)

    if config.dynamic_layer_targeting:
        # Dynamic layer targeting: extract from ALL layers
        extraction_layers = list(range(num_layers))
        logger.info(f"Dynamic layer targeting enabled: extracting from all {num_layers} layers")
    elif hybrid_info and hybrid_info.is_hybrid and config.hybrid_strategy == "auto" and config.extraction_layer_indices is None:
        # Hybrid-aware extraction: all full attention layers + the linear attention layer before each
        extraction_layers = []
        for fa_idx in hybrid_info.full_attention_indices:
            if fa_idx - 1 >= 0 and fa_idx - 1 not in extraction_layers:
                extraction_layers.append(fa_idx - 1)
            extraction_layers.append(fa_idx)
        extraction_layers.sort()
        logger.info(
            f"Hybrid-aware extraction: {len(extraction_layers)} layers "
            f"({len(hybrid_info.full_attention_indices)} full attn + "
            f"{len(extraction_layers) - len(hybrid_info.full_attention_indices)} pre-full-attn linear)"
        )
    elif config.extraction_layer_indices is None:
        # Default: extract from middle-to-later layers
        extraction_layers = list(range(num_layers // 4, 3 * num_layers // 4))
    else:
        extraction_layers = config.extraction_layer_indices

    extractor.register_hooks(extraction_layers)

    # Store Welford means if enabled (computed during extraction)
    harmful_welford_means = {}
    harmless_welford_means = {}

    try:
        # Extract activations for harmful prompts
        logger.info(f"Extracting activations for {len(config.harmful_prompts)} harmful prompts...")
        harmful_activations = extractor.extract_activations(config.harmful_prompts)

        # Get Welford means before clearing (if enabled)
        if config.use_welford_mean:
            harmful_welford_means = extractor.get_welford_means()

        extractor.clear_activations()

        # Extract activations for harmless prompts
        logger.info(f"Extracting activations for {len(config.harmless_prompts)} harmless prompts...")
        harmless_activations = extractor.extract_activations(config.harmless_prompts)

        # Get Welford means (if enabled)
        if config.use_welford_mean:
            harmless_welford_means = extractor.get_welford_means()

    finally:
        extractor.remove_hooks()

    # Apply outlier clipping if enabled
    # Option 1: Per-dimension Winsorization
    if config.use_winsorization:
        logger.info(f"Applying per-dimension Winsorization (percentile={config.winsorize_percentile})...")
        for layer_idx in harmful_activations:
            harmful_activations[layer_idx] = winsorize_activations(
                harmful_activations[layer_idx], config.winsorize_percentile
            )
        for layer_idx in harmless_activations:
            harmless_activations[layer_idx] = winsorize_activations(
                harmless_activations[layer_idx], config.winsorize_percentile
            )

    # Option 2: Global magnitude clipping
    if config.use_magnitude_clipping:
        logger.info(f"Applying global magnitude clipping (percentile={config.magnitude_clip_percentile})...")
        for layer_idx in harmful_activations:
            harmful_activations[layer_idx] = magnitude_clip_activations(
                harmful_activations[layer_idx], config.magnitude_clip_percentile
            )
        for layer_idx in harmless_activations:
            harmless_activations[layer_idx] = magnitude_clip_activations(
                harmless_activations[layer_idx], config.magnitude_clip_percentile
            )

    # Log numerical stability and projection settings
    stability_features = []
    if config.use_welford_mean:
        stability_features.append("Welford mean")
    if config.use_float64_subtraction:
        stability_features.append("float64 subtraction")
    if config.use_projected_refusal:
        stability_features.append("projected (orthogonalized)")
    if stability_features:
        logger.info(f"Direction computation: {', '.join(stability_features)}")

    # Compute refusal directions as mean difference
    directions = {}
    for layer_idx in extraction_layers:
        # Check if we have data for this layer (either activations or Welford means)
        has_harmful = layer_idx in harmful_activations or layer_idx in harmful_welford_means
        has_harmless = layer_idx in harmless_activations or layer_idx in harmless_welford_means
        if has_harmful and has_harmless:
            # Use Welford means if available (more numerically stable), otherwise compute from activations
            if config.use_welford_mean and layer_idx in harmful_welford_means and layer_idx in harmless_welford_means:
                harmful_mean = harmful_welford_means[layer_idx]
                harmless_mean = harmless_welford_means[layer_idx]
            else:
                harmful_mean = harmful_activations[layer_idx].mean(dim=0)
                harmless_mean = harmless_activations[layer_idx].mean(dim=0)

            # Refusal direction: harmful - harmless
            # Use float64 for subtraction if enabled (handles high cosine similarity)
            if config.use_float64_subtraction:
                direction = compute_refusal_direction_float64(harmful_mean, harmless_mean)
            else:
                direction = harmful_mean - harmless_mean

            # Orthogonalize against harmless direction
            # This removes the "general helpfulness magnitude" confound, keeping only
            # the mechanistically-specific refusal component
            if config.use_projected_refusal:
                direction = orthogonalize_against_harmless(direction, harmless_mean)

            if config.normalize_directions:
                direction = F.normalize(direction, dim=0)

            directions[layer_idx] = direction.to(config.dtype)

            logger.debug(f"Layer {layer_idx}: direction norm = {direction.norm().item():.4f}")

    # Compute mean direction across all layers
    # NOTE: We always compute this, even when use_mean_direction=False,
    # because it's needed as a fallback for layers outside the extraction range
    mean_direction = None
    if directions:
        stacked = torch.stack(list(directions.values()))
        mean_direction = stacked.mean(dim=0)
        if config.normalize_directions:
            mean_direction = F.normalize(mean_direction, dim=0)

    # Biprojection: Compute quality scores for layer selection
    quality_scores = None
    if config.use_biprojection or config.use_quality_selection:
        logger.info("Computing SNR-based layer quality scores for biprojection...")
        quality_scores = compute_direction_quality_scores(
            harmful_activations,
            harmless_activations,
            hybrid_info=hybrid_info,
        )
        # Log top quality layers
        if quality_scores:
            top_layers = sorted(quality_scores.items(), key=lambda x: x[1]["quality"], reverse=True)[:5]
            for layer_idx, scores in top_layers:
                logger.info(f"  Layer {layer_idx}: quality={scores['quality']:.3f}, SNR={scores['snr']:.3f}")

    # Biprojection: Store harmless directions for boundary clamping
    harmless_directions = None
    if config.use_harmless_boundary:
        logger.info("Storing harmless directions for boundary clamping...")
        harmless_directions = {}
        for layer_idx in extraction_layers:
            if layer_idx in harmless_activations:
                harmless_mean = harmless_activations[layer_idx].mean(dim=0)
                if config.normalize_directions:
                    harmless_mean = F.normalize(harmless_mean, dim=0)
                harmless_directions[layer_idx] = harmless_mean.to(config.dtype)

    # Biprojection: Compute combined direction from measurement layers
    biprojected_direction = None
    if config.use_biprojection and quality_scores:
        measurement_layers, _ = select_biprojection_layers(
            quality_scores,
            num_layers=num_layers,
            num_measurement_layers=config.num_measurement_layers,
            intervention_range=config.intervention_range,
        )

        # Average directions from measurement layers
        measurement_directions = [
            directions[layer_idx] for layer_idx in measurement_layers
            if layer_idx in directions
        ]
        if measurement_directions:
            biprojected_direction = torch.stack(measurement_directions).mean(dim=0)
            if config.normalize_directions:
                biprojected_direction = F.normalize(biprojected_direction, dim=0)
            biprojected_direction = biprojected_direction.to(config.dtype)
            logger.info(f"Computed biprojected direction from layers {measurement_layers}")

    metadata = {
        "num_harmful_prompts": len(config.harmful_prompts),
        "num_harmless_prompts": len(config.harmless_prompts),
        "extraction_layers": extraction_layers,
        "normalized": config.normalize_directions,
        "token_position": config.token_position,
        # Outlier clipping
        "winsorized": config.use_winsorization,
        "winsorize_percentile": config.winsorize_percentile if config.use_winsorization else None,
        "magnitude_clipped": config.use_magnitude_clipping,
        "magnitude_clip_percentile": config.magnitude_clip_percentile if config.use_magnitude_clipping else None,
        # Numerical stability and projection
        "use_welford_mean": config.use_welford_mean,
        "use_float64_subtraction": config.use_float64_subtraction,
        "use_projected_refusal": config.use_projected_refusal,
        # Biprojection metadata
        "use_biprojection": config.use_biprojection,
        "use_harmless_boundary": config.use_harmless_boundary,
        "num_measurement_layers": config.num_measurement_layers if config.use_biprojection else None,
        "intervention_range": list(config.intervention_range) if config.use_biprojection else None,
    }

    logger.info(f"Computed refusal directions for {len(directions)} layers")

    return RefusalDirections(
        directions=directions,
        mean_direction=mean_direction,
        metadata=metadata,
        harmless_directions=harmless_directions,
        quality_scores=quality_scores,
        biprojected_direction=biprojected_direction,
    )


# Norm-Preserving Orthogonal Projection


def orthogonal_projection_matrix(direction: torch.Tensor) -> torch.Tensor:
    """
    Compute the orthogonal projection matrix that removes the component along `direction`.

    P = I - (d @ d^T) / (d^T @ d)

    For a normalized direction vector, this simplifies to:
    P = I - d @ d^T
    """
    direction = direction.view(-1, 1)  # Column vector
    # Ensure direction is normalized
    direction = direction / direction.norm()
    # Projection matrix: I - d @ d^T
    proj_matrix = torch.eye(direction.size(0), device=direction.device, dtype=direction.dtype)
    proj_matrix -= direction @ direction.T
    return proj_matrix


def apply_norm_preserving_projection(
    weight: torch.Tensor,
    direction: torch.Tensor,
    preserve_norm: bool = True,
    multiplier: float = 1.0,
) -> torch.Tensor:
    """
    Apply norm-preserving orthogonal projection to a weight matrix.

    This removes the component of each row (or column, depending on the weight's role)
    that aligns with the refusal direction, then optionally rescales to preserve
    the original Frobenius norm.

    Args:
        weight: The weight matrix to modify [out_features, in_features]
        direction: The refusal direction vector [hidden_size]
        preserve_norm: Whether to rescale weights to preserve Frobenius norm
        multiplier: Scale factor for ablation strength (1.0 = full ablation)

    Returns:
        Modified weight matrix
    """
    original_dtype = weight.dtype
    original_norm = weight.float().norm()

    # Work in float32 for numerical stability
    # Ensure both tensors are on the same device and dtype
    weight_float = weight.float()
    direction_float = direction.to(device=weight.device, dtype=torch.float32)

    # Ensure direction is normalized
    direction_float = F.normalize(direction_float, dim=0)

    # Determine which dimension to project based on weight shape
    # For most linear layers: weight is [out_features, in_features]
    # The direction typically matches in_features (the hidden dimension being projected)

    if direction_float.shape[0] == weight_float.shape[1]:
        # Project along input dimension (columns)
        # W_new = W @ P where P is the projection matrix
        # Equivalent to: W_new = W - (W @ d) @ d^T
        proj_coeffs = weight_float @ direction_float  # [out_features]
        proj_coeffs = proj_coeffs * multiplier
        adjustment = torch.outer(proj_coeffs, direction_float)  # [out_features, in_features]
        weight_new = weight_float - adjustment

    elif direction_float.shape[0] == weight_float.shape[0]:
        # Project along output dimension (rows)
        # W_new = P @ W
        # Equivalent to: W_new = W - d @ (d^T @ W)
        proj_coeffs = direction_float @ weight_float  # [in_features]
        proj_coeffs = proj_coeffs * multiplier
        adjustment = torch.outer(direction_float, proj_coeffs)  # [out_features, in_features]
        weight_new = weight_float - adjustment

    else:
        logger.warning(
            f"Direction shape {direction_float.shape} doesn't match weight shape {weight_float.shape}, skipping"
        )
        return weight

    # Norm preservation: rescale to maintain original Frobenius norm
    if preserve_norm:
        new_norm = weight_new.norm()
        if new_norm > 1e-8:  # Avoid division by zero
            weight_new = weight_new * (original_norm / new_norm)

    return weight_new.to(original_dtype)


def apply_per_neuron_norm_preserving_projection(
    weight: torch.Tensor,
    refusal_dir: torch.Tensor,
    scale_factor: float = 1.0,
    harmless_dir: Optional[torch.Tensor] = None,
    clamp_ratio: float = 0.1,
    null_space_V: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Decompose W into magnitude M (per-row norms) and direction W_hat,
    ablate only the direction component, then recombine with original magnitudes.
    This preserves per-neuron activation scales better than Frobenius norm preservation.

    Mathematical formulation:
        W = M @ W_hat  where M = diag(||W[i,:]||) and W_hat[i,:] = W[i,:] / ||W[i,:]||
        refusal_normalized = refusal_dir / ||refusal_dir||
        projection = W_hat @ refusal_normalized  (per-neuron alignment)
        W_hat_new = W_hat - scale_factor * outer(refusal_normalized, projection)
        W_hat_new = normalize(W_hat_new, dim=1)  (re-normalize each row)
        W_new = M @ W_hat_new  (recombine with original magnitudes)

    Args:
        weight: Weight matrix [out_features, in_features]
        refusal_dir: Refusal direction vector [in_features] (for o_proj/down_proj)
        scale_factor: Ablation strength (1.0 = full removal)
        harmless_dir: Optional harmless direction for boundary clamping
        clamp_ratio: How much to clamp toward harmless direction (0.1 = 10%)
        null_space_V: Optional null-space basis for combined mode (applies constraint before re-norm)

    Returns:
        Modified weight matrix with same per-neuron norms as original
    """
    original_dtype = weight.dtype
    weight_float = weight.float()

    # Ensure refusal_dir is on the same device and dtype as the weight
    refusal_dir = refusal_dir.to(device=weight.device, dtype=torch.float32)

    # Determine direction alignment (input vs output dimension)
    if refusal_dir.shape[0] == weight_float.shape[1]:
        # Direction aligns with input dimension (typical for o_proj, down_proj)
        project_input = True
    elif refusal_dir.shape[0] == weight_float.shape[0]:
        # Direction aligns with output dimension
        project_input = False
    else:
        logger.warning(
            f"Direction shape {refusal_dir.shape} doesn't match weight shape {weight_float.shape}, skipping"
        )
        return weight

    # Normalize refusal direction
    refusal_normalized = F.normalize(refusal_dir.float(), dim=0)

    if project_input:
        # Decompose weight: W = M @ W_hat where M is diagonal per-row norms
        W_norm = torch.norm(weight_float, dim=1, keepdim=True)  # [out_features, 1]

        # Handle zero-norm rows (shouldn't happen but be safe)
        W_norm = torch.clamp(W_norm, min=1e-8)

        W_direction = weight_float / W_norm  # Normalized direction [out_features, in_features]

        # Compute per-neuron projection onto refusal direction
        # projection[i] = W_direction[i,:] . refusal_normalized
        projection = W_direction @ refusal_normalized  # [out_features]

        # Ablate: remove component along refusal direction
        # W_direction_new[i,:] = W_direction[i,:] - scale_factor * projection[i] * refusal_normalized
        W_direction_new = W_direction - scale_factor * torch.outer(projection, refusal_normalized)

        # Optional: apply null-space constraint before re-normalization
        if null_space_V is not None:
            # Project the adjustment into null space of preservation activations
            # This ensures we don't affect outputs for preservation prompts
            adjustment = W_direction - W_direction_new  # What we're removing
            # Project adjustment into null space: adjustment_constrained = adjustment @ (I - V @ V^T)
            adjustment_constrained = adjustment - adjustment @ null_space_V @ null_space_V.T
            W_direction_new = W_direction - adjustment_constrained

        # Optional: clamp toward harmless direction to prevent over-ablation
        if harmless_dir is not None:
            harmless_normalized = F.normalize(harmless_dir.to(device=weight.device, dtype=torch.float32), dim=0)
            # Compute how far we've moved from harmless direction
            harmless_proj = W_direction @ harmless_normalized  # [out_features]
            harmless_proj_new = W_direction_new @ harmless_normalized

            # If we've moved away from harmless too much, clamp back
            moved_away = harmless_proj_new < harmless_proj * (1 - clamp_ratio)
            if moved_away.any():
                # Interpolate back toward preserving harmless direction
                correction_scale = (harmless_proj - harmless_proj_new) * moved_away.float()
                correction = torch.outer(correction_scale, harmless_normalized)
                W_direction_new = W_direction_new + clamp_ratio * correction

        # Re-normalize direction (each row should be unit vector)
        W_direction_new = F.normalize(W_direction_new, dim=1)

        # Recombine with original magnitudes
        W_new = W_norm * W_direction_new

    else:
        # Project along output dimension (rows) - less common case
        # Decompose by columns instead
        W_norm = torch.norm(weight_float, dim=0, keepdim=True)  # [1, in_features]
        W_norm = torch.clamp(W_norm, min=1e-8)
        W_direction = weight_float / W_norm

        projection = refusal_normalized @ W_direction  # [in_features]
        W_direction_new = W_direction - scale_factor * torch.outer(refusal_normalized, projection)

        if null_space_V is not None:
            adjustment = W_direction - W_direction_new
            adjustment_constrained = adjustment - null_space_V @ null_space_V.T @ adjustment
            W_direction_new = W_direction - adjustment_constrained

        if harmless_dir is not None:
            harmless_normalized = F.normalize(harmless_dir.to(device=weight.device, dtype=torch.float32), dim=0)
            harmless_proj = harmless_normalized @ W_direction
            harmless_proj_new = harmless_normalized @ W_direction_new
            moved_away = harmless_proj_new < harmless_proj * (1 - clamp_ratio)
            if moved_away.any():
                correction_scale = (harmless_proj - harmless_proj_new) * moved_away.float()
                correction = torch.outer(harmless_normalized, correction_scale)
                W_direction_new = W_direction_new + clamp_ratio * correction

        W_direction_new = F.normalize(W_direction_new, dim=0)
        W_new = W_norm * W_direction_new

    return W_new.to(original_dtype)


# Model Abliteration


def _get_language_model_root(model: AutoModelForCausalLM) -> tuple[torch.nn.Module, str]:
    """
    Find the language model root module for VL and standard architectures.

    Returns (root_module, prefix) where prefix is the dotted path from the
    top-level model to root_module. For standard text models, prefix may be ""
    and root_module is the model itself.
    """
    # VL models: model.model.model.layers (Qwen2-VL, Qwen3-VL, LLaVA)
    # Prefix is the dotted path as seen by model.named_modules() — NOT the
    # Python attribute chain. E.g., model.model is named "model" in the iterator.
    if hasattr(model, "model"):
        if hasattr(model.model, "model") and hasattr(model.model.model, "layers"):
            return model.model.model, "model.model"
        # VL MoE / Mistral3 / Qwen3.5: model.model.language_model.layers
        if hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
            return model.model.language_model, "model.language_model"
        # Standard text models: model.model.layers
        if hasattr(model.model, "layers"):
            return model.model, "model"
        if hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
            return model.model.decoder, "model.decoder"

    # VL with language_model at top (InternVL, GLM4v)
    if hasattr(model, "language_model"):
        if hasattr(model.language_model, "model") and hasattr(model.language_model.model, "layers"):
            return model.language_model.model, "language_model.model"
        if hasattr(model.language_model, "transformer") and hasattr(model.language_model.transformer, "layers"):
            return model.language_model.transformer, "language_model.transformer"
        if hasattr(model.language_model, "layers"):
            return model.language_model, "language_model"

    # GPT-2 / GPT-NeoX / backbone / encoder / direct
    if hasattr(model, "transformer"):
        return model.transformer, "transformer"
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox, "gpt_neox"
    if hasattr(model, "backbone"):
        return model.backbone, "backbone"
    if hasattr(model, "encoder"):
        return model.encoder, "encoder"

    # Fallback: entire model
    return model, ""


def get_linear_layer_names(model: AutoModelForCausalLM) -> list[str]:
    """
    Get names of all linear layers in the language model.

    For VL models, only returns layers within the language model submodule
    (excluding vision encoder, projector, etc.) plus lm_head if present.
    Names are full dotted paths from the top-level model so that getattr
    traversal works.
    """
    root, prefix = _get_language_model_root(model)

    linear_names = []
    for name, module in root.named_modules():
        if isinstance(module, torch.nn.Linear):
            full_name = f"{prefix}.{name}" if prefix else name
            linear_names.append(full_name)

    # Also include lm_head if it exists at the top level and wasn't already found
    # (lm_head is often a sibling of the language model root, not inside it)
    lm_head_candidates = ["lm_head", "model.lm_head", "model.model.lm_head"]
    found_names = set(linear_names)
    for lm_name in lm_head_candidates:
        if lm_name not in found_names:
            parts = lm_name.split(".")
            try:
                m = model
                for part in parts:
                    m = getattr(m, part)
                if isinstance(m, torch.nn.Linear):
                    linear_names.append(lm_name)
            except AttributeError:
                pass

    return linear_names


def get_layer_index_from_name(name: str) -> Optional[int]:
    """Extract layer index from a parameter name."""
    import re

    # Common patterns: layers.0., h.0., decoder.layers.0., etc.
    patterns = [
        r"layers\.(\d+)\.",
        r"h\.(\d+)\.",
        r"block\.(\d+)\.",
        r"decoder\.layers\.(\d+)\.",
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))
    return None


def get_layer_type_from_name(name: str) -> Optional[str]:
    """
    Extract the layer type (sublayer name) from a full parameter path.

    For transformer layers, identifies specific sublayers:
        - q_proj, k_proj, v_proj: attention query/key/value projections
        - o_proj: attention output projection
        - gate_proj, up_proj: MLP input projections
        - down_proj: MLP output projection
        - lm_head: final output projection

    Args:
        name: Full parameter name like 'model.layers.10.self_attn.o_proj.weight'

    Returns:
        Layer type string or None if not recognized
    """
    # Common layer type patterns (order matters - check more specific first)
    layer_types = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",      # MLP
        "qkv_proj", "out_proj",                   # Alternative naming
        "in_proj_a", "in_proj_b",                 # Hybrid linear attention (Qwen3.5 SSM)
        "in_proj_qkv", "in_proj_z",              # Hybrid linear attention (Qwen3.5 SSM)
        "fc1", "fc2",                             # GPT-style MLP
        "c_attn", "c_proj",                       # GPT-2 naming
        "w1", "w2", "w3",                         # MoE expert layers (Mixtral/GptOss)
        "router",                                  # MoE router
        "lm_head",                                # Output head
    ]

    name_lower = name.lower()
    for layer_type in layer_types:
        if layer_type in name_lower:
            return layer_type

    return None


def filter_layers_by_type(
    linear_names: list[str],
    target_types: Optional[list[str]] = None,
) -> list[str]:
    """
    Filter linear layer names to only include specified types.

    Args:
        linear_names: List of all linear layer names
        target_types: Layer types to include (default: ['o_proj', 'down_proj'])

    Returns:
        Filtered list of layer names matching target types
    """
    if target_types is None:
        target_types = ["o_proj", "down_proj"]

    # Normalize to lowercase for comparison
    target_types_lower = [t.lower() for t in target_types]

    filtered = []
    for name in linear_names:
        layer_type = get_layer_type_from_name(name)
        if layer_type is not None and layer_type.lower() in target_types_lower:
            filtered.append(name)

    logger.info(f"Filtered {len(linear_names)} layers to {len(filtered)} targeting {target_types}")

    return filtered


def abliterate_model(
    model: AutoModelForCausalLM,
    directions: RefusalDirections,
    config: AbliterationConfig,
    null_space_projector: Optional["NullSpaceProjector"] = None,
) -> AutoModelForCausalLM:
    """
    Apply norm-preserving orthogonal projection abliteration to the model.

    This modifies the model's weights in-place to remove refusal behavior.

    Args:
        model: The model to abliterate
        directions: Computed refusal directions
        config: Abliteration configuration
        null_space_projector: Optional null-space projector for capability preservation

    Returns:
        Modified model
    """
    # Get hybrid architecture info
    hybrid_info = getattr(config, "_hybrid_info", None)

    # Log mode
    mode_parts = []
    if config.use_biprojection:
        mode_parts.append("biprojection")
    if config.use_per_neuron_norm:
        mode_parts.append("per-neuron norm")
    if null_space_projector is not None:
        mode_parts.append("null-space")
    if config.use_harmless_boundary:
        mode_parts.append("harmless boundary")
    if hybrid_info and hybrid_info.is_hybrid and config.hybrid_strategy == "auto":
        mode_parts.append("hybrid-aware")

    if mode_parts:
        logger.info(f"Applying abliteration with: {', '.join(mode_parts)}")
    else:
        logger.info("Applying standard Frobenius norm-preserving abliteration...")

    linear_names = get_linear_layer_names(model)
    logger.info(f"Found {len(linear_names)} linear layers")

    # Filter by layer type if configured
    if config.target_layer_types:
        linear_names = filter_layers_by_type(linear_names, config.target_layer_types)

    # Determine intervention layers for biprojection
    intervention_layers = None
    if config.use_biprojection:
        if config.intervention_layers:
            intervention_layers = set(config.intervention_layers)
        elif directions.quality_scores:
            # Estimate num_layers
            layer_indices = [get_layer_index_from_name(n) for n in linear_names]
            max_layer = max((i for i in layer_indices if i is not None), default=0)
            num_layers = max_layer + 1

            _, auto_intervention = select_biprojection_layers(
                directions.quality_scores,
                num_layers=num_layers,
                num_measurement_layers=config.num_measurement_layers,
                intervention_range=config.intervention_range,
            )
            intervention_layers = set(auto_intervention)

    # Select primary direction
    if config.use_biprojection and directions.biprojected_direction is not None:
        primary_direction = directions.biprojected_direction.to(config.device)
        logger.info("Using biprojected direction from measurement layers")
    elif config.use_mean_direction and directions.mean_direction is not None:
        primary_direction = directions.mean_direction.to(config.device)
        logger.info("Using mean direction across layers")
    else:
        primary_direction = None
        logger.info("Using per-layer directions")

    # Compute layer weights based on targeting mode
    layer_weights = None
    use_target_map = config.per_layer_multipliers is not None

    if use_target_map:
        # Use target map multipliers directly
        layer_weights = config.per_layer_multipliers
        logger.info(f"Using layer target map with {len(layer_weights)} layer multipliers")
        if config.exclude_layers:
            logger.info(f"  - Excluding {len(config.exclude_layers)} layers: {sorted(config.exclude_layers)[:10]}...")
    elif config.use_adaptive_weighting:
        # Fall back to Gaussian-based adaptive weighting
        layer_indices = [get_layer_index_from_name(n) for n in linear_names]
        max_layer = max((i for i in layer_indices if i is not None), default=0)
        num_layers = max_layer + 1

        layer_weights = compute_adaptive_layer_weights(
            num_layers,
            center=config.adaptive_position_center,
            sigma=config.adaptive_position_sigma,
        )
        logger.info(f"Using adaptive layer weighting (center={config.adaptive_position_center:.2f})")

    modified_count = 0
    skipped_count = 0

    for name in tqdm(linear_names, desc="Abliterating layers"):
        # Get the module
        parts = name.split(".")
        module = model
        for part in parts:
            module = getattr(module, part)

        weight = module.weight.data
        layer_idx = get_layer_index_from_name(name)

        # Check intervention layer range for biprojection
        if intervention_layers is not None and layer_idx is not None:
            if layer_idx not in intervention_layers:
                skipped_count += 1
                continue

        # Check target_layers (legacy config option)
        if config.target_layers is not None and layer_idx is not None:
            if layer_idx not in config.target_layers:
                skipped_count += 1
                continue

        # Check excluded layers from target map
        if config.exclude_layers and layer_idx is not None:
            if layer_idx in config.exclude_layers:
                skipped_count += 1
                continue

        # Hybrid architecture: skip recurrent dynamics projections
        if hybrid_info and hybrid_info.is_hybrid and config.hybrid_strategy == "auto":
            layer_type = get_layer_type_from_name(name)

            # Skip recurrent dynamics projections (tiny matrices that control gating)
            if config.hybrid_skip_recurrent_proj and layer_type in ("in_proj_a", "in_proj_b"):
                skipped_count += 1
                continue

            # Optionally skip state projections too (more conservative)
            if config.hybrid_skip_state_proj and layer_type in ("in_proj_qkv", "in_proj_z"):
                skipped_count += 1
                continue

        # Get direction (biprojected > per-layer > mean fallback)
        if primary_direction is not None:
            direction = primary_direction
        elif layer_idx is not None and layer_idx in directions.directions:
            direction = directions.directions[layer_idx].to(config.device)
        elif directions.mean_direction is not None:
            # Fallback to mean direction for layers outside extraction range
            direction = directions.mean_direction.to(config.device)
            logger.debug(f"Using mean direction fallback for {name} (layer {layer_idx} not in extraction range)")
        else:
            logger.warning(f"No direction available for {name}, skipping")
            skipped_count += 1
            continue

        # Check if direction dimension matches either weight dimension
        if direction.shape[0] not in weight.shape:
            skipped_count += 1
            continue

        # Get harmless direction for boundary clamping
        harmless_dir = None
        if config.use_harmless_boundary and directions.harmless_directions:
            harmless_dir = directions.harmless_directions.get(layer_idx)
            if harmless_dir is not None:
                harmless_dir = harmless_dir.to(config.device)

        # Compute effective multiplier (with layer targeting if enabled)
        effective_multiplier = config.direction_multiplier
        if layer_weights is not None and layer_idx is not None:
            if use_target_map:
                # For target map: use multiplier from map, handle unmapped layers
                if layer_idx in layer_weights:
                    layer_weight = layer_weights[layer_idx]
                elif config.unmapped_layer_behavior == "default":
                    layer_weight = config.unmapped_layer_multiplier
                else:
                    # "skip" behavior - use 0.0 multiplier (effectively skips)
                    layer_weight = 0.0
            else:
                # For adaptive weighting: use computed weight with 1.0 default
                layer_weight = layer_weights.get(layer_idx, 1.0)

            effective_multiplier = config.direction_multiplier * layer_weight

            # Skip if effective multiplier is zero
            if effective_multiplier == 0.0:
                skipped_count += 1
                continue

        # Hybrid architecture: apply differentiated layer weighting
        if hybrid_info and hybrid_info.is_hybrid and config.hybrid_strategy == "auto" and layer_idx is not None:
            if layer_idx in hybrid_info.full_attention_indices:
                effective_multiplier *= config.hybrid_full_attn_weight
            elif layer_idx in hybrid_info.linear_attention_indices:
                effective_multiplier *= config.hybrid_linear_attn_weight

        # Apply projection based on mode
        try:
            if config.use_per_neuron_norm:
                null_V = None
                if null_space_projector is not None:
                    null_V = null_space_projector.get_projector_for_layer(layer_idx)

                new_weight = apply_per_neuron_norm_preserving_projection(
                    weight,
                    direction,
                    scale_factor=effective_multiplier,
                    harmless_dir=harmless_dir,
                    clamp_ratio=config.harmless_clamp_ratio,
                    null_space_V=null_V,
                )
            elif null_space_projector is not None:
                # Null-space constrained (legacy/combined mode)
                from src.null_space import apply_null_space_constrained_projection

                null_V = null_space_projector.get_projector_for_layer(layer_idx)
                new_weight = apply_null_space_constrained_projection(
                    weight,
                    direction,
                    null_space_V=null_V,
                    preserve_norm=config.norm_preservation,
                    multiplier=effective_multiplier,
                )
            else:
                # Standard Frobenius norm preservation
                new_weight = apply_norm_preserving_projection(
                    weight,
                    direction,
                    preserve_norm=config.norm_preservation,
                    multiplier=effective_multiplier,
                )

            module.weight.data = new_weight
            modified_count += 1

        except Exception as e:
            logger.warning(f"Failed to abliterate {name}: {e}")
            skipped_count += 1

    logger.info(f"Modified {modified_count} layers, skipped {skipped_count} layers")

    return model


# Main Pipeline


def load_prompts_from_file(path: str, num_prompts: Optional[int] = None) -> list[str]:
    """Load prompts from a JSON or text file, optionally sampling randomly."""
    path = Path(path)

    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                prompts = data
            elif isinstance(data, dict) and "prompts" in data:
                prompts = data["prompts"]
            else:
                raise ValueError(f"Unexpected JSON structure in {path}")
    else:
        # Assume text file with one prompt per line
        with open(path, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]

    # Randomly sample if num_prompts is specified and less than total
    if num_prompts is not None and num_prompts < len(prompts):
        prompts = random.sample(prompts, num_prompts)

    return prompts


# Refusal Detection and Filtering


def is_refusal_response(response: str, refusal_keywords: list[str]) -> bool:
    """
    Check if a model response indicates a refusal.

    Args:
        response: The model's generated response
        refusal_keywords: List of phrases that indicate refusal

    Returns:
        True if the response appears to be a refusal
    """
    response_lower = response.lower()
    for keyword in refusal_keywords:
        if keyword.lower() in response_lower:
            return True
    return False


@torch.no_grad()
def test_prompts_for_refusal_batch(
    prompts: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: AbliterationConfig,
    show_progress: bool = False,
) -> list[tuple[str, bool, str]]:
    """
    Test multiple prompts for refusal in batches (much faster than one-by-one).

    Args:
        prompts: List of prompts to test
        model: The model to test
        tokenizer: The tokenizer
        config: Configuration with refusal keywords and generation settings
        show_progress: Whether to show a progress bar for batch processing

    Returns:
        List of (prompt, is_refused, response) tuples
    """
    model.eval()
    results = []
    batch_size = config.refusal_test_batch_size

    # Calculate total batches for progress bar
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    batch_iter = range(0, len(prompts), batch_size)

    if show_progress:
        batch_iter = tqdm(batch_iter, total=total_batches, desc="Testing batches", leave=False)

    # Process in batches
    for i in batch_iter:
        batch_prompts = prompts[i : i + batch_size]

        # Format with chat template if available
        if hasattr(tokenizer, "apply_chat_template"):
            formatted_prompts = []
            for prompt in batch_prompts:
                messages = [{"role": "user", "content": prompt}]
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                formatted_prompts.append(formatted)
        else:
            formatted_prompts = batch_prompts

        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(config.device)

        # Generate responses for entire batch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.refusal_test_max_tokens,
                do_sample=False,  # Greedy for consistency
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode each response in the batch
        for j, (prompt, output) in enumerate(zip(batch_prompts, outputs)):
            # Get the input length for this specific prompt
            input_len = (inputs["attention_mask"][j] == 1).sum().item()
            generated_tokens = output[input_len:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            is_refused = is_refusal_response(response, config.refusal_keywords)
            results.append((prompt, is_refused, response))

    return results


def filter_harmful_prompts_by_refusal(
    prompt_pool: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: AbliterationConfig,
    target_count: Optional[int] = None,
) -> tuple[list[str], list[str]]:
    """
    Filter harmful prompts to only include those that the model actually refuses.
    Uses log-likelihood based detection for fast, accurate refusal prediction
    without generating responses.

    Args:
        prompt_pool: Full pool of available harmful prompts to sample from
        model: The model to test against
        tokenizer: The tokenizer
        config: Configuration settings
        target_count: Target number of refused prompts to collect (None = test all)

    Returns:
        Tuple of (refused_prompts, non_refused_prompts)
    """
    if target_count is None:
        target_count = len(prompt_pool)

    logger.info(f"Finding {target_count} refused prompts from pool of {len(prompt_pool)}...")

    # Create log-likelihood based refusal detector
    detector_config = RefusalDetectorConfig(threshold=config.refusal_threshold)
    detector = LogLikelihoodRefusalDetector(model, tokenizer, detector_config)

    refused_prompts = []
    non_refused_prompts = []
    tested_count = 0

    # Shuffle prompts for random sampling
    shuffled_prompts = prompt_pool.copy()
    random.shuffle(shuffled_prompts)

    # Process in batches until we have enough refusals
    batch_size = config.refusal_test_batch_size

    # Progress bar tracks found refusals vs target
    with tqdm(total=target_count, desc="Finding refused prompts", unit="prompt") as pbar:
        for i in range(0, len(shuffled_prompts), batch_size):
            if len(refused_prompts) >= target_count:
                break

            batch_prompts = shuffled_prompts[i : i + batch_size]

            # Use log-likelihood detection (no generation needed)
            batch_results = detector.detect_refusal_with_scores(batch_prompts)

            for prompt, (is_refused, score) in zip(batch_prompts, batch_results):
                tested_count += 1
                if is_refused:
                    refused_prompts.append(prompt)
                    logger.debug(f"REFUSED (score={score:.2f}): {prompt[:50]}...")
                    # Update progress bar for each refusal found
                    pbar.update(1)
                    if len(refused_prompts) >= target_count:
                        break
                else:
                    non_refused_prompts.append(prompt)
                    logger.debug(f"NOT REFUSED (score={score:.2f}): {prompt[:50]}...")

    logger.info(f"Refusal filtering complete:")
    logger.info(f"  - Prompts tested: {tested_count}")
    logger.info(f"  - Prompts refused: {len(refused_prompts)}")
    logger.info(f"  - Prompts NOT refused: {len(non_refused_prompts)}")

    if len(refused_prompts) < target_count:
        logger.warning(
            f"Could only find {len(refused_prompts)} refused prompts "
            f"(target was {target_count}) after testing all {len(prompt_pool)} prompts"
        )

    return refused_prompts, non_refused_prompts


def run_abliteration(config: AbliterationConfig):
    """Run the complete abliteration pipeline."""
    # Detect hybrid architecture early
    hybrid_info = detect_hybrid_architecture(config.model_path)
    if hybrid_info.is_hybrid:
        logger.info(
            f"Hybrid architecture detected: {len(hybrid_info.full_attention_indices)} full attention, "
            f"{len(hybrid_info.linear_attention_indices)} linear attention layers "
            f"(interval: {hybrid_info.full_attention_interval})"
        )
    # Store hybrid_info in config for downstream access
    config._hybrid_info = hybrid_info

    # Dynamic layer targeting implies per-layer directions
    if config.dynamic_layer_targeting:
        config.use_mean_direction = False
        logger.info("Dynamic layer targeting enabled: using per-layer directions")

    logger.info("=" * 60)
    features = ["Norm-Preserving Abliteration"]
    if hybrid_info.is_hybrid and config.hybrid_strategy == "auto":
        features.append("Hybrid-Aware")
    if config.dynamic_layer_targeting:
        features.append("Dynamic Layer Targeting")
    if config.use_biprojection:
        features.append("Biprojection")
    if config.use_per_neuron_norm:
        features.append("Per-Neuron Norm")
    if config.use_null_space:
        features.append("Null-Space Constraints")
    if config.use_winsorization:
        features.append("Winsorization")
    if config.use_magnitude_clipping:
        features.append("Magnitude Clipping")
    if config.use_adaptive_weighting:
        features.append("Adaptive Weighting")
    if config.use_harmless_boundary:
        features.append("Harmless Boundary")
    if config.target_layer_types:
        features.append(f"Target: {','.join(config.target_layer_types)}")
    # Numerical stability features (enabled by default)
    stability = []
    if config.use_welford_mean:
        stability.append("Welford")
    if config.use_float64_subtraction:
        stability.append("float64")
    if stability:
        features.append(f"Stability: {'+'.join(stability)}")
    logger.info(f"Features: {', '.join(features)}")
    logger.info("=" * 60)

    # Load model and tokenizer first (needed for refusal filtering)
    logger.info(f"Loading model from {config.model_path}...")
    model, tokenizer = load_model_and_tokenizer(
        config.model_path,
        device=config.device,
        dtype=config.dtype,
        trust_remote_code=True,
    )

    # Load prompts from files
    # For harmful prompts with filtering: load ALL prompts as a pool, then filter to target count
    # For harmless prompts: sample directly since no filtering needed
    if config.filter_harmful_prompts and not config.load_directions_path:
        # Load full pool of harmful prompts for filtering
        harmful_prompt_pool = load_prompts_from_file(config.harmful_prompts_path, num_prompts=None)
        logger.info(f"Loaded {len(harmful_prompt_pool)} harmful prompts from {config.harmful_prompts_path}")

        # Filter to find target_count refused prompts
        target_count = config.num_prompts if config.num_prompts else len(harmful_prompt_pool)
        refused_prompts, _ = filter_harmful_prompts_by_refusal(
            harmful_prompt_pool, model, tokenizer, config, target_count=target_count
        )

        if len(refused_prompts) == 0:
            raise ValueError(
                "No harmful prompts were refused by the model! "
                "Cannot compute refusal directions without refused prompts. "
                "Either use different harmful prompts or disable filtering with --no_filter_prompts"
            )

        config.harmful_prompts = refused_prompts
        logger.info(f"Using {len(config.harmful_prompts)} refused prompts for refusal direction computation")
    else:
        # No filtering - just sample directly
        config.harmful_prompts = load_prompts_from_file(config.harmful_prompts_path, config.num_prompts)
        logger.info(f"Loaded {len(config.harmful_prompts)} harmful prompts from {config.harmful_prompts_path}")

    config.harmless_prompts = load_prompts_from_file(config.harmless_prompts_path, config.num_prompts)
    logger.info(f"Loaded {len(config.harmless_prompts)} harmless prompts from {config.harmless_prompts_path}")

    # Get or compute refusal directions
    if config.load_directions_path:
        logger.info(f"Loading pre-computed directions from {config.load_directions_path}")
        directions = RefusalDirections.load(config.load_directions_path)
    else:
        directions = compute_refusal_directions(model, tokenizer, config)

        if config.save_directions:
            directions_path = Path(config.output_path) / "refusal_directions.pt"
            directions_path.parent.mkdir(parents=True, exist_ok=True)
            directions.save(str(directions_path))

    # Compute null-space projectors if enabled (for capability preservation)
    null_space_projector = None
    if config.use_null_space:
        from src.null_space import (
            NullSpaceConfig,
            NullSpaceProjector,
            compute_null_space_projectors,
            get_default_preservation_prompts_path,
        )

        preservation_path = config.preservation_prompts_path
        if preservation_path is None:
            preservation_path = get_default_preservation_prompts_path()

        # Determine layer indices for null-space computation
        layers = directions.directions.keys()
        if not layers:
            # Use extraction layers from config
            layers = config.extraction_layer_indices or []

        null_config = NullSpaceConfig(
            preservation_prompts_path=preservation_path,
            svd_rank_ratio=config.null_space_rank_ratio,
            regularization=config.null_space_regularization,
        )

        logger.info("Computing null-space projectors for capability preservation...")
        # Clear GPU cache to ensure clean state after refusal direction extraction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        null_space_projector = compute_null_space_projectors(
            model, tokenizer, null_config, list(layers), config.device, config.dtype
        )

        # Save projectors for reuse
        if config.save_directions:
            projector_path = Path(config.output_path) / "null_space_projectors.pt"
            projector_path.parent.mkdir(parents=True, exist_ok=True)
            null_space_projector.save(str(projector_path))
            logger.info(f"Saved null-space projectors to {projector_path}")

    # KL monitoring setup (must cache BEFORE abliteration modifies weights in-place)
    kl_monitor = None
    kl_reference_prompts = None
    auto_tune_result = None
    kl_result = None

    if config.use_kl_monitoring or config.use_kl_auto_tune:
        from src.kl_monitor import (
            KLDivergenceMonitor,
            KLMonitorConfig,
            auto_tune_multiplier,
            load_reference_prompts,
            save_kl_report,
        )

        kl_reference_prompts = load_reference_prompts(
            path=config.kl_reference_prompts_path,
            num_prompts=config.kl_num_reference_prompts,
        )
        logger.info(f"Loaded {len(kl_reference_prompts)} reference prompts for KL monitoring")

        kl_mon_config = KLMonitorConfig(
            num_reference_prompts=config.kl_num_reference_prompts,
            top_k=config.kl_top_k,
            batch_size=config.kl_batch_size,
            search_min=config.kl_search_min,
            search_max=config.kl_search_max,
            search_tolerance=config.kl_search_tolerance,
            max_search_iterations=config.kl_max_search_iterations,
            kl_threshold=config.kl_threshold,
        )

        kl_monitor = KLDivergenceMonitor(model, tokenizer, kl_mon_config, config.device)
        kl_monitor.cache_reference_logits(kl_reference_prompts)

    # Apply abliteration (or auto-tune)
    if config.use_kl_auto_tune and kl_monitor is not None:
        logger.info("Starting KL auto-tune binary search...")
        auto_tune_result = auto_tune_multiplier(
            model=model,
            tokenizer=tokenizer,
            directions=directions,
            config=config,
            kl_monitor=kl_monitor,
            kl_config=kl_mon_config,
            reference_prompts=kl_reference_prompts,
            null_space_projector=null_space_projector,
        )
        logger.info(f"Auto-tune complete: best_multiplier={auto_tune_result.best_multiplier:.4f}, "
                     f"KL={auto_tune_result.best_kl:.4f}, converged={auto_tune_result.converged}")
    else:
        model = abliterate_model(model, directions, config, null_space_projector)

    # KL monitoring (post-abliteration measurement for monitor-only mode)
    if config.use_kl_monitoring and kl_monitor is not None and auto_tune_result is None:
        logger.info("Computing KL divergence on reference prompts...")
        kl_result = kl_monitor.compute_kl_divergence(kl_reference_prompts, config.direction_multiplier)
        logger.info(f"KL divergence: mean={kl_result.mean_kl:.4f}, median={kl_result.median_kl:.4f}, "
                     f"max={kl_result.max_kl:.4f}, std={kl_result.std_kl:.4f}")

    # Save the modified model (with version suffix if path already exists)
    output_path = get_versioned_path(config.output_path)
    if output_path != Path(config.output_path):
        logger.info(f"Output path exists, using versioned path: {output_path}")
    logger.info(f"Saving abliterated model to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model and tokenizer (handles FP8 dequantized models specially)
    save_model_safe(model, tokenizer, output_path)

    # Preserve original model config fields that transformers might not save
    source_path = Path(config.model_path)
    preserve_model_config(source_path, output_path)

    # Copy essential model files (tokenizer vocab, generation config, etc.)
    essential_copied = copy_essential_model_files(source_path, output_path)
    if essential_copied:
        logger.info(f"Copied {len(essential_copied)} essential files: {', '.join(essential_copied)}")

    # Copy vision files for VL models (needed for GGUF mmproj export)
    if is_vision_model(source_path):
        logger.info("Detected Vision-Language model, copying vision encoder files...")
        copied_files = copy_vision_files(source_path, output_path)
        if copied_files:
            logger.info(f"Copied {len(copied_files)} vision-related files: {', '.join(copied_files)}")
        else:
            logger.warning("No vision files found to copy")

    # Save config for reproducibility
    # Convert dtype to string for JSON serialization
    dtype_str = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }.get(config.dtype, str(config.dtype))

    config_save = {
        # Core settings
        "model_path": config.model_path,
        "output_path": str(output_path),
        "target_layers": config.target_layers,
        "extraction_layer_indices": config.extraction_layer_indices,
        "use_mean_direction": config.use_mean_direction,
        "normalize_directions": config.normalize_directions,
        "norm_preservation": config.norm_preservation,
        "direction_multiplier": config.direction_multiplier,
        "token_position": config.token_position,
        "dtype": dtype_str,
        "batch_size": config.batch_size,
        # Prompt info
        "num_harmful_prompts": len(config.harmful_prompts),
        "num_harmless_prompts": len(config.harmless_prompts),
        "filter_harmful_prompts": config.filter_harmful_prompts,
        "refusal_test_batch_size": config.refusal_test_batch_size,
        "refusal_threshold": config.refusal_threshold,
        # Winsorization options
        "use_winsorization": config.use_winsorization,
        "winsorize_percentile": config.winsorize_percentile if config.use_winsorization else None,
        # Magnitude clipping options
        "use_magnitude_clipping": config.use_magnitude_clipping,
        "magnitude_clip_percentile": config.magnitude_clip_percentile if config.use_magnitude_clipping else None,
        # Numerical stability options (from llm-abliteration)
        "use_welford_mean": config.use_welford_mean,
        "use_float64_subtraction": config.use_float64_subtraction,
        "use_projected_refusal": config.use_projected_refusal,
        # Null-space options
        "use_null_space": config.use_null_space,
        "null_space_rank_ratio": config.null_space_rank_ratio if config.use_null_space else None,
        "null_space_regularization": config.null_space_regularization if config.use_null_space else None,
        "preservation_prompts_path": config.preservation_prompts_path if config.use_null_space else None,
        # Adaptive weighting options
        "use_adaptive_weighting": config.use_adaptive_weighting,
        "adaptive_position_center": config.adaptive_position_center if config.use_adaptive_weighting else None,
        "adaptive_position_sigma": config.adaptive_position_sigma if config.use_adaptive_weighting else None,
        # Biprojection options
        "use_biprojection": config.use_biprojection,
        "use_per_neuron_norm": config.use_per_neuron_norm,
        "target_layer_types": config.target_layer_types,
        "num_measurement_layers": config.num_measurement_layers if config.use_biprojection else None,
        "measurement_layers": config.measurement_layers,
        "intervention_layers": config.intervention_layers,
        "intervention_range": list(config.intervention_range) if config.use_biprojection else None,
        # Harmless boundary clamping
        "use_harmless_boundary": config.use_harmless_boundary,
        "harmless_clamp_ratio": config.harmless_clamp_ratio if config.use_harmless_boundary else None,
        # Quality-based layer selection
        "use_quality_selection": config.use_quality_selection,
        "min_quality_threshold": config.min_quality_threshold if config.use_quality_selection else None,
        # Layer target map
        "layer_targeting_mode": config.layer_targeting_mode,
        "layer_target_map_path": config.layer_target_map_path,
        "exclude_layers": config.exclude_layers,
        "unmapped_layer_behavior": config.unmapped_layer_behavior if config.layer_target_map_path else None,
        "unmapped_layer_multiplier": config.unmapped_layer_multiplier if config.unmapped_layer_behavior == "default" else None,
        "num_layers_with_multipliers": len(config.per_layer_multipliers) if config.per_layer_multipliers else None,
        # Dynamic layer targeting
        "dynamic_layer_targeting": config.dynamic_layer_targeting,
        # KL monitoring
        "use_kl_monitoring": config.use_kl_monitoring,
        "use_kl_auto_tune": config.use_kl_auto_tune,
        "kl_threshold": config.kl_threshold if config.use_kl_auto_tune else None,
        "kl_top_k": config.kl_top_k if (config.use_kl_monitoring or config.use_kl_auto_tune) else None,
        "kl_num_reference_prompts": config.kl_num_reference_prompts if (config.use_kl_monitoring or config.use_kl_auto_tune) else None,
    }

    # Add auto-tune result to config if available
    if auto_tune_result is not None:
        config_save["kl_auto_tune_result"] = {
            "best_multiplier": auto_tune_result.best_multiplier,
            "best_kl": auto_tune_result.best_kl,
            "converged": auto_tune_result.converged,
            "num_iterations": auto_tune_result.num_iterations,
        }

    with open(output_path / "abliteration_config.json", "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(config_save), f, indent=2)

    # Save KL divergence report if monitoring was active
    if kl_result is not None or auto_tune_result is not None:
        save_kl_report(output_path, kl_result=kl_result, auto_tune_result=auto_tune_result)

    logger.info("=" * 60)
    logger.info("Abliteration complete!")
    logger.info(f"Output saved to: {output_path}")
    logger.info("=" * 60)

    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(
        description="Norm-Preserving Orthogonal Projection Abliteration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (uses ./prompts/harmful.txt and ./prompts/harmless.txt)
  python abliterate.py --model_path ./my_model --output_path ./abliterated_model

  # Sample 100 random prompts from each file
  python abliterate.py --model_path ./my_model --output_path ./abliterated_model --num_prompts 100

  # With custom prompt files
  python abliterate.py \\
    --model_path ./my_model \\
    --output_path ./abliterated_model \\
    --harmful_prompts /path/to/harmful.json \\
    --harmless_prompts /path/to/harmless.json

  # Target specific layers only
  python abliterate.py \\
    --model_path ./my_model \\
    --output_path ./abliterated_model \\
    --target_layers 10 11 12 13 14 15
        """,
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the input model (HuggingFace format)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the abliterated model",
    )
    parser.add_argument(
        "--harmful_prompts",
        type=str,
        default=get_default_prompts_path("harmful.txt"),
        help="Path to JSON or text file with harmful prompts (default: <package>/prompts/harmful.txt)",
    )
    parser.add_argument(
        "--harmless_prompts",
        type=str,
        default=get_default_prompts_path("harmless.txt"),
        help="Path to JSON or text file with harmless prompts (default: <package>/prompts/harmless.txt)",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=30,
        help="Number of prompts to randomly sample from each file (default: use all)",
    )
    parser.add_argument(
        "--target_layers",
        type=int,
        nargs="+",
        default=None,
        help="Specific layer indices to abliterate (default: all layers)",
    )
    parser.add_argument(
        "--extraction_layers",
        type=int,
        nargs="+",
        default=None,
        help="Layer indices to extract directions from (default: middle layers)",
    )
    parser.add_argument(
        "--direction_multiplier",
        type=float,
        default=1.0,
        help="Scale factor for ablation strength (default: 1.0)",
    )
    parser.add_argument(
        "--no_norm_preservation",
        action="store_true",
        help="Disable norm preservation (not recommended)",
    )
    parser.add_argument(
        "--per_layer_directions",
        action="store_true",
        help="Use per-layer directions instead of mean direction",
    )
    parser.add_argument(
        "--load_directions",
        type=str,
        default=None,
        help="Load pre-computed refusal directions from file",
    )
    parser.add_argument(
        "--no_save_directions",
        action="store_true",
        help="Don't save computed refusal directions",
    )
    parser.add_argument(
        "--no_filter_prompts",
        action="store_true",
        help="Disable filtering of harmful prompts (use all prompts even if not refused)",
    )
    parser.add_argument(
        "--refusal_test_tokens",
        type=int,
        default=50,
        help="Max tokens to generate when testing for refusal (default: 50)",
    )
    parser.add_argument(
        "--token_position",
        type=str,
        default="last",
        help="Token position for activation extraction: 'last', 'mean', or integer",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for activation extraction",
    )
    parser.add_argument(
        "--refusal_test_batch_size",
        type=int,
        default=16,
        help="Batch size for refusal testing (larger = faster, but more VRAM). Default: 16",
    )
    parser.add_argument(
        "--refusal_threshold",
        type=float,
        default=-7.0,
        help="Log-likelihood threshold for refusal detection (higher = stricter). Default: -7.0",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "bfloat16", "float32"],
        help="Computation dtype",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    # Parse token position
    token_pos = args.token_position
    if token_pos.isdigit():
        token_pos = int(token_pos)

    config = AbliterationConfig(
        model_path=args.model_path,
        output_path=args.output_path,
        harmful_prompts_path=args.harmful_prompts,
        harmless_prompts_path=args.harmless_prompts,
        num_prompts=args.num_prompts,
        target_layers=args.target_layers,
        extraction_layer_indices=args.extraction_layers,
        use_mean_direction=not args.per_layer_directions,
        norm_preservation=not args.no_norm_preservation,
        direction_multiplier=args.direction_multiplier,
        device=args.device,
        dtype=dtype_map[args.dtype],
        save_directions=not args.no_save_directions,
        load_directions_path=args.load_directions,
        batch_size=args.batch_size,
        refusal_test_batch_size=args.refusal_test_batch_size,
        refusal_threshold=args.refusal_threshold,
        token_position=token_pos,
        filter_harmful_prompts=not args.no_filter_prompts,
        refusal_test_max_tokens=args.refusal_test_tokens,
    )

    model, tokenizer = run_abliteration(config)

    # Unload model from memory
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("Model unloaded from memory")


if __name__ == "__main__":
    main()

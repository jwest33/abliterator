#!/usr/bin/env python3
"""
Model loading utilities with Vision-Language (VL) model support.

This module provides centralized model loading that automatically detects
VL models (like Qwen3-VL) and uses the appropriate AutoModel class.
It also handles FP8 quantized models that require special loading.
"""

import json
import locale
import logging
import os
import re
import sys
from pathlib import Path
from typing import Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Fix Windows encoding issues - set preferred encoding for file I/O
if sys.platform == "win32":
    # Try to set UTF-8 mode for the process
    os.environ["PYTHONUTF8"] = "1"
    # Set locale to UTF-8 if possible
    try:
        locale.setlocale(locale.LC_ALL, '.UTF-8')
    except locale.Error:
        pass

logger = logging.getLogger(__name__)

# VL architecture patterns - maps model_type to known architecture class names
VL_ARCHITECTURES = {
    "qwen3_5": ["Qwen3_5ForConditionalGeneration"],
    "qwen3_vl": ["Qwen3VLForConditionalGeneration"],
    "qwen3_vl_moe": ["Qwen3VLMoeForConditionalGeneration"],
    "qwen2_vl": ["Qwen2VLForConditionalGeneration"],
    "qwen2_vl_moe": ["Qwen2VLMoeForConditionalGeneration"],
    "llava": ["LlavaForConditionalGeneration"],
    "llava_next": ["LlavaNextForConditionalGeneration"],
    "idefics2": ["Idefics2ForConditionalGeneration"],
    "paligemma": ["PaliGemmaForConditionalGeneration"],
    "glm4v": ["Glm4vForConditionalGeneration"],
    "mistral3": ["Mistral3ForConditionalGeneration"],
}

# Architectures that need special text-only loading even though they're multimodal
# These use ForConditionalGeneration but can be loaded with the text-only model class
# NOTE: Removed Gemma3 - forcing text-only class drops vision encoder params
FORCE_TEXT_MODEL_ARCHITECTURES = {
    # "Gemma3ForConditionalGeneration": "Gemma3ForCausalLM",  # Drops ~400M vision params
}

# Patterns that indicate FP8 quantization (scale factors)
FP8_SCALE_PATTERNS = [
    r"_scale_inv$",  # Common FP8 scale inverse suffix
    r"_scale$",      # FP8 scale suffix
    r"\.fp8_scale",  # Explicit FP8 scale
]


def detect_fp8_quantization(model_path: str) -> bool:
    """
    Detect if a model uses FP8 quantization by checking for scale factors.

    FP8 quantized models have weight tensors with names ending in '_scale_inv'
    or '_scale'. These models require special loading to avoid meta tensor issues.

    Args:
        model_path: Path to the model directory

    Returns:
        True if FP8 quantization is detected
    """
    model_dir = Path(model_path)

    # Check safetensors index first (most common for large models)
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            for key in weight_map.keys():
                for pattern in FP8_SCALE_PATTERNS:
                    if re.search(pattern, key):
                        logger.info(f"Detected FP8 quantization (found {key})")
                        return True
        except (json.JSONDecodeError, IOError) as e:
            logger.debug(f"Could not read safetensors index: {e}")

    # Check pytorch bin index as fallback
    bin_index_path = model_dir / "pytorch_model.bin.index.json"
    if bin_index_path.exists():
        try:
            with open(bin_index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            for key in weight_map.keys():
                for pattern in FP8_SCALE_PATTERNS:
                    if re.search(pattern, key):
                        logger.info(f"Detected FP8 quantization (found {key})")
                        return True
        except (json.JSONDecodeError, IOError) as e:
            logger.debug(f"Could not read pytorch bin index: {e}")

    # Check config.json for quantization_config
    config_path = model_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            quant_config = config.get("quantization_config", {})
            quant_method = quant_config.get("quant_method", "")
            if "fp8" in quant_method.lower():
                logger.info(f"Detected FP8 quantization from config (method: {quant_method})")
                return True
        except (json.JSONDecodeError, IOError) as e:
            logger.debug(f"Could not read config.json: {e}")

    return False


def _fix_weight_tying(model, model_path: str):
    """Fix weight tying for models loaded with FP8 dequantization.

    When loading FP8 models with dequantize=True, the weight tying between
    embed_tokens and lm_head may not be properly applied, resulting in
    lm_head.weight being randomly initialized.

    This function checks if tie_word_embeddings is enabled and manually
    copies the embed_tokens weights to lm_head if needed.

    Args:
        model: The loaded model
        model_path: Path to model directory (to read config)

    Returns:
        The model with fixed weight tying
    """
    # Check if weight tying should be enabled
    config_path = Path(model_path) / "config.json"
    tie_embeddings = False

    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            # Check main config and text_config for tie_word_embeddings
            tie_embeddings = config.get("tie_word_embeddings", False)
            if not tie_embeddings and "text_config" in config:
                tie_embeddings = config["text_config"].get("tie_word_embeddings", False)
        except (json.JSONDecodeError, IOError):
            pass

    if not tie_embeddings:
        return model

    # Find embed_tokens and lm_head
    embed_tokens = None
    lm_head = None

    # Try different paths for embed_tokens
    if hasattr(model, "model"):
        if hasattr(model.model, "embed_tokens"):
            embed_tokens = model.model.embed_tokens
        elif hasattr(model.model, "language_model") and hasattr(model.model.language_model, "embed_tokens"):
            embed_tokens = model.model.language_model.embed_tokens
        elif hasattr(model.model, "model") and hasattr(model.model.model, "embed_tokens"):
            embed_tokens = model.model.model.embed_tokens

    # Find lm_head
    if hasattr(model, "lm_head"):
        lm_head = model.lm_head

    if embed_tokens is None or lm_head is None:
        logger.debug("Could not find embed_tokens or lm_head for weight tying fix")
        return model

    # Check if weights are already tied (same data pointer)
    if embed_tokens.weight.data_ptr() == lm_head.weight.data_ptr():
        logger.debug("Weights already tied")
        return model

    # Check if lm_head weights look uninitialized (random) vs embed_tokens
    # If they're very different, copy embed_tokens to lm_head
    embed_norm = embed_tokens.weight.data.norm().item()
    lm_head_norm = lm_head.weight.data.norm().item()

    # If norms are very different, lm_head is likely uninitialized
    if abs(embed_norm - lm_head_norm) / max(embed_norm, lm_head_norm, 1e-8) > 0.1:
        logger.info("Fixing lm_head weight tying (copying from embed_tokens)")
        with torch.no_grad():
            lm_head.weight.data.copy_(embed_tokens.weight.data)

    return model


def detect_model_type(model_path: str) -> dict:
    """
    Detect model type by reading config.json.

    Args:
        model_path: Path to the model directory or HuggingFace model ID

    Returns:
        dict with:
            - is_vl: bool - True if this is a vision-language model
            - model_type: str - The model_type from config
            - architectures: list - The architectures list from config
    """
    config_path = Path(model_path) / "config.json"

    if not config_path.exists():
        # Might be a HuggingFace Hub model ID - we can't detect locally
        # Let transformers handle it and we'll try AutoModelForCausalLM first
        logger.debug(f"No config.json at {model_path}, assuming standard model")
        return {"is_vl": False, "model_type": None, "architectures": []}

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model_type = config.get("model_type", "")
    architectures = config.get("architectures", [])

    # Check if any architecture matches known VL patterns
    is_vl = False
    for arch in architectures:
        for vl_patterns in VL_ARCHITECTURES.values():
            if arch in vl_patterns:
                is_vl = True
                logger.info(f"Detected VL model: {arch}")
                break
        if is_vl:
            break

    # Also check model_type directly
    if not is_vl and model_type in VL_ARCHITECTURES:
        is_vl = True
        logger.info(f"Detected VL model type: {model_type}")

    # Check for FP8 quantization
    is_fp8 = detect_fp8_quantization(model_path)

    return {
        "is_vl": is_vl,
        "is_fp8": is_fp8,
        "model_type": model_type,
        "architectures": architectures,
    }


def load_model_and_tokenizer(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    trust_remote_code: bool = True,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load model and tokenizer, automatically handling VL models.

    For VL models, uses AutoModelForImageTextToText.
    For text models, uses AutoModelForCausalLM.

    Args:
        model_path: Path to model directory or HuggingFace model ID
        device: Device to load model on ("cuda", "cpu", or "auto")
        dtype: Torch dtype for model weights
        trust_remote_code: Whether to trust remote code in model files

    Returns:
        Tuple of (model, tokenizer)
    """
    model_info = detect_model_type(model_path)

    # Load tokenizer (same for both VL and text models for abliteration purposes)
    logger.info(f"Loading tokenizer from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
    except UnicodeDecodeError as e:
        # Windows encoding issue - try with use_fast=False as fallback
        logger.warning(f"Encoding error loading tokenizer, trying slow tokenizer: {e}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=trust_remote_code, use_fast=False
            )
        except UnicodeDecodeError:
            # Last resort: monkey-patch open to use UTF-8 for text mode only
            import builtins
            original_open = builtins.open
            def utf8_open(file, mode='r', *args, **kwargs):
                # Only patch text mode opens (no 'b' in mode)
                if 'b' not in mode and 'encoding' not in kwargs:
                    kwargs['encoding'] = 'utf-8'
                    kwargs['errors'] = 'replace'
                return original_open(file, mode, *args, **kwargs)
            builtins.open = utf8_open
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, trust_remote_code=trust_remote_code
                )
            except UnicodeDecodeError as final_error:
                builtins.open = original_open
                raise RuntimeError(
                    f"Failed to load tokenizer due to encoding issues: {final_error}\n"
                    "On Windows, try running with UTF-8 mode:\n"
                    "  set PYTHONUTF8=1 && abliterate\n"
                    "Or: python -X utf8 -m src.cli"
                ) from final_error
            finally:
                builtins.open = original_open
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check if we need to force a text-only model class
    # Some multimodal architectures (like Gemma3) have a separate text-only class
    force_text_class = None
    for arch in model_info.get("architectures", []):
        if arch in FORCE_TEXT_MODEL_ARCHITECTURES:
            force_text_class = FORCE_TEXT_MODEL_ARCHITECTURES[arch]
            logger.info(f"Detected {arch}, will use text-only class: {force_text_class}")
            break

    # Determine if we need special loading for FP8 models
    # FP8 models with scale factors cause meta tensor issues with device_map
    is_fp8 = model_info.get("is_fp8", False)
    if is_fp8:
        logger.warning(
            "FP8 quantized model detected. Loading without device_map to avoid "
            "meta tensor issues. This requires more CPU memory during loading."
        )

    # Check if this is a Mistral3 model (requires special handling for FP8)
    is_mistral3 = any("Mistral3" in arch for arch in model_info.get("architectures", []))

    # Load model with appropriate class
    if model_info["is_vl"]:
        # Mistral3 VL models require special handling
        if is_mistral3:
            logger.info(f"Loading Mistral3 VL model...")
            try:
                from transformers import Mistral3ForConditionalGeneration

                if is_fp8:
                    # Mistral3 FP8 models require FineGrainedFP8Config with dequantize=True
                    # We explicitly set torch_dtype to ensure consistent output dtype
                    logger.info("Using FineGrainedFP8Config for Mistral3 FP8 model...")
                    try:
                        from transformers import FineGrainedFP8Config
                        quant_config = FineGrainedFP8Config(dequantize=True)
                        model = Mistral3ForConditionalGeneration.from_pretrained(
                            model_path,
                            quantization_config=quant_config,
                            torch_dtype=dtype,  # Ensure consistent dtype after dequantization
                            device_map=device,
                            trust_remote_code=trust_remote_code,
                        )
                        # Fix weight tying for FP8 dequantized models
                        # The lm_head.weight may not be properly tied after dequantization
                        model = _fix_weight_tying(model, model_path)
                        logger.info(f"Model loaded with dtype={dtype} after FP8 dequantization")
                    except ImportError:
                        logger.warning(
                            "FineGrainedFP8Config not available. "
                            "Install mistral-common or upgrade transformers."
                        )
                        # Fallback: try loading without quantization config
                        model = Mistral3ForConditionalGeneration.from_pretrained(
                            model_path,
                            torch_dtype=dtype,
                            low_cpu_mem_usage=False,
                            trust_remote_code=trust_remote_code,
                        )
                        if device != "cpu":
                            model = model.to(device)
                else:
                    model = Mistral3ForConditionalGeneration.from_pretrained(
                        model_path,
                        torch_dtype=dtype,
                        device_map=device,
                        trust_remote_code=trust_remote_code,
                    )
            except ImportError:
                raise ImportError(
                    "Mistral3ForConditionalGeneration requires transformers >= 4.48.0. "
                    "Please upgrade with: pip install --upgrade transformers"
                )
        else:
            logger.info(f"Loading VL model with AutoModelForImageTextToText...")
            try:
                from transformers import AutoModelForImageTextToText

                if is_fp8:
                    # FP8 models: load without device_map, then move to device
                    model = AutoModelForImageTextToText.from_pretrained(
                        model_path,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=False,  # Avoid meta tensors
                        trust_remote_code=trust_remote_code,
                    )
                    if device != "cpu":
                        logger.info(f"Moving FP8 model to {device}...")
                        model = model.to(device)
                else:
                    model = AutoModelForImageTextToText.from_pretrained(
                        model_path,
                        torch_dtype=dtype,
                        device_map=device,
                        trust_remote_code=trust_remote_code,
                    )
            except ImportError:
                raise ImportError(
                    "AutoModelForImageTextToText requires transformers >= 4.57.0. "
                    "Please upgrade with: pip install --upgrade transformers"
                )
    elif force_text_class:
        # Load with specific text-only model class to avoid multimodal issues
        logger.info(f"Loading model with specific class: {force_text_class}...")
        try:
            # Dynamically import the model class from transformers
            import transformers
            model_class = getattr(transformers, force_text_class, None)
            if model_class is None:
                logger.warning(f"Could not find {force_text_class}, falling back to AutoModelForCausalLM")
                model_class = AutoModelForCausalLM

            if is_fp8:
                model = model_class.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=False,
                    trust_remote_code=trust_remote_code,
                )
                if device != "cpu":
                    logger.info(f"Moving FP8 model to {device}...")
                    model = model.to(device)
            else:
                model = model_class.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    device_map=device,
                    trust_remote_code=trust_remote_code,
                )
        except Exception as e:
            logger.warning(f"Failed to load with {force_text_class}: {e}, falling back to AutoModelForCausalLM")
            if is_fp8:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=False,
                    trust_remote_code=trust_remote_code,
                )
                if device != "cpu":
                    model = model.to(device)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    device_map=device,
                    trust_remote_code=trust_remote_code,
                )
    else:
        logger.info(f"Loading model with AutoModelForCausalLM...")
        if is_fp8:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                low_cpu_mem_usage=False,
                trust_remote_code=trust_remote_code,
            )
            if device != "cpu":
                logger.info(f"Moving FP8 model to {device}...")
                model = model.to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device,
                trust_remote_code=trust_remote_code,
            )

    logger.info(f"Model loaded: {type(model).__name__}")
    return model, tokenizer

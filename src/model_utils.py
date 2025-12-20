#!/usr/bin/env python3
"""
Model loading utilities with Vision-Language (VL) model support.

This module provides centralized model loading that automatically detects
VL models (like Qwen3-VL) and uses the appropriate AutoModel class.
"""

import json
import logging
from pathlib import Path
from typing import Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# VL architecture patterns - maps model_type to known architecture class names
VL_ARCHITECTURES = {
    "qwen3_vl": ["Qwen3VLForConditionalGeneration"],
    "qwen2_vl": ["Qwen2VLForConditionalGeneration"],
    "llava": ["LlavaForConditionalGeneration"],
    "llava_next": ["LlavaNextForConditionalGeneration"],
    "idefics2": ["Idefics2ForConditionalGeneration"],
    "paligemma": ["PaliGemmaForConditionalGeneration"],
}

# Architectures that need special text-only loading even though they're multimodal
# These use ForConditionalGeneration but can be loaded with the text-only model class
# NOTE: Removed Gemma3 - forcing text-only class drops vision encoder params
FORCE_TEXT_MODEL_ARCHITECTURES = {
    # "Gemma3ForConditionalGeneration": "Gemma3ForCausalLM",  # Drops ~400M vision params
}


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

    return {
        "is_vl": is_vl,
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
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )
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

    # Load model with appropriate class
    if model_info["is_vl"]:
        logger.info(f"Loading VL model with AutoModelForImageTextToText...")
        try:
            from transformers import AutoModelForImageTextToText

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
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    device_map=device,
                    trust_remote_code=trust_remote_code,
                )
            else:
                model = model_class.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    device_map=device,
                    trust_remote_code=trust_remote_code,
                )
        except Exception as e:
            logger.warning(f"Failed to load with {force_text_class}: {e}, falling back to AutoModelForCausalLM")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device,
                trust_remote_code=trust_remote_code,
            )
    else:
        logger.info(f"Loading model with AutoModelForCausalLM...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=trust_remote_code,
        )

    logger.info(f"Model loaded: {type(model).__name__}")
    return model, tokenizer

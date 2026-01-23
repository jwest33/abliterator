#!/usr/bin/env python3
"""
SAE Loader Utility

Handles loading of GemmaScope Sparse Autoencoders from HuggingFace Hub
with caching and dimension validation.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class SAEConfig:
    """Configuration for SAE loading."""

    repo_id: str = "google/gemma-scope-2-12b-it"
    sae_type: str = "resid_post_all"  # 12b uses resid_post_all, 4b uses resid_post
    width: str = "16k"
    l0: str = "small"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    cache_dir: Optional[str] = None
    local_files_only: bool = True  # Use cached files only, don't download


class JumpReLUSAE(nn.Module):
    """
    JumpReLU Sparse Autoencoder.

    Adapted from Google DeepMind's Gemma Scope implementation.
    Uses a threshold-based activation with ReLU.
    """

    def __init__(self, d_in: int, d_sae: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae

        # Encoder weights and bias
        self.w_enc = nn.Parameter(torch.zeros(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # Decoder weights and bias
        self.w_dec = nn.Parameter(torch.zeros(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # JumpReLU threshold
        self.threshold = nn.Parameter(torch.zeros(d_sae))

    def encode(self, input_acts: torch.Tensor) -> torch.Tensor:
        """Encode input activations to sparse latent space."""
        pre_acts = input_acts @ self.w_enc + self.b_enc
        mask = pre_acts > self.threshold
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def encode_pre_relu(self, input_acts: torch.Tensor) -> torch.Tensor:
        """Get pre-activation values before thresholding and ReLU."""
        return input_acts @ self.w_enc + self.b_enc

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        """Decode sparse latents back to activation space."""
        return acts @ self.w_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: encode then decode."""
        return self.decode(self.encode(x))

    def get_decoder_vector(self, feature_id: int) -> torch.Tensor:
        """Get the decoder vector for a specific feature."""
        return self.w_dec[feature_id].clone()

    @property
    def hidden_dim(self) -> int:
        """Model hidden dimension."""
        return self.d_in

    @property
    def num_features(self) -> int:
        """Number of SAE features."""
        return self.d_sae


class SAELoader:
    """
    Load GemmaScope SAEs from HuggingFace Hub with caching.

    Supports lazy loading of individual layers and maintains a cache
    to avoid reloading the same SAE multiple times.
    """

    # Known SAE repositories and their layer configurations
    # GemmaScope 2 SAEs are trained on Gemma 3 models (not Gemma 2!)
    # Note: d_model is inferred from loaded weights, these are just for reference
    KNOWN_REPOS = {
        # GemmaScope 1 (Gemma 2 models)
        "google/gemma-scope-2b-it": {
            "layers": list(range(26)),
            "model_family": "gemma2",
        },
        "google/gemma-scope-2b-pt": {
            "layers": list(range(26)),
            "model_family": "gemma2",
        },
        # GemmaScope 2 (Gemma 3 models) - naming is confusing but "gemma-scope-2" = version 2
        "google/gemma-scope-2-1b-it": {
            "layers": list(range(26)),  # Gemma 3 1B has 26 layers
            "model_family": "gemma3",
        },
        "google/gemma-scope-2-4b-it": {
            "layers": list(range(34)),  # Gemma 3 4B has 34 layers
            "model_family": "gemma3",
        },
        "google/gemma-scope-2-12b-it": {
            "layers": list(range(48)),  # Gemma 3 12B has 48 layers
            "model_family": "gemma3",
        },
        "google/gemma-scope-2-27b-it": {
            "layers": list(range(62)),  # Gemma 3 27B has 62 layers
            "model_family": "gemma3",
        },
    }

    # Width string to dimension mapping
    WIDTH_MAP = {
        "16k": 16384,
        "32k": 32768,
        "65k": 65536,
        "131k": 131072,
        "262k": 262144,
    }

    def __init__(self, config: SAEConfig):
        """
        Initialize the SAE loader.

        Args:
            config: SAE configuration
        """
        self.config = config
        self._cache: dict[int, JumpReLUSAE] = {}
        self._available_layers: Optional[list[int]] = None
        self._d_model: Optional[int] = None  # Inferred from loaded weights

        # Get repo info if known
        if config.repo_id in self.KNOWN_REPOS:
            repo_info = self.KNOWN_REPOS[config.repo_id]
            self._available_layers = repo_info["layers"]

    @property
    def available_layers(self) -> list[int]:
        """Get list of available SAE layers."""
        if self._available_layers is None:
            # Try to discover from repo (would need HF API)
            logger.warning(f"Unknown repo {self.config.repo_id}, layer info not available")
            return []
        return self._available_layers

    @property
    def d_model(self) -> Optional[int]:
        """Expected model hidden dimension."""
        return self._d_model

    @property
    def sae_width(self) -> int:
        """SAE width (number of features)."""
        return self.WIDTH_MAP.get(self.config.width, 16384)

    def _get_sae_path(self, layer: int) -> str:
        """
        Construct the path to SAE weights within the repo.

        GemmaScope path pattern:
        {sae_type}/layer_{layer}_width_{width}_l0_{l0}/params.safetensors
        """
        width = self.config.width
        l0 = self.config.l0
        sae_type = self.config.sae_type

        # Handle different path formats
        # Standard format: resid_post_all/layer_0_width_16k_l0_small/params.safetensors
        path = f"{sae_type}/layer_{layer}_width_{width}_l0_{l0}/params.safetensors"
        return path

    def load_sae(self, layer: int) -> JumpReLUSAE:
        """
        Load SAE for a specific layer.

        Uses caching to avoid reloading the same SAE.

        Args:
            layer: Layer index

        Returns:
            Loaded JumpReLUSAE instance
        """
        if layer in self._cache:
            return self._cache[layer]

        logger.info(f"Loading SAE for layer {layer} from {self.config.repo_id}")

        try:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
        except ImportError as e:
            raise ImportError(
                "SAE loading requires huggingface_hub and safetensors. "
                "Install with: pip install huggingface_hub safetensors"
            ) from e

        # Download the SAE weights
        sae_path = self._get_sae_path(layer)

        try:
            local_path = hf_hub_download(
                repo_id=self.config.repo_id,
                filename=sae_path,
                cache_dir=self.config.cache_dir,
                local_files_only=self.config.local_files_only,
            )
        except Exception as e:
            if self.config.local_files_only:
                raise RuntimeError(
                    f"SAE for layer {layer} not found in cache. "
                    f"Either download it first or set local_files_only=False.\n"
                    f"Repo: {self.config.repo_id}, Path: {sae_path}\n"
                    f"Original error: {e}"
                ) from e
            raise RuntimeError(
                f"Failed to download SAE for layer {layer}: {e}\n"
                f"Repo: {self.config.repo_id}, Path: {sae_path}"
            ) from e

        # Load weights
        state_dict = load_file(local_path)

        # Infer dimensions from weights (GemmaScope uses lowercase keys: w_enc, w_dec)
        if "w_enc" in state_dict:
            d_in, d_sae = state_dict["w_enc"].shape
        elif "W_enc" in state_dict:
            d_in, d_sae = state_dict["W_enc"].shape
        else:
            # Try to infer from decoder
            if "w_dec" in state_dict:
                d_sae, d_in = state_dict["w_dec"].shape
            elif "W_dec" in state_dict:
                d_sae, d_in = state_dict["W_dec"].shape
            else:
                raise ValueError(f"Cannot infer SAE dimensions from state dict keys: {state_dict.keys()}")

        # Store d_model for validation
        if self._d_model is None:
            self._d_model = d_in
            logger.info(f"SAE d_model inferred from weights: {d_in}")

        # Create SAE and load weights directly
        # GemmaScope uses lowercase keys matching our parameter names
        sae = JumpReLUSAE(d_in, d_sae)
        sae.load_state_dict(state_dict)
        sae = sae.to(device=self.config.device, dtype=self.config.dtype)
        sae.eval()

        # Cache and return
        self._cache[layer] = sae

        logger.info(f"Loaded SAE layer {layer}: d_model={d_in}, d_sae={d_sae:,}")
        return sae

    def get_decoder_vector(self, layer: int, feature_id: int) -> torch.Tensor:
        """
        Get decoder vector for a specific feature.

        Args:
            layer: SAE layer index
            feature_id: Feature index within the SAE

        Returns:
            Decoder vector [d_model]
        """
        sae = self.load_sae(layer)

        if feature_id < 0 or feature_id >= sae.num_features:
            raise ValueError(
                f"Feature ID {feature_id} out of range [0, {sae.num_features})"
            )

        return sae.get_decoder_vector(feature_id)

    def validate_model_compatibility(
        self,
        model_hidden_dim: int,
        strict: bool = True
    ) -> bool:
        """
        Check if SAE dimensions match the target model.

        Args:
            model_hidden_dim: Target model's hidden dimension
            strict: If True, raise error on mismatch; if False, warn only

        Returns:
            True if compatible
        """
        if self._d_model is None:
            # Load one SAE to check dimension
            if self.available_layers:
                sae = self.load_sae(self.available_layers[0])
                self._d_model = sae.hidden_dim
            else:
                logger.warning("Cannot validate compatibility without loading SAE")
                return True

        if self._d_model != model_hidden_dim:
            msg = (
                f"SAE dimension mismatch: SAE d_model={self._d_model}, "
                f"target model hidden_dim={model_hidden_dim}"
            )
            if strict:
                raise ValueError(msg)
            else:
                logger.warning(msg)
                return False

        return True

    def clear_cache(self):
        """Clear the SAE cache to free memory."""
        self._cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def preload_layers(self, layers: list[int]):
        """
        Preload SAEs for specified layers.

        Args:
            layers: List of layer indices to preload
        """
        for layer in layers:
            self.load_sae(layer)
        logger.info(f"Preloaded {len(layers)} SAEs")


def get_sae_loader(
    repo_id: str = "google/gemma-scope-2-12b-it",
    width: str = "16k",
    l0: str = "small",
    sae_type: str = "resid_post_all",
    device: str = None,
    local_files_only: bool = True,
) -> SAELoader:
    """
    Convenience function to create an SAE loader.

    Args:
        repo_id: HuggingFace repo ID
        width: SAE width (16k, 32k, 65k, 131k, 262k)
        l0: Sparsity variant (small, medium, big)
        sae_type: SAE type/path prefix (resid_post, attn_out_all, etc.)
        device: Device to load SAEs to (default: auto-detect)
        local_files_only: Use cached files only, don't download (default: True)

    Returns:
        Configured SAELoader instance
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = SAEConfig(
        repo_id=repo_id,
        sae_type=sae_type,
        width=width,
        l0=l0,
        device=device,
        local_files_only=local_files_only,
    )

    return SAELoader(config)

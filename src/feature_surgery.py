#!/usr/bin/env python3
"""
Norm-Preserving Feature-Level Surgery (NPFS)

Performs row-level (per-neuron) feature surgery on transformer weight matrices,
enabling precise strengthening or weakening of specific SAE features while
preserving per-neuron activation norms.

Key innovations:
- Row-level attribution: Computes per-neuron contribution scores
- Attribution-gated surgery: Only modifies high-attribution neurons
- Bidirectional modulation: Supports both weakening (mu < 1) and strengthening (mu > 1)
- Per-neuron norm preservation: Maintains activation scale distribution
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# Small epsilon for numerical stability
EPS = 1e-8


@dataclass
class FeatureSpec:
    """Specification for a single feature to modify."""

    layer: int  # SAE layer index
    feature_id: int  # Feature index in SAE
    modulation: float  # <1 weaken, >1 strengthen, 1 = no change
    decoder_vector: Optional[torch.Tensor] = None  # Cached decoder, loaded lazily

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "layer": self.layer,
            "feature_id": self.feature_id,
            "modulation": self.modulation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureSpec":
        """Create from dict."""
        return cls(
            layer=d["layer"],
            feature_id=d["feature_id"],
            modulation=d.get("modulation", 1.0),
        )


@dataclass
class FeatureSurgeryConfig:
    """Configuration for feature surgery."""

    # Target features
    features: list[FeatureSpec] = field(default_factory=list)

    # SAE configuration (for loading decoders)
    sae_repo: str = "google/gemma-scope-2-12b-it"
    sae_width: str = "16k"
    sae_l0: str = "small"
    sae_type: str = "resid_post_all"  # 12b uses resid_post_all, 4b uses resid_post
    sae_local_only: bool = True  # Use cached files only, don't download

    # Surgery parameters
    attribution_percentile: float = 0.9  # Top 10% of neurons
    max_modulation: float = 3.0  # Safety cap
    min_modulation: float = 0.1  # Safety floor

    # Norm preservation
    per_neuron_norm: bool = True

    # Target layers in each transformer block
    target_layer_types: list[str] = field(
        default_factory=lambda: ["mlp.down_proj", "mlp.up_proj"]
    )

    # Device/precision
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype: torch.dtype = torch.float32

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "features": [f.to_dict() for f in self.features],
            "sae_repo": self.sae_repo,
            "sae_width": self.sae_width,
            "sae_l0": self.sae_l0,
            "sae_type": self.sae_type,
            "sae_local_only": self.sae_local_only,
            "attribution_percentile": self.attribution_percentile,
            "max_modulation": self.max_modulation,
            "min_modulation": self.min_modulation,
            "per_neuron_norm": self.per_neuron_norm,
            "target_layer_types": self.target_layer_types,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureSurgeryConfig":
        """Create from dict."""
        features = [FeatureSpec.from_dict(f) for f in d.get("features", [])]
        return cls(
            features=features,
            sae_repo=d.get("sae_repo", "google/gemma-scope-2-12b-it"),
            sae_width=d.get("sae_width", "16k"),
            sae_l0=d.get("sae_l0", "small"),
            sae_type=d.get("sae_type", "resid_post_all"),
            sae_local_only=d.get("sae_local_only", True),
            attribution_percentile=d.get("attribution_percentile", 0.9),
            max_modulation=d.get("max_modulation", 3.0),
            min_modulation=d.get("min_modulation", 0.1),
            per_neuron_norm=d.get("per_neuron_norm", True),
            target_layer_types=d.get("target_layer_types", ["mlp.down_proj", "mlp.up_proj"]),
            device=d.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        )


class FeatureAttributionComputer:
    """
    Compute per-neuron attribution scores for SAE features.

    Attribution quantifies how much each row (neuron) in a weight matrix
    contributes to producing/processing a specific SAE feature.
    """

    def __init__(self, config: FeatureSurgeryConfig):
        """
        Initialize the attribution computer.

        Args:
            config: Feature surgery configuration
        """
        self.config = config
        self._sae_loader = None

    @property
    def sae_loader(self):
        """Lazy-load SAE loader."""
        if self._sae_loader is None:
            from utils.sae_loader import SAELoader, SAEConfig

            sae_config = SAEConfig(
                repo_id=self.config.sae_repo,
                sae_type=self.config.sae_type,
                width=self.config.sae_width,
                l0=self.config.sae_l0,
                device=self.config.device,
                dtype=self.config.compute_dtype,
                local_files_only=self.config.sae_local_only,
            )
            self._sae_loader = SAELoader(sae_config)
        return self._sae_loader

    def get_decoder_vector(self, layer: int, feature_id: int) -> torch.Tensor:
        """
        Get decoder vector for a specific feature.

        Args:
            layer: SAE layer index
            feature_id: Feature index within SAE

        Returns:
            Decoder vector [d_model]
        """
        return self.sae_loader.get_decoder_vector(layer, feature_id)

    def compute_row_attribution(
        self,
        weight: torch.Tensor,
        decoder: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute how much each row (neuron) contributes to the feature.

        For input-space features (decoder matches in_features):
            attribution[i] = |w_i . d| / ||w_i||

        For output-space features (decoder matches out_features):
            attribution = |d|  (direct magnitude)

        Args:
            weight: Weight matrix [out_features, in_features]
            decoder: SAE decoder vector [d_model]

        Returns:
            Attribution scores [out_features]
        """
        weight = weight.float()
        decoder = decoder.float().to(weight.device)

        # Normalize decoder
        d_norm = decoder / (decoder.norm() + EPS)

        out_features, in_features = weight.shape

        if decoder.shape[0] == in_features:
            # Input-space feature: project each row onto decoder direction
            projections = weight @ d_norm  # [out_features]
            row_norms = weight.norm(dim=1)  # [out_features]
            # Normalized projection magnitude as attribution
            attribution = projections.abs() / (row_norms + EPS)
        elif decoder.shape[0] == out_features:
            # Output-space feature: direct magnitude as attribution
            attribution = d_norm.abs()
        else:
            raise ValueError(
                f"Decoder shape {decoder.shape} doesn't match weight matrix "
                f"[{out_features}, {in_features}]"
            )

        return attribution

    def compute_surgery_mask(
        self,
        attribution: torch.Tensor,
        percentile: float = None,
    ) -> tuple[torch.Tensor, float]:
        """
        Create mask selecting high-attribution neurons for surgery.

        Only neurons above the percentile threshold are modified,
        preserving low-attribution neurons that handle other features.

        Args:
            attribution: Attribution scores [num_neurons]
            percentile: Percentile threshold (default: from config)

        Returns:
            (mask, threshold) where mask is boolean [num_neurons]
        """
        if percentile is None:
            percentile = self.config.attribution_percentile

        threshold = torch.quantile(attribution, percentile).item()
        mask = attribution >= threshold

        return mask, threshold


class NormPreservingFeatureSurgery:
    """
    Apply norm-preserving feature surgery to model weights.

    Performs row-level modification of weight matrices to weaken or strengthen
    specific SAE features while preserving per-neuron activation norms.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        config: FeatureSurgeryConfig,
    ):
        """
        Initialize the surgery module.

        Args:
            model: Model to modify
            config: Surgery configuration
        """
        self.model = model
        self.config = config
        self.attribution_computer = FeatureAttributionComputer(config)
        self._stats: dict = {
            "features_processed": 0,
            "weights_modified": 0,
            "neurons_modified": 0,
            "norm_changes": [],
        }

    def _find_layers(self) -> nn.ModuleList:
        """Find transformer layers in the model."""
        model = self.model

        # Try common paths for different architectures
        # Gemma3 multimodal: model.model.language_model.layers
        if (hasattr(model, 'model') and
            hasattr(model.model, 'language_model') and
            hasattr(model.model.language_model, 'layers')):
            return model.model.language_model.layers

        # Vision-Language models: model.model.model.layers
        if (hasattr(model, 'model') and
            hasattr(model.model, 'model') and
            hasattr(model.model.model, 'layers')):
            return model.model.model.layers

        # Standard Llama/Qwen/Gemma: model.model.layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return model.model.layers

        # VL with language_model: model.language_model.model.layers
        if hasattr(model, 'language_model'):
            if hasattr(model.language_model, 'model') and hasattr(model.language_model.model, 'layers'):
                return model.language_model.model.layers
            if hasattr(model.language_model, 'layers'):
                return model.language_model.layers

        # GPT-2 style: model.transformer.h
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            return model.transformer.h

        # Direct layers
        if hasattr(model, 'layers'):
            return model.layers

        raise ValueError(f"Could not find transformer layers in {type(model)}")

    def _get_target_modules(
        self,
        layer: nn.Module,
    ) -> list[tuple[str, nn.Linear]]:
        """
        Get target linear modules in a layer.

        Args:
            layer: Transformer layer module

        Returns:
            List of (name, module) tuples
        """
        targets = []

        for name in self.config.target_layer_types:
            parts = name.split(".")
            module = layer
            try:
                for part in parts:
                    module = getattr(module, part)
                if isinstance(module, nn.Linear):
                    targets.append((name, module))
            except AttributeError:
                continue

        return targets

    @torch.no_grad()
    def apply_feature_surgery(
        self,
        weight: torch.Tensor,
        decoder: torch.Tensor,
        modulation: float,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Apply row-level surgery with per-neuron norm preservation.

        Args:
            weight: Weight matrix [out_features, in_features]
            decoder: SAE decoder vector [d_model]
            modulation: Modulation factor (<1 weaken, >1 strengthen)
            mask: Boolean mask selecting neurons to modify [out_features]

        Returns:
            (modified_weight, stats)
        """
        original_dtype = weight.dtype
        weight = weight.float()
        decoder = decoder.float().to(weight.device)

        # Normalize decoder
        d_norm = decoder / (decoder.norm() + EPS)

        # Store original row norms for norm preservation
        original_row_norms = weight.norm(dim=1, keepdim=True)

        out_features, in_features = weight.shape

        # Compute projection coefficients and adjustment
        if d_norm.shape[0] == in_features:
            # Input-space feature
            proj_coeff = weight @ d_norm  # [out_features]

            # Scale factor: (mu - 1) is negative for weakening, positive for strengthening
            scale = modulation - 1.0

            # Create adjustment matrix
            # adjustment[i] = scale * proj_coeff[i] * d_norm
            adjustment = torch.zeros_like(weight)
            for i in range(out_features):
                if mask[i]:
                    adjustment[i] = scale * proj_coeff[i] * d_norm

        elif d_norm.shape[0] == out_features:
            # Output-space feature
            proj_coeff = d_norm @ weight  # [in_features]
            scale = modulation - 1.0

            adjustment = torch.zeros_like(weight)
            for i in range(out_features):
                if mask[i]:
                    adjustment[i] = scale * d_norm[i] * proj_coeff

        else:
            raise ValueError(
                f"Decoder shape {d_norm.shape} incompatible with weight {weight.shape}"
            )

        # Apply adjustment
        weight_new = weight + adjustment

        # Per-neuron norm preservation (only for modified rows)
        if self.config.per_neuron_norm:
            new_row_norms = weight_new.norm(dim=1, keepdim=True)
            # Compute scale factors: preserve original norm for modified rows
            scale_factors = torch.where(
                (mask.unsqueeze(1)) & (new_row_norms > EPS),
                original_row_norms / new_row_norms,
                torch.ones_like(original_row_norms),
            )
            weight_new = weight_new * scale_factors

        # Compute statistics
        norm_before = original_row_norms[mask].mean().item() if mask.any() else 0.0
        norm_after = weight_new.norm(dim=1, keepdim=True)[mask].mean().item() if mask.any() else 0.0

        stats = {
            "neurons_modified": mask.sum().item(),
            "norm_before": norm_before,
            "norm_after": norm_after,
            "norm_ratio": norm_after / (norm_before + EPS),
        }

        return weight_new.to(original_dtype), stats

    @torch.no_grad()
    def apply_surgery_for_feature(
        self,
        feature: FeatureSpec,
        dry_run: bool = False,
    ) -> dict:
        """
        Apply surgery for a single feature across all target layers.

        Args:
            feature: Feature specification
            dry_run: If True, compute stats but don't modify weights

        Returns:
            Statistics dict
        """
        # Clamp modulation to safety bounds
        modulation = max(
            self.config.min_modulation,
            min(self.config.max_modulation, feature.modulation)
        )

        if modulation == 1.0:
            logger.debug(f"Skipping feature {feature.layer}:{feature.feature_id} (modulation=1.0)")
            return {"skipped": True, "reason": "modulation=1.0"}

        # Get decoder vector
        if feature.decoder_vector is not None:
            decoder = feature.decoder_vector
        else:
            decoder = self.attribution_computer.get_decoder_vector(
                feature.layer, feature.feature_id
            )
            feature.decoder_vector = decoder

        layers = self._find_layers()
        layer_idx = feature.layer

        if layer_idx >= len(layers):
            logger.warning(f"Layer {layer_idx} out of range (model has {len(layers)} layers)")
            return {"error": f"Layer {layer_idx} out of range"}

        layer = layers[layer_idx]
        targets = self._get_target_modules(layer)

        if not targets:
            logger.warning(f"No target modules found in layer {layer_idx}")
            return {"error": "No target modules"}

        feature_stats = {
            "layer": layer_idx,
            "feature_id": feature.feature_id,
            "modulation": modulation,
            "modules": [],
        }

        for module_name, module in targets:
            weight = module.weight.data

            # Compute attribution
            attribution = self.attribution_computer.compute_row_attribution(
                weight, decoder
            )

            # Compute surgery mask
            mask, threshold = self.attribution_computer.compute_surgery_mask(
                attribution, self.config.attribution_percentile
            )

            # Apply surgery
            if not dry_run:
                new_weight, surgery_stats = self.apply_feature_surgery(
                    weight, decoder, modulation, mask
                )
                module.weight.data = new_weight
            else:
                # Dry run: compute stats without modifying
                _, surgery_stats = self.apply_feature_surgery(
                    weight.clone(), decoder, modulation, mask
                )

            module_stats = {
                "name": module_name,
                "attribution_threshold": threshold,
                "attribution_max": attribution.max().item(),
                "attribution_mean": attribution.mean().item(),
                **surgery_stats,
            }
            feature_stats["modules"].append(module_stats)

            self._stats["weights_modified"] += 1
            self._stats["neurons_modified"] += surgery_stats["neurons_modified"]

        self._stats["features_processed"] += 1
        return feature_stats

    @torch.no_grad()
    def apply_multi_feature_surgery(
        self,
        features: list[FeatureSpec] = None,
        dry_run: bool = False,
    ) -> dict:
        """
        Apply surgery for multiple features.

        Args:
            features: List of feature specs (default: from config)
            dry_run: If True, compute stats but don't modify weights

        Returns:
            Combined statistics dict
        """
        if features is None:
            features = self.config.features

        if not features:
            logger.warning("No features specified for surgery")
            return {"error": "No features"}

        # Reset stats
        self._stats = {
            "features_processed": 0,
            "weights_modified": 0,
            "neurons_modified": 0,
            "feature_results": [],
        }

        action = "Analyzing" if dry_run else "Applying surgery for"
        for feature in tqdm(features, desc=f"{action} features"):
            result = self.apply_surgery_for_feature(feature, dry_run=dry_run)
            self._stats["feature_results"].append(result)

        logger.info(
            f"Surgery complete: {self._stats['features_processed']} features, "
            f"{self._stats['weights_modified']} weight matrices, "
            f"{self._stats['neurons_modified']} neurons modified"
        )

        return self._stats

    def get_stats(self) -> dict:
        """Get surgery statistics."""
        return self._stats


class FeatureSurgeryPipeline:
    """
    End-to-end pipeline for feature surgery.

    Handles loading models, features, applying surgery, and saving results.
    """

    def __init__(self, config: FeatureSurgeryConfig):
        """
        Initialize the pipeline.

        Args:
            config: Surgery configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self._surgery = None

    @classmethod
    def from_differential_features(
        cls,
        differential_path: str,
        top_k_per_layer: int = 10,
        weaken_strength: float = 0.3,
        strengthen_strength: float = 1.5,
        sae_repo: str = "google/gemma-scope-2-12b-it",
        **config_kwargs,
    ) -> "FeatureSurgeryPipeline":
        """
        Create pipeline from circuit_analysis differential output.

        Automatically selects features to weaken (high in harmful/set A)
        and optionally strengthen (high in harmless/set B).

        Args:
            differential_path: Path to differential analysis JSON
            top_k_per_layer: Number of features per layer
            weaken_strength: Modulation for features to weaken (0-1)
            strengthen_strength: Modulation for features to strengthen (>1)
            sae_repo: SAE repository ID
            **config_kwargs: Additional config parameters

        Returns:
            Configured pipeline
        """
        with open(differential_path, "r", encoding="utf-8") as f:
            diff_data = json.load(f)

        features = []

        # Process each layer's differential features
        layers_data = diff_data.get("layers", {})
        for layer_str, layer_features in layers_data.items():
            layer = int(layer_str)

            # Sort by absolute difference
            sorted_features = sorted(
                layer_features,
                key=lambda x: abs(x.get("diff", x.get("mean_diff", 0))),
                reverse=True
            )[:top_k_per_layer]

            for feat in sorted_features:
                feature_id = feat.get("feature_id", feat.get("feature"))
                diff = feat.get("diff", feat.get("mean_diff", 0))

                # Positive diff = higher in set A (harmful) -> weaken
                # Negative diff = higher in set B (harmless) -> strengthen
                if diff > 0:
                    modulation = weaken_strength
                else:
                    modulation = strengthen_strength

                features.append(FeatureSpec(
                    layer=layer,
                    feature_id=feature_id,
                    modulation=modulation,
                ))

        logger.info(f"Loaded {len(features)} features from differential analysis")

        config = FeatureSurgeryConfig(
            features=features,
            sae_repo=sae_repo,
            **config_kwargs,
        )

        return cls(config)

    @classmethod
    def from_feature_list(
        cls,
        features: list[dict],
        sae_repo: str = "google/gemma-scope-2-12b-it",
        **config_kwargs,
    ) -> "FeatureSurgeryPipeline":
        """
        Create pipeline from explicit feature list.

        Args:
            features: List of {layer, feature_id, modulation} dicts
            sae_repo: SAE repository ID
            **config_kwargs: Additional config parameters

        Returns:
            Configured pipeline
        """
        feature_specs = [FeatureSpec.from_dict(f) for f in features]

        config = FeatureSurgeryConfig(
            features=feature_specs,
            sae_repo=sae_repo,
            **config_kwargs,
        )

        return cls(config)

    @classmethod
    def from_json(cls, path: str) -> "FeatureSurgeryPipeline":
        """
        Load pipeline configuration from JSON file.

        Args:
            path: Path to config JSON

        Returns:
            Configured pipeline
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        config = FeatureSurgeryConfig.from_dict(data)
        return cls(config)

    def load_model(
        self,
        model_path: str,
        dtype: torch.dtype = None,
    ) -> AutoModelForCausalLM:
        """
        Load model for surgery.

        Args:
            model_path: Path or HuggingFace ID
            dtype: Model dtype (default: auto)

        Returns:
            Loaded model
        """
        logger.info(f"Loading model: {model_path}")

        if dtype is None:
            dtype = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        # Initialize surgery module
        self._surgery = NormPreservingFeatureSurgery(self.model, self.config)

        return self.model

    def validate_sae_compatibility(self) -> bool:
        """
        Check if SAE dimensions match the model.

        Returns:
            True if compatible
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Get model hidden dimension - handle different config structures
        config = self.model.config
        hidden_dim = None

        # Try common attribute names
        if hasattr(config, 'hidden_size'):
            hidden_dim = config.hidden_size
        elif hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
            # Gemma 3 and other multimodal models
            hidden_dim = config.text_config.hidden_size
        elif hasattr(config, 'd_model'):
            hidden_dim = config.d_model
        elif hasattr(config, 'n_embd'):
            # GPT-2 style
            hidden_dim = config.n_embd
        elif hasattr(config, 'hidden_dim'):
            hidden_dim = config.hidden_dim

        if hidden_dim is None:
            logger.warning(
                f"Could not determine hidden_dim from config: {type(config).__name__}. "
                "Skipping SAE compatibility check."
            )
            return True

        # Validate with SAE loader
        return self._surgery.attribution_computer.sae_loader.validate_model_compatibility(
            hidden_dim, strict=False
        )

    def run(
        self,
        output_path: str = None,
        dry_run: bool = False,
    ) -> dict:
        """
        Run the full surgery pipeline.

        Args:
            output_path: Path to save modified model (None = don't save)
            dry_run: If True, compute stats but don't modify weights

        Returns:
            Surgery statistics
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        logger.info(f"Running feature surgery ({'dry run' if dry_run else 'applying changes'})")

        # Apply surgery
        stats = self._surgery.apply_multi_feature_surgery(dry_run=dry_run)

        # Save if requested
        if output_path and not dry_run:
            self.save(output_path, stats)

        return stats

    def save(self, output_path: str, stats: dict = None):
        """
        Save the modified model and config.

        Args:
            output_path: Directory to save to
            stats: Optional stats to include in metadata
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving model to {output_path}")

        # Save model and tokenizer
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        # Save config
        config_path = output_path / "feature_surgery_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save stats if provided
        if stats:
            stats_path = output_path / "feature_surgery_stats.json"
            # Filter non-serializable items
            serializable_stats = {
                k: v for k, v in stats.items()
                if k != "feature_results"
            }
            serializable_stats["num_features"] = len(self.config.features)
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(serializable_stats, f, indent=2)

        logger.info(f"Saved model and config to {output_path}")

    def export_config(self, path: str):
        """
        Save configuration for reproducibility.

        Args:
            path: Path to save config JSON
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        logger.info(f"Exported config to {path}")


def weaken_feature(
    model: AutoModelForCausalLM,
    layer: int,
    feature_id: int,
    strength: float = 0.3,
    sae_repo: str = "google/gemma-scope-2-12b-it",
    **kwargs,
) -> dict:
    """
    Convenience function to weaken a single feature.

    Args:
        model: Model to modify
        layer: SAE layer index
        feature_id: Feature index
        strength: Modulation strength (0 = full removal, 1 = no change)
        sae_repo: SAE repository
        **kwargs: Additional config parameters

    Returns:
        Surgery statistics
    """
    config = FeatureSurgeryConfig(
        features=[FeatureSpec(layer=layer, feature_id=feature_id, modulation=strength)],
        sae_repo=sae_repo,
        **kwargs,
    )
    surgery = NormPreservingFeatureSurgery(model, config)
    return surgery.apply_multi_feature_surgery()


def strengthen_feature(
    model: AutoModelForCausalLM,
    layer: int,
    feature_id: int,
    strength: float = 1.5,
    sae_repo: str = "google/gemma-scope-2-12b-it",
    **kwargs,
) -> dict:
    """
    Convenience function to strengthen a single feature.

    Args:
        model: Model to modify
        layer: SAE layer index
        feature_id: Feature index
        strength: Modulation strength (>1 = amplification)
        sae_repo: SAE repository
        **kwargs: Additional config parameters

    Returns:
        Surgery statistics
    """
    config = FeatureSurgeryConfig(
        features=[FeatureSpec(layer=layer, feature_id=feature_id, modulation=strength)],
        sae_repo=sae_repo,
        **kwargs,
    )
    surgery = NormPreservingFeatureSurgery(model, config)
    return surgery.apply_multi_feature_surgery()

#!/usr/bin/env python3
"""
Norm-Preserving Feature Surgery CLI

Command-line interface for applying feature-level surgery to transformer models
using SAE decoder vectors.

Usage:
    # From differential analysis (circuit_analysis output)
    python -m src.feature_surgery_cli \
        --model google/gemma-3-4b-it \
        --differential diff_features/differential.json \
        --weaken-strength 0.3 \
        --output models/gemma3-4b-surgery/

    # From explicit feature list
    python -m src.feature_surgery_cli \
        --model google/gemma-3-4b-it \
        --features-json features.json \
        --output models/gemma3-4b-surgery/

    # Single feature operation
    python -m src.feature_surgery_cli \
        --model google/gemma-3-4b-it \
        --layer 15 --feature-id 1234 --modulation 0.3 \
        --output models/gemma3-4b-surgery/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Norm-Preserving Feature Surgery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From differential analysis
  %(prog)s -m google/gemma-3-4b-it --differential diff.json -o output/

  # From feature list
  %(prog)s -m google/gemma-3-4b-it --features-json features.json -o output/

  # Single feature
  %(prog)s -m google/gemma-3-4b-it --layer 15 --feature-id 1234 --modulation 0.3 -o output/
        """,
    )

    # Model paths
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Model path or HuggingFace ID",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for modified model",
    )

    # Feature specification (one of these required)
    feature_group = parser.add_argument_group("Feature Specification (choose one)")
    feature_group.add_argument(
        "--features-json",
        help="JSON file with feature specs [{layer, feature_id, modulation}, ...]",
    )
    feature_group.add_argument(
        "--differential",
        help="Path to differential analysis JSON from circuit_analysis",
    )
    feature_group.add_argument(
        "--config",
        help="Path to full FeatureSurgeryConfig JSON",
    )

    # Single feature operation
    single_group = parser.add_argument_group("Single Feature Operation")
    single_group.add_argument(
        "--layer",
        type=int,
        help="SAE layer index for single feature operation",
    )
    single_group.add_argument(
        "--feature-id",
        type=int,
        help="Feature ID for single feature operation",
    )
    single_group.add_argument(
        "--modulation",
        type=float,
        default=0.3,
        help="Modulation factor (<1 weaken, >1 strengthen) (default: 0.3)",
    )

    # SAE configuration
    sae_group = parser.add_argument_group("SAE Configuration")
    sae_group.add_argument(
        "--sae-repo",
        default="google/gemma-scope-2-12b-it",
        help="SAE repository ID (default: google/gemma-scope-2-12b-it)",
    )
    sae_group.add_argument(
        "--sae-width",
        default="16k",
        choices=["16k", "32k", "65k", "131k", "262k"],
        help="SAE width (default: 16k)",
    )
    sae_group.add_argument(
        "--sae-l0",
        default="small",
        choices=["small", "medium", "big"],
        help="SAE L0 variant (default: small)",
    )
    sae_group.add_argument(
        "--sae-type",
        default="resid_post_all",
        help="SAE type/path prefix (default: resid_post_all for 12b, use resid_post for 4b)",
    )

    # Surgery parameters
    surgery_group = parser.add_argument_group("Surgery Parameters")
    surgery_group.add_argument(
        "--attribution-percentile",
        type=float,
        default=0.9,
        help="Attribution percentile threshold (default: 0.9 = top 10%%)",
    )
    surgery_group.add_argument(
        "--weaken-strength",
        type=float,
        default=0.3,
        help="Modulation for features to weaken (0-1) (default: 0.3)",
    )
    surgery_group.add_argument(
        "--strengthen-strength",
        type=float,
        default=1.5,
        help="Modulation for features to strengthen (>1) (default: 1.5)",
    )
    surgery_group.add_argument(
        "--top-k-per-layer",
        type=int,
        default=10,
        help="Top features per layer from differential analysis (default: 10)",
    )
    surgery_group.add_argument(
        "--target-layers",
        nargs="+",
        default=["mlp.down_proj", "mlp.up_proj"],
        help="Layer types to target (default: mlp.down_proj mlp.up_proj)",
    )

    # Options
    options_group = parser.add_argument_group("Options")
    options_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute stats but don't modify weights",
    )
    options_group.add_argument(
        "--no-per-neuron-norm",
        action="store_true",
        help="Disable per-neuron norm preservation",
    )
    options_group.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Model dtype (default: float16)",
    )
    options_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    options_group.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip SAE compatibility validation",
    )
    options_group.add_argument(
        "--download-sae",
        action="store_true",
        help="Download SAE if not cached locally (default: use cached only)",
    )

    return parser.parse_args()


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch.dtype."""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map[dtype_str]


def main():
    """Main CLI entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Import here to avoid slow startup
    from src.feature_surgery import (
        FeatureSurgeryPipeline,
        FeatureSurgeryConfig,
        FeatureSpec,
    )

    # Determine which input mode
    has_differential = args.differential is not None
    has_features_json = args.features_json is not None
    has_config = args.config is not None
    has_single = args.layer is not None and args.feature_id is not None

    input_count = sum([has_differential, has_features_json, has_config, has_single])

    if input_count == 0:
        logger.error(
            "Must specify one of: --differential, --features-json, --config, "
            "or --layer/--feature-id"
        )
        sys.exit(1)
    elif input_count > 1:
        logger.error("Only one input mode allowed")
        sys.exit(1)

    # Create pipeline based on input mode
    if has_config:
        logger.info(f"Loading config from: {args.config}")
        pipeline = FeatureSurgeryPipeline.from_json(args.config)

    elif has_differential:
        logger.info(f"Loading differential features from: {args.differential}")
        pipeline = FeatureSurgeryPipeline.from_differential_features(
            differential_path=args.differential,
            top_k_per_layer=args.top_k_per_layer,
            weaken_strength=args.weaken_strength,
            strengthen_strength=args.strengthen_strength,
            sae_repo=args.sae_repo,
            sae_type=args.sae_type,
            sae_width=args.sae_width,
            sae_l0=args.sae_l0,
            sae_local_only=not args.download_sae,
            attribution_percentile=args.attribution_percentile,
            per_neuron_norm=not args.no_per_neuron_norm,
            target_layer_types=args.target_layers,
        )

    elif has_features_json:
        logger.info(f"Loading features from: {args.features_json}")
        with open(args.features_json, "r", encoding="utf-8") as f:
            features = json.load(f)

        pipeline = FeatureSurgeryPipeline.from_feature_list(
            features=features,
            sae_repo=args.sae_repo,
            sae_type=args.sae_type,
            sae_width=args.sae_width,
            sae_l0=args.sae_l0,
            sae_local_only=not args.download_sae,
            attribution_percentile=args.attribution_percentile,
            per_neuron_norm=not args.no_per_neuron_norm,
            target_layer_types=args.target_layers,
        )

    else:  # has_single
        logger.info(f"Single feature: layer={args.layer}, feature_id={args.feature_id}")
        config = FeatureSurgeryConfig(
            features=[FeatureSpec(
                layer=args.layer,
                feature_id=args.feature_id,
                modulation=args.modulation,
            )],
            sae_repo=args.sae_repo,
            sae_type=args.sae_type,
            sae_width=args.sae_width,
            sae_l0=args.sae_l0,
            sae_local_only=not args.download_sae,
            attribution_percentile=args.attribution_percentile,
            per_neuron_norm=not args.no_per_neuron_norm,
            target_layer_types=args.target_layers,
        )
        pipeline = FeatureSurgeryPipeline(config)

    # Load model
    dtype = get_dtype(args.dtype)
    pipeline.load_model(args.model, dtype=dtype)

    # Validate SAE compatibility
    if not args.skip_validation:
        logger.info("Validating SAE compatibility...")
        compatible = pipeline.validate_sae_compatibility()
        if not compatible:
            logger.warning(
                "SAE dimensions may not match model. "
                "Use --skip-validation to proceed anyway."
            )

    # Run surgery
    logger.info(f"Running feature surgery ({'dry run' if args.dry_run else 'applying changes'})...")
    stats = pipeline.run(
        output_path=args.output if not args.dry_run else None,
        dry_run=args.dry_run,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("FEATURE SURGERY SUMMARY")
    print("=" * 60)
    print(f"Features processed: {stats.get('features_processed', 0)}")
    print(f"Weight matrices modified: {stats.get('weights_modified', 0)}")
    print(f"Total neurons modified: {stats.get('neurons_modified', 0)}")

    if args.dry_run:
        print("\n[DRY RUN - No changes were made]")
    else:
        print(f"\nModel saved to: {args.output}")

    print("=" * 60)


if __name__ == "__main__":
    main()

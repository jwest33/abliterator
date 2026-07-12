#!/usr/bin/env python3
"""
Interactive CLI for Abliteration Toolkit

A modern terminal interface for removing refusal behavior from language models.
Supports interactive mode (default) and batch mode for automation.
"""

import gc
import json
import os
import sys

# Fix Windows encoding issues with tokenizer files
# Must be set before any file I/O operations
if sys.platform == "win32":
    os.environ.setdefault("PYTHONUTF8", "1")
    # Also try to set console output encoding
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, OSError):
        pass
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import questionary
import torch
from questionary import Style as QStyle
from rich.console import Console
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from src.abliterate import get_default_prompts_path
from utils.refusal_eval import RefusalScanner
from src.gguf_export import (
    GGUFExportConfig,
    QUANT_TYPES,
    QUANT_ONLY_TYPES,
    check_tools_available,
    detect_vision_model,
    export_to_gguf,
    export_all_quants,
    find_llama_cpp_path,
    has_vision_files,
)
from src.cli_components import (
    THEME,
    add_model_path,
    clear_screen,
    console,
    copy_prompts_to_user_dir,
    create_progress_bar,
    display_banner,
    display_comparison_panel,
    display_config_panel,
    display_error,
    display_menu,
    display_model_list,
    display_results_table,
    display_success,
    display_system_info,
    display_training_config_details,
    display_training_config_list,
    display_warning,
    find_models,
    get_config_path,
    get_default_direction_multiplier,
    get_default_dtype,
    get_default_num_prompts,
    get_default_output_dir,
    get_eval_results_dir,
    get_model_paths,
    get_user_prompts_dir,
    get_versioned_path,
    load_config,
    print_divider,
    remove_model_path,
    save_config,
    set_default_output_dir,
    set_eval_results_dir,
    user_prompts_exist,
)
from src.config_manager import (
    config_exists,
    delete_training_config,
    get_configs_dir,
    get_default_settings,
    list_configs,
    load_training_config,
    save_training_config,
    settings_from_abliteration_config,
    apply_config_to_runtime,
    sanitize_config_name,
    validate_config_settings,
)

# Questionary custom style (orange theme)
custom_style = QStyle([
    ("qmark", "fg:#ff8c00 bold"),
    ("question", "fg:#00ffff bold"),
    ("answer", "fg:#ff8c00 bold"),
    ("pointer", "fg:#ff8c00 bold"),
    ("highlighted", "fg:#ff8c00 bold"),
    ("selected", "fg:#ffa500"),
    ("separator", "fg:#6c6c6c"),
    ("instruction", "fg:#6c6c6c"),
])


def select_model(title: str = "Select a model", allow_manual: bool = True) -> Optional[str]:
    """Interactive model selection."""
    models = find_models()

    if not models and not allow_manual:
        display_error("No models found in common locations.")
        return None

    choices = []
    for model in models:
        prefix = "[A] " if model["is_abliterated"] else "    "
        choices.append(questionary.Choice(
            title=f"{prefix}{model['name']}",
            value=model["path"],
        ))

    if allow_manual:
        choices.append(questionary.Choice(
            title="[M] Enter path manually...",
            value="__manual__",
        ))

    choices.append(questionary.Choice(
        title="[B] Back",
        value="__back__",
    ))

    selected = questionary.select(
        title,
        choices=choices,
        style=custom_style,
    ).ask()

    if selected == "__back__" or selected is None:
        return None

    if selected == "__manual__":
        path = questionary.path(
            "Enter model path:",
            style=custom_style,
        ).ask()
        return path

    return selected


def get_abliteration_config() -> Optional[dict]:
    """Interactive configuration for abliteration."""
    config = {}
    loaded_settings = None

    # Step 0: Check if user wants to load a saved config
    saved_configs = list_configs()
    if saved_configs:
        console.print(f"\n[bold {THEME['primary']}]Configuration Source[/bold {THEME['primary']}]\n")

        config_choice = questionary.select(
            "How would you like to configure this abliteration?",
            choices=[
                questionary.Choice("New config", value="fresh"),
                questionary.Choice("Load saved config", value="load"),
            ],
            style=custom_style,
        ).ask()

        if config_choice is None:
            return None

        if config_choice == "load":
            loaded_settings = _select_training_config_for_abliteration()
            if loaded_settings is None:
                # User cancelled or chose to go back - continue with manual config
                pass

    # Model selection
    console.print(f"\n[bold {THEME['primary']}]Step 1: Select Base Model[/bold {THEME['primary']}]\n")
    model_path = select_model("Select the model to abliterate:")
    if not model_path:
        return None
    config["model_path"] = model_path

    # Detect architecture immediately after model selection
    from src.abliterate import detect_hybrid_architecture
    hybrid_info = detect_hybrid_architecture(model_path)

    if hybrid_info.is_hybrid:
        console.print(f"\n[bold {THEME['success']}]Hybrid architecture detected[/bold {THEME['success']}]")
        console.print(
            f"  {len(hybrid_info.full_attention_indices)} full attention + "
            f"{len(hybrid_info.linear_attention_indices)} linear attention layers "
            f"(every {hybrid_info.full_attention_interval}th layer is full attention)"
        )

    # Output path
    console.print(f"\n[bold {THEME['primary']}]Step 2: Output Path[/bold {THEME['primary']}]\n")
    output_dir = get_default_output_dir()
    default_output = f"{output_dir}/{Path(model_path).name}-abliterated"

    use_default = questionary.confirm(
        f"Use default output path? ({default_output})",
        default=True,
        style=custom_style,
    ).ask()

    if use_default:
        config["output_path"] = default_output
    else:
        config["output_path"] = questionary.path(
            "Enter output path:",
            default=default_output,
            style=custom_style,
        ).ask()

    # If we loaded settings from a config, apply them and skip manual configuration
    if loaded_settings:
        config.update(loaded_settings)
        config["_loaded_from_saved_config"] = True  # Flag to skip save prompt later
        console.print(f"\n[{THEME['success']}]Using settings from loaded config.[/{THEME['success']}]")
        return config

    # Manual configuration - Advanced options
    console.print(f"\n[bold {THEME['primary']}]Step 3: Configuration[/bold {THEME['primary']}]\n")

    config["num_prompts"] = int(questionary.text(
        "Number of prompts to use:",
        default=str(get_default_num_prompts()),
        style=custom_style,
    ).ask())

    config["direction_multiplier"] = float(questionary.text(
        "Direction multiplier (ablation strength):",
        default=str(get_default_direction_multiplier()),
        style=custom_style,
    ).ask())

    config["norm_preservation"] = questionary.confirm(
        "Enable norm preservation? (recommended)",
        default=True,
        style=custom_style,
    ).ask()

    config["filter_prompts"] = questionary.confirm(
        "Filter harmful prompts by refusal? (recommended)",
        default=True,
        style=custom_style,
    ).ask()

    # Device selection
    if torch.cuda.is_available():
        config["device"] = questionary.select(
            "Select device:",
            choices=["cuda", "cpu"],
            default="cuda",
            style=custom_style,
        ).ask()
    else:
        config["device"] = "cpu"
        console.print(f"[{THEME['warning']}]CUDA not available, using CPU[/{THEME['warning']}]")

    # Dtype selection
    config["dtype"] = questionary.select(
        "Select precision:",
        choices=[
            questionary.Choice("float16 (faster, less memory)", value="float16"),
            questionary.Choice("bfloat16 (better precision)", value="bfloat16"),
            questionary.Choice("float32 (full precision)", value="float32"),
        ],
        default=get_default_dtype(),
        style=custom_style,
    ).ask()

    # Step 4: Advanced Options
    console.print(f"\n[bold {THEME['primary']}]Step 4: Advanced Options[/bold {THEME['primary']}]\n")

    # Set defaults for all advanced options
    config["use_projected_refusal"] = True
    config["use_winsorization"] = False
    config["winsorize_percentile"] = 0.995
    config["use_magnitude_clipping"] = False
    config["magnitude_clip_percentile"] = 0.99
    config["use_null_space"] = False
    config["preservation_prompts_path"] = None
    config["null_space_rank_ratio"] = 0.95
    config["use_adaptive_weighting"] = False
    config["use_biprojection"] = False
    config["use_per_neuron_norm"] = False
    config["target_layer_types"] = None
    config["use_harmless_boundary"] = False
    config["harmless_clamp_ratio"] = 0.1
    config["num_measurement_layers"] = 2
    config["intervention_range"] = (0.25, 0.95)
    config["dynamic_layer_targeting"] = False
    config["use_gabliteration"] = False
    config["gab_num_directions"] = 2
    config["gab_ridge_lambda"] = 0.1
    config["gab_layer_scaling_beta"] = 0.5
    config["gab_skip_first_layers"] = 2
    config["gab_skip_last_layers"] = 2
    config["hybrid_strategy"] = "auto"
    config["hybrid_full_attn_weight"] = 1.0
    config["hybrid_linear_attn_weight"] = 0.4
    config["hybrid_skip_recurrent_proj"] = True
    config["hybrid_skip_state_proj"] = False
    # Numerical stability (defaults: all on)
    config["use_welford_mean"] = True
    config["use_float64_subtraction"] = True
    # KL divergence monitoring
    config["use_kl_monitoring"] = False
    config["kl_reference_prompts_path"] = None
    config["kl_num_reference_prompts"] = 50
    config["kl_top_k"] = 200
    config["use_kl_auto_tune"] = False
    config["kl_threshold"] = 0.5
    config["kl_search_min"] = 0.1
    config["kl_search_max"] = 2.0

    # Build preset choices based on detected architecture
    preset_choices = []
    if hybrid_info.is_hybrid:
        preset_choices.append(questionary.Choice(
            "Recommended for hybrid model (hybrid-aware + null-space + winsorization)",
            value="hybrid_recommended",
        ))
        preset_choices.append(questionary.Choice(
            "Hybrid + biprojection (best NatInt preservation for hybrid models)",
            value="hybrid_biprojection",
        ))

    preset_choices.extend([
        questionary.Choice("Baseline (no advanced options)", value="baseline"),
        questionary.Choice("Recommended standard (winsorize + null-space)", value="standard_recommended"),
        questionary.Choice("Gabliteration (multi-directional SVD)", value="gabliteration"),
        questionary.Choice("Custom (configure each option manually)", value="custom"),
    ])

    preset = questionary.select(
        "Select configuration preset:",
        choices=preset_choices,
        style=custom_style,
    ).ask()

    if preset == "hybrid_recommended":
        config["use_winsorization"] = True
        config["winsorize_percentile"] = 0.995
        config["use_null_space"] = True
        config["hybrid_strategy"] = "auto"
        config["hybrid_skip_recurrent_proj"] = True
        config["hybrid_skip_state_proj"] = False

    elif preset == "hybrid_biprojection":
        config["use_winsorization"] = True
        config["winsorize_percentile"] = 0.995
        config["use_null_space"] = True
        config["hybrid_strategy"] = "auto"
        config["hybrid_skip_recurrent_proj"] = True
        config["hybrid_skip_state_proj"] = False
        config["use_biprojection"] = True
        config["use_per_neuron_norm"] = True
        config["target_layer_types"] = ["o_proj", "down_proj"]
        config["use_harmless_boundary"] = True
        config["harmless_clamp_ratio"] = 0.1

    elif preset == "standard_recommended":
        config["use_winsorization"] = True
        config["winsorize_percentile"] = 0.995
        config["use_null_space"] = True
        if hybrid_info.is_hybrid:
            config["hybrid_strategy"] = "auto"
            config["hybrid_skip_recurrent_proj"] = True

    # For presets that enable null-space, prompt for rank ratio
    if preset in ("hybrid_recommended", "hybrid_biprojection", "standard_recommended"):
        config["null_space_rank_ratio"] = float(questionary.text(
            "Null-space SVD rank ratio (0.1-0.99, lower = more aggressive):",
            default="0.95",
            style=custom_style,
        ).ask())

        # Print summary
        console.print(f"\n[{THEME['success']}]Applied preset settings:[/{THEME['success']}]")
        console.print(f"[{THEME['muted']}]  → Winsorization enabled (0.995)[/{THEME['muted']}]")
        console.print(f"[{THEME['muted']}]  → Null-space constraints enabled (rank ratio {config['null_space_rank_ratio']})[/{THEME['muted']}]")
        if config.get("hybrid_strategy") == "auto" and hybrid_info.is_hybrid:
            console.print(f"[{THEME['muted']}]  → Hybrid-aware extraction (full attn + pre-full-attn layers)[/{THEME['muted']}]")
            console.print(f"[{THEME['muted']}]  → Full attention layers: {config['hybrid_full_attn_weight']}x weight[/{THEME['muted']}]")
            console.print(f"[{THEME['muted']}]  → Linear attention layers: {config['hybrid_linear_attn_weight']}x weight[/{THEME['muted']}]")
            console.print(f"[{THEME['muted']}]  → Skipping in_proj_a, in_proj_b (recurrent dynamics)[/{THEME['muted']}]")
        if config.get("use_biprojection"):
            console.print(f"[{THEME['muted']}]  → Biprojection with per-neuron norm[/{THEME['muted']}]")
            console.print(f"[{THEME['muted']}]  → Targeting o_proj, down_proj only[/{THEME['muted']}]")
            console.print(f"[{THEME['muted']}]  → Harmless boundary clamping ({config['harmless_clamp_ratio']})[/{THEME['muted']}]")

    elif preset == "gabliteration":
        config["use_gabliteration"] = True
        config["use_winsorization"] = True
        config["winsorize_percentile"] = 0.995
        config["gab_num_directions"] = int(questionary.text(
            "Number of SVD directions (1-10, paper recommends 1-3):",
            default="2",
            style=custom_style,
        ).ask())
        config["gab_ridge_lambda"] = float(questionary.text(
            "Ridge regularization lambda (0.0-1.0):",
            default="0.1",
            style=custom_style,
        ).ask())
        if hybrid_info.is_hybrid:
            config["hybrid_strategy"] = "auto"
            config["hybrid_skip_recurrent_proj"] = True
        console.print(f"\n[{THEME['success']}]Applied Gabliteration preset:[/{THEME['success']}]")
        console.print(f"[{THEME['muted']}]  → Multi-directional SVD (k={config['gab_num_directions']})[/{THEME['muted']}]")
        console.print(f"[{THEME['muted']}]  → Ridge regularization (λ={config['gab_ridge_lambda']})[/{THEME['muted']}]")
        console.print(f"[{THEME['muted']}]  → Winsorization enabled (0.995)[/{THEME['muted']}]")
        console.print(f"[{THEME['muted']}]  → Layer scaling (β={config['gab_layer_scaling_beta']}, skip first {config['gab_skip_first_layers']}, last {config['gab_skip_last_layers']})[/{THEME['muted']}]")

    elif preset == "baseline":
        if hybrid_info.is_hybrid:
            config["hybrid_strategy"] = "auto"
            config["hybrid_skip_recurrent_proj"] = True
            console.print(f"\n[{THEME['muted']}]Baseline with hybrid-aware mode auto-enabled[/{THEME['muted']}]")
        else:
            console.print(f"\n[{THEME['muted']}]Using baseline settings (no advanced options)[/{THEME['muted']}]")

    use_advanced = preset == "custom"
    if use_advanced:
        # Winsorization (recommended for Gemma models)
        config["use_winsorization"] = questionary.confirm(
            "Enable Winsorization? (clips outlier activations, recommended for Gemma models)",
            default=False,
            style=custom_style,
        ).ask()

        if config["use_winsorization"]:
            config["winsorize_percentile"] = float(questionary.text(
                "Winsorize percentile (0.99-0.999):",
                default="0.995",
                style=custom_style,
            ).ask())

        # Magnitude clipping (alternative to Winsorization)
        if not config["use_winsorization"]:
            config["use_magnitude_clipping"] = questionary.confirm(
                "Enable magnitude clipping? (global outlier clipping, alternative to Winsorization)",
                default=False,
                style=custom_style,
            ).ask()

            if config["use_magnitude_clipping"]:
                config["magnitude_clip_percentile"] = float(questionary.text(
                    "Magnitude clip percentile (0.9-0.999):",
                    default="0.99",
                    style=custom_style,
                ).ask())

        # Null-space constraints (preserves model capabilities)
        config["use_null_space"] = questionary.confirm(
            "Enable null-space constraints? (preserves model capabilities)",
            default=False,
            style=custom_style,
        ).ask()

        if config["use_null_space"]:
            use_default_preservation = questionary.confirm(
                "Use default preservation prompts?",
                default=True,
                style=custom_style,
            ).ask()

            if use_default_preservation:
                config["preservation_prompts_path"] = None
            else:
                config["preservation_prompts_path"] = questionary.path(
                    "Path to custom preservation prompts:",
                    style=custom_style,
                ).ask()

            config["null_space_rank_ratio"] = float(questionary.text(
                "Null-space SVD rank ratio (0.9-0.99):",
                default="0.95",
                style=custom_style,
            ).ask())

        # Layer targeting options
        console.print(f"\n[bold {THEME['secondary']}]Layer Targeting[/bold {THEME['secondary']}]")

        # Dynamic layer targeting (per-layer directions)
        config["dynamic_layer_targeting"] = questionary.confirm(
            "Enable dynamic layer targeting? (extract ALL layers, use per-layer directions & null-space)",
            default=False,
            style=custom_style,
        ).ask()

        if config["dynamic_layer_targeting"]:
            console.print(f"[{THEME['muted']}]  → Activations will be extracted from all layers[/{THEME['muted']}]")
            console.print(f"[{THEME['muted']}]  → Each layer gets its own refusal direction[/{THEME['muted']}]")
            console.print(f"[{THEME['muted']}]  → Null-space projectors are computed per-layer[/{THEME['muted']}]")

        # Explain how strength control works with dynamic targeting
        strength_question = "How would you like to control per-layer ablation strength?"
        if config["dynamic_layer_targeting"]:
            console.print(f"\n[{THEME['muted']}]With dynamic targeting, each layer uses its own direction.[/{THEME['muted']}]")
            console.print(f"[{THEME['muted']}]Strength control scales how much each layer's direction is applied.[/{THEME['muted']}]")

        layer_targeting_choice = questionary.select(
            strength_question,
            choices=[
                questionary.Choice("None (uniform strength across all layers)", value="none"),
                questionary.Choice("Adaptive weighting (Gaussian, stronger in middle layers)", value="adaptive"),
                questionary.Choice("Layer target map (JSON file with custom weights)", value="target_map"),
            ],
            style=custom_style,
        ).ask()

        config["layer_targeting_mode"] = layer_targeting_choice
        config["use_adaptive_weighting"] = False
        config["layer_target_map_path"] = None
        config["per_layer_multipliers"] = None
        config["exclude_layers"] = None
        config["unmapped_layer_behavior"] = "skip"

        if layer_targeting_choice == "adaptive":
            if config["dynamic_layer_targeting"]:
                console.print(f"[{THEME['muted']}]  → Per-layer directions + Gaussian strength weighting[/{THEME['muted']}]")
            config["use_adaptive_weighting"] = True

        elif layer_targeting_choice == "target_map":
            config["layer_target_map_path"] = questionary.path(
                "Path to layer_target_map.json:",
                style=custom_style,
            ).ask()

            if config["layer_target_map_path"]:
                # Load and validate the target map
                from src.abliterate import load_layer_target_map
                try:
                    target_map = load_layer_target_map(config["layer_target_map_path"])

                    # Show summary to user
                    console.print(f"\n[{THEME['success']}]Loaded target map:[/{THEME['success']}]")
                    console.print(f"  - Target layers: {len(target_map.get('target_layer_indices', []))}")
                    console.print(f"  - Excluded layers: {len(target_map.get('exclude_layers', []))}")
                    console.print(f"  - Aggressive layers: {len(target_map['metadata'].get('aggressive_layers', []))}")
                    console.print(f"  - Protected layers: {len(target_map['metadata'].get('protected_layers', []))}")

                    config["per_layer_multipliers"] = target_map["per_layer_multipliers"]
                    config["exclude_layers"] = target_map.get("exclude_layers", [])

                    # Ask about unmapped layer behavior
                    config["unmapped_layer_behavior"] = questionary.select(
                        "How should layers NOT in the target map be handled?",
                        choices=[
                            questionary.Choice("Skip ablation (safest)", value="skip"),
                            questionary.Choice("Apply default multiplier (1.0)", value="default"),
                        ],
                        style=custom_style,
                    ).ask()

                except Exception as e:
                    display_error(f"Failed to load target map: {e}")
                    return None

        # Hybrid architecture options (uses hybrid_info detected at model selection)
        console.print(f"\n[bold {THEME['secondary']}]Hybrid Architecture (Qwen3.5, etc.)[/bold {THEME['secondary']}]")

        if hybrid_info.is_hybrid:
            config["hybrid_strategy"] = questionary.select(
                "Hybrid architecture strategy:",
                choices=[
                    questionary.Choice("Auto (recommended - architecture-aware extraction & weighting)", value="auto"),
                    questionary.Choice("Uniform (treat all layers the same, legacy behavior)", value="uniform"),
                ],
                default="auto",
                style=custom_style,
            ).ask()

            if config["hybrid_strategy"] == "auto":
                configure_hybrid = questionary.confirm(
                    "Configure hybrid weights? (defaults: full_attn=1.0, linear_attn=0.4)",
                    default=False,
                    style=custom_style,
                ).ask()

                if configure_hybrid:
                    config["hybrid_full_attn_weight"] = float(questionary.text(
                        "Full attention layer weight (0.0-2.0):",
                        default="1.0",
                        style=custom_style,
                    ).ask())

                    config["hybrid_linear_attn_weight"] = float(questionary.text(
                        "Linear attention layer weight (0.0-1.0):",
                        default="0.4",
                        style=custom_style,
                    ).ask())

                config["hybrid_skip_recurrent_proj"] = questionary.confirm(
                    "Skip recurrent dynamics projections (in_proj_a, in_proj_b)? (recommended)",
                    default=True,
                    style=custom_style,
                ).ask()

                config["hybrid_skip_state_proj"] = questionary.confirm(
                    "Also skip state projections (in_proj_qkv, in_proj_z)? (more conservative)",
                    default=False,
                    style=custom_style,
                ).ask()

                console.print(f"[{THEME['muted']}]  → Full attention layers weighted at {config['hybrid_full_attn_weight']}x[/{THEME['muted']}]")
                console.print(f"[{THEME['muted']}]  → Linear attention layers weighted at {config['hybrid_linear_attn_weight']}x[/{THEME['muted']}]")
                if config["hybrid_skip_recurrent_proj"]:
                    console.print(f"[{THEME['muted']}]  → Skipping in_proj_a, in_proj_b (recurrent dynamics)[/{THEME['muted']}]")
                if config["hybrid_skip_state_proj"]:
                    console.print(f"[{THEME['muted']}]  → Skipping in_proj_qkv, in_proj_z (state projections)[/{THEME['muted']}]")
        else:
            console.print(f"[{THEME['muted']}]No hybrid architecture detected (standard model)[/{THEME['muted']}]")

        # Biprojection options
        console.print(f"\n[bold {THEME['secondary']}]Biprojection (improved NatInt preservation)[/bold {THEME['secondary']}]")

        config["use_biprojection"] = questionary.confirm(
            "Enable biprojection mode? (measure at high-quality layers, apply broadly)",
            default=False,
            style=custom_style,
        ).ask()

        if config["use_biprojection"]:
            config["use_per_neuron_norm"] = questionary.confirm(
                "Use per-neuron norm preservation? (recommended for biprojection)",
                default=True,
                style=custom_style,
            ).ask()

            # Target layer types
            use_target_layers = questionary.confirm(
                "Target specific layer types? (Recommended: o_proj, down_proj only)",
                default=True,
                style=custom_style,
            ).ask()

            if use_target_layers:
                selected_layers = questionary.checkbox(
                    "Select layer types to target:",
                    choices=[
                        questionary.Choice("o_proj (attention output)", value="o_proj", checked=True),
                        questionary.Choice("down_proj (MLP output)", value="down_proj", checked=True),
                        questionary.Choice("q_proj (attention query)", value="q_proj"),
                        questionary.Choice("k_proj (attention key)", value="k_proj"),
                        questionary.Choice("v_proj (attention value)", value="v_proj"),
                        questionary.Choice("gate_proj (MLP gate)", value="gate_proj"),
                        questionary.Choice("up_proj (MLP up)", value="up_proj"),
                    ],
                    style=custom_style,
                ).ask()
                config["target_layer_types"] = selected_layers if selected_layers else None

            config["use_harmless_boundary"] = questionary.confirm(
                "Enable harmless direction boundary clamping? (prevents over-ablation)",
                default=True,
                style=custom_style,
            ).ask()

            if config["use_harmless_boundary"]:
                config["harmless_clamp_ratio"] = float(questionary.text(
                    "Harmless clamp ratio (0.0-1.0):",
                    default="0.1",
                    style=custom_style,
                ).ask())

            # Advanced biprojection settings
            configure_biprojection_advanced = questionary.confirm(
                "Configure advanced biprojection settings? (measurement layers, intervention range)",
                default=False,
                style=custom_style,
            ).ask()

            if configure_biprojection_advanced:
                config["num_measurement_layers"] = int(questionary.text(
                    "Number of measurement layers (top quality):",
                    default="2",
                    style=custom_style,
                ).ask())

                intervention_start = float(questionary.text(
                    "Intervention range start (0.0-1.0, fraction of depth):",
                    default="0.25",
                    style=custom_style,
                ).ask())

                intervention_end = float(questionary.text(
                    "Intervention range end (0.0-1.0, fraction of depth):",
                    default="0.95",
                    style=custom_style,
                ).ask())

                config["intervention_range"] = (intervention_start, intervention_end)
        else:
            # Allow per-neuron norm even without full biprojection
            config["use_per_neuron_norm"] = questionary.confirm(
                "Use per-neuron norm preservation?",
                default=False,
                style=custom_style,
            ).ask()

        # Gabliteration options
        console.print(f"\n[bold {THEME['secondary']}]Gabliteration (multi-directional SVD abliteration)[/bold {THEME['secondary']}]")

        config["use_gabliteration"] = questionary.confirm(
            "Enable Gabliteration? (multi-directional SVD for more thorough refusal removal)",
            default=False,
            style=custom_style,
        ).ask()

        if config["use_gabliteration"]:
            config["gab_num_directions"] = int(questionary.text(
                "Number of SVD directions (1-10, paper recommends 1-3):",
                default="2",
                style=custom_style,
            ).ask())

            config["gab_ridge_lambda"] = float(questionary.text(
                "Ridge regularization lambda (0.0-1.0):",
                default="0.1",
                style=custom_style,
            ).ask())

            configure_gab_advanced = questionary.confirm(
                "Configure advanced Gabliteration settings? (layer scaling, skip layers)",
                default=False,
                style=custom_style,
            ).ask()

            if configure_gab_advanced:
                config["gab_layer_scaling_beta"] = float(questionary.text(
                    "Layer scaling curvature beta (0=uniform, 0.5=moderate, 1.0=peaked):",
                    default="0.5",
                    style=custom_style,
                ).ask())

                config["gab_skip_first_layers"] = int(questionary.text(
                    "Skip first N layers:",
                    default="2",
                    style=custom_style,
                ).ask())

                config["gab_skip_last_layers"] = int(questionary.text(
                    "Skip last N layers:",
                    default="2",
                    style=custom_style,
                ).ask())

        # Numerical stability options
        console.print(f"\n[bold {THEME['secondary']}]Numerical Stability[/bold {THEME['secondary']}]")

        configure_stability = questionary.confirm(
            "Configure numerical stability options? (defaults are recommended)",
            default=False,
            style=custom_style,
        ).ask()

        if configure_stability:
            config["use_projected_refusal"] = questionary.confirm(
                "Orthogonalize refusal against harmless direction? (recommended)",
                default=True,
                style=custom_style,
            ).ask()

            config["use_welford_mean"] = questionary.confirm(
                "Use Welford's algorithm for streaming mean? (more numerically stable)",
                default=True,
                style=custom_style,
            ).ask()

            config["use_float64_subtraction"] = questionary.confirm(
                "Use float64 for mean subtraction? (handles high cosine similarity)",
                default=True,
                style=custom_style,
            ).ask()

        # KL divergence monitoring
        console.print(f"\n[bold {THEME['secondary']}]KL Divergence Monitoring (capability drift)[/bold {THEME['secondary']}]")

        kl_mode = questionary.select(
            "KL divergence monitoring:",
            choices=[
                questionary.Choice("None (skip KL monitoring)", value="none"),
                questionary.Choice("Monitor only (report KL after abliteration)", value="monitor"),
                questionary.Choice("Auto-tune (binary search for best multiplier within KL budget)", value="auto_tune"),
            ],
            style=custom_style,
        ).ask()

        if kl_mode == "monitor":
            config["use_kl_monitoring"] = True
        elif kl_mode == "auto_tune":
            config["use_kl_monitoring"] = True
            config["use_kl_auto_tune"] = True

        if kl_mode in ("monitor", "auto_tune"):
            use_default_kl_prompts = questionary.confirm(
                "Use default reference prompts for KL monitoring?",
                default=True,
                style=custom_style,
            ).ask()

            if not use_default_kl_prompts:
                config["kl_reference_prompts_path"] = questionary.path(
                    "Path to custom KL reference prompts:",
                    style=custom_style,
                ).ask()

            config["kl_num_reference_prompts"] = int(questionary.text(
                "Number of reference prompts:",
                default="50",
                style=custom_style,
            ).ask())

            configure_kl_advanced = questionary.confirm(
                "Configure advanced KL settings?",
                default=False,
                style=custom_style,
            ).ask()

            if configure_kl_advanced:
                config["kl_top_k"] = int(questionary.text(
                    "Top-k tokens for KL approximation:",
                    default="200",
                    style=custom_style,
                ).ask())

            if kl_mode == "auto_tune":
                config["kl_threshold"] = float(questionary.text(
                    "Max mean KL divergence (nats) for auto-tune:",
                    default="0.5",
                    style=custom_style,
                ).ask())

                configure_search = questionary.confirm(
                    "Configure auto-tune search range?",
                    default=False,
                    style=custom_style,
                ).ask()

                if configure_search:
                    config["kl_search_min"] = float(questionary.text(
                        "Search range minimum multiplier:",
                        default="0.1",
                        style=custom_style,
                    ).ask())

                    config["kl_search_max"] = float(questionary.text(
                        "Search range maximum multiplier:",
                        default="2.0",
                        style=custom_style,
                    ).ask())

            console.print(f"\n[{THEME['muted']}]KL monitoring: {'auto-tune' if kl_mode == 'auto_tune' else 'report only'}[/{THEME['muted']}]")
            console.print(f"[{THEME['muted']}]  → {config['kl_num_reference_prompts']} reference prompts[/{THEME['muted']}]")
            if kl_mode == "auto_tune":
                console.print(f"[{THEME['muted']}]  → Threshold: {config['kl_threshold']} nats, range [{config['kl_search_min']}, {config['kl_search_max']}][/{THEME['muted']}]")

    return config


def _display_mined_concepts(mined: list, mode: str = "consensus") -> None:
    """Render mined refusal concepts as a Rich table for review.

    In consensus mode, MinedConcept fields are reused as:
      specificity -> score (coverage * mean_prob)
      p_harmful   -> mean_prob across harmful prompts
      p_harmless  -> coverage fraction across harmful prompts
    """
    from rich.table import Table
    if mode == "consensus":
        title = "Mined refusal concepts (consensus mode)"
        score_col = "Score"
        col_a, col_b = "Mean P(harmful)", "Coverage"
    else:
        title = "Mined refusal concepts (contrast mode)"
        score_col = "Specificity"
        col_a, col_b = "P(harmful)", "P(harmless)"
    table = Table(title=title, show_lines=False)
    table.add_column("#", style=THEME["muted"], justify="right")
    table.add_column("Concept", style=THEME["accent"])
    table.add_column("Token ID", style=THEME["muted"], justify="right")
    table.add_column(score_col, style=THEME["primary"], justify="right")
    table.add_column(col_a, justify="right")
    table.add_column(col_b, justify="right")
    for i, m in enumerate(mined, 1):
        table.add_row(
            str(i),
            repr(m.concept),
            str(m.token_id),
            f"{m.specificity:.4f}",
            f"{m.p_harmful:.4f}",
            f"{m.p_harmless:.4f}",
        )
    console.print(table)


def preview_and_edit_mined_concepts(
    model_path: str,
    device: str,
    dtype: str,
    mining_mode: str,
    mine_top_k: int,
    mine_min_specificity: float,
    mine_num_positions: int,
    mine_num_prompts: int,
    batch_size: int,
    max_seq_len: int,
) -> tuple[Optional[list[str]], Optional[dict[str, int]], bool]:
    """Mine refusal concepts, display them, and let the user deselect any before use.

    Loads the model temporarily just to mine, then unloads it. This means the
    subsequent abliteration pass reloads the model — the tradeoff is that the
    user gets to review and edit the consensus list before committing.

    Returns (concepts, concept_token_ids, ok). ok=False means the user aborted
    the whole workflow. If ok=True and concepts is None, mining yielded nothing
    or the user cleared the list; the caller should fall back to defaults.
    """
    from src.abliterate import load_prompts_from_file
    from src.jlens import mine_refusal_concepts
    from src.model_utils import load_model_and_tokenizer

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(dtype, torch.float16)

    console.print(
        f"\n[{THEME['primary']}]Loading model to mine consensus refusal concepts "
        f"(preview step — model will reload for abliteration)...[/{THEME['primary']}]"
    )
    try:
        model, tokenizer = load_model_and_tokenizer(
            model_path, device=device, dtype=torch_dtype, trust_remote_code=True
        )
        tokenizer.padding_side = "left"
    except Exception as e:
        console.print(f"[red]Model load for mining failed: {e}[/red]")
        return None, None, False

    mined: list = []
    try:
        harmful_for_mining = load_prompts_from_file(
            get_default_prompts_path("harmful.txt"), num_prompts=mine_num_prompts
        )
        harmless_for_mining = None
        if mining_mode == "contrast":
            harmless_for_mining = load_prompts_from_file(
                get_default_prompts_path("harmless.txt"), num_prompts=mine_num_prompts
            )
        console.print(
            f"[{THEME['primary']}]Mining refusal concepts (mode={mining_mode}, "
            f"top_k={mine_top_k}, min_score={mine_min_specificity})...[/{THEME['primary']}]"
        )
        mined = mine_refusal_concepts(
            model, tokenizer,
            harmful_prompts=harmful_for_mining,
            harmless_prompts=harmless_for_mining,
            top_k=mine_top_k,
            min_specificity=mine_min_specificity,
            batch_size=max(batch_size, 4),
            max_seq_len=max(max_seq_len, 128),
            device=device,
            mode=mining_mode,
            num_positions=mine_num_positions,
        )
    except Exception as e:
        console.print(f"[{THEME['warning']}]Mining failed: {e}[/{THEME['warning']}]")
        mined = []
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not mined:
        console.print(
            f"[{THEME['warning']}]Mining yielded no concepts; abliteration will "
            f"fall back to defaults.[/{THEME['warning']}]"
        )
        return None, None, True

    _display_mined_concepts(mined, mode=mining_mode)

    choices = [
        questionary.Choice(
            title=f"{repr(m.concept)}  (score={m.specificity:.4f}, token_id={m.token_id})",
            value=m.concept,
            checked=True,
        )
        for m in mined
    ]
    kept = questionary.checkbox(
        "Select the concepts to KEEP (space to toggle, enter to confirm):",
        choices=choices,
        style=custom_style,
    ).ask()

    if kept is None:
        console.print(f"[{THEME['warning']}]Concept selection cancelled; aborting workflow.[/{THEME['warning']}]")
        return None, None, False

    if not kept:
        console.print(
            f"[{THEME['warning']}]No concepts selected; abliteration will "
            f"fall back to defaults.[/{THEME['warning']}]"
        )
        return None, None, True

    kept_set = set(kept)
    filtered = [m for m in mined if m.concept in kept_set]
    concepts = [m.concept for m in filtered]
    token_ids = {m.concept: m.token_id for m in filtered}
    dropped = [m.concept for m in mined if m.concept not in kept_set]
    if dropped:
        console.print(
            f"[{THEME['muted']}]Dropped {len(dropped)} concept(s): "
            f"{', '.join(repr(c) for c in dropped)}[/{THEME['muted']}]"
        )
    console.print(
        f"[{THEME['success']}]Using {len(concepts)} concept(s) for J-space "
        f"restriction: {', '.join(repr(c) for c in concepts)}[/{THEME['success']}]"
    )
    return concepts, token_ids, True


def run_jlens_map_generation(
    model_path: str,
    output_path: str,
    concepts: Optional[list[str]] = None,
    num_prompts: int = 32,
    batch_size: int = 2,
    max_seq_len: int = 64,
    grad_checkpoint: bool = False,
    basis_rank: Optional[int] = 16,
    exclude_threshold: float = 0.2,
    min_multiplier: float = 0.1,
    device: str = "cuda",
    dtype: str = "float16",
    mine_concepts: bool = False,
    mine_top_k: int = 8,
    mine_min_specificity: float = 0.005,
    mine_num_prompts: int = 64,
    mining_mode: str = "consensus",
    mine_num_positions: int = 4,
) -> bool:
    """Compute J-lens vectors for a model and write out layer_target_map.json.

    Uses harmful prompts from the standard prompts directory. Writes two files:
      - <output_path>/layer_target_map.json  (feeds --layer-target-map)
      - <output_path>/jlens_vectors.pt        (cache for --jlens-vectors)

    If `mine_concepts` is True, the refusal concepts are discovered empirically
    via contrastive next-token probs (harmful vs harmless) before running J-lens.
    """
    from src.abliterate import (
        get_transformer_layers,
        load_prompts_from_file,
        validate_layer_target_map,
    )
    from src.jlens import (
        DEFAULT_REFUSAL_CONCEPTS,
        JLensConfig,
        build_jspace_config_dict,
        compute_jlens_vectors,
        generate_layer_target_map,
        mine_refusal_concepts,
        save_jspace_config_json,
        save_target_map_json,
    )
    from src.model_utils import load_model_and_tokenizer

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(dtype, torch.float16)

    try:
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"[{THEME['primary']}]Loading harmful prompts...[/{THEME['primary']}]")
        prompts = load_prompts_from_file(
            get_default_prompts_path("harmful.txt"),
            num_prompts=max(num_prompts, mine_num_prompts if mine_concepts else num_prompts),
        )

        console.print(f"[{THEME['primary']}]Loading model {model_path}...[/{THEME['primary']}]")
        model, tokenizer = load_model_and_tokenizer(
            model_path, device=device, dtype=torch_dtype, trust_remote_code=True
        )
        tokenizer.padding_side = "left"

        concept_token_ids: Optional[dict[str, int]] = None
        mined_records: list = []
        if mine_concepts:
            console.print(
                f"[{THEME['primary']}]Mining refusal concepts "
                f"(mode={mining_mode})...[/{THEME['primary']}]"
            )
            harmful_for_mining = prompts[:mine_num_prompts]
            harmless_for_mining = None
            if mining_mode == "contrast":
                harmless_for_mining = load_prompts_from_file(
                    get_default_prompts_path("harmless.txt"), num_prompts=mine_num_prompts
                )
            mined = mine_refusal_concepts(
                model, tokenizer,
                harmful_prompts=harmful_for_mining,
                harmless_prompts=harmless_for_mining,
                top_k=mine_top_k,
                min_specificity=mine_min_specificity,
                batch_size=max(batch_size, 4),
                max_seq_len=max_seq_len,
                device=device,
                mode=mining_mode,
                num_positions=mine_num_positions,
            )
            if mined:
                concepts = [m.concept for m in mined]
                concept_token_ids = {m.concept: m.token_id for m in mined}
                mined_records = [
                    {"concept": m.concept, "token_id": m.token_id,
                     "score": m.specificity, "mean_prob": m.p_harmful,
                     "coverage": m.p_harmless}
                    for m in mined
                ]
                _display_mined_concepts(mined, mode=mining_mode)
            else:
                console.print(
                    f"[{THEME['warning']}]Mining yielded no concepts; "
                    f"falling back to defaults[/{THEME['warning']}]"
                )

        jlens_config = JLensConfig(
            concepts=list(concepts) if concepts else list(DEFAULT_REFUSAL_CONCEPTS),
            num_prompts=num_prompts,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            gradient_checkpointing=grad_checkpoint,
            device=device,
            dtype=torch_dtype,
            basis_rank=basis_rank,
            exclude_threshold=exclude_threshold,
            min_multiplier=min_multiplier,
            concept_token_ids=concept_token_ids,
        )

        # Use the (possibly-truncated) jlens prompt slice, not the mining slice
        jlens_prompts = prompts[:num_prompts]

        console.print(
            f"[{THEME['primary']}]Computing J-lens vectors "
            f"({len(jlens_config.concepts)} concepts, {len(jlens_prompts)} prompts)...[/{THEME['primary']}]"
        )
        jlens = compute_jlens_vectors(model, tokenizer, jlens_prompts, jlens_config)

        num_layers = len(get_transformer_layers(model))
        target_map = generate_layer_target_map(
            jlens,
            num_layers=num_layers,
            exclude_threshold=exclude_threshold,
            min_multiplier=min_multiplier,
            model_path=model_path,
        )

        # Validate against loader schema
        from src.abliterate import load_layer_target_map as _load_map  # for round-trip
        map_path = str(out_dir / "layer_target_map.json")
        save_target_map_json(target_map, map_path)
        parsed = _load_map(map_path)
        warnings = validate_layer_target_map(parsed, num_layers)
        for w in warnings:
            console.print(f"[{THEME['warning']}]{w}[/{THEME['warning']}]")

        vectors_path = str(out_dir / "jlens_vectors.pt")
        jlens.save(vectors_path)

        # Assemble and write J-space abliteration_config.json
        multipliers = target_map.get("layer_multipliers", {})
        excluded = target_map.get("excluded_layers", [])
        aggressive = target_map.get("aggressive_layers", [])
        concept_source = (
            "auto-mine" if mine_concepts
            else ("manual" if concepts else "defaults")
        )
        payload = build_jspace_config_dict(
            mode="map",
            model_path=model_path,
            output_path=str(out_dir),
            device=device,
            dtype=dtype,
            jlens_params={
                "num_prompts": num_prompts,
                "batch_size": batch_size,
                "max_seq_len": max_seq_len,
                "grad_checkpoint": grad_checkpoint,
                "basis_rank": basis_rank,
                "vectors_path": vectors_path,
            },
            concepts_params={
                "source": concept_source,
                "concepts": jlens_config.concepts,
                "concept_token_ids": concept_token_ids,
                "mining_mode": mining_mode if mine_concepts else None,
                "mining_top_k": mine_top_k if mine_concepts else None,
                "mining_min_score": mine_min_specificity if mine_concepts else None,
                "mining_num_positions": mine_num_positions if mine_concepts else None,
                "mining_num_prompts": mine_num_prompts if mine_concepts else None,
                "mined_concepts": mined_records if mine_concepts else None,
            },
            target_map_params={
                "map_path": map_path,
                "exclude_threshold": exclude_threshold,
                "min_multiplier": min_multiplier,
                "num_layers_total": num_layers,
                "num_active_layers": len(multipliers),
                "num_excluded_layers": len(excluded),
                "num_aggressive_layers": len(aggressive),
            },
        )
        save_jspace_config_json(payload, str(out_dir))

        # Free the model before returning
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Summary
        console.print(
            f"[{THEME['success']}]J-lens map generated: "
            f"{len(multipliers)} active layers, {len(excluded)} excluded[/{THEME['success']}]"
        )
        console.print(f"[{THEME['muted']}]  → {map_path}[/{THEME['muted']}]")
        console.print(f"[{THEME['muted']}]  → {vectors_path}[/{THEME['muted']}]")
        console.print(f"[{THEME['muted']}]  → {out_dir / 'abliteration_config.json'}[/{THEME['muted']}]")
        return True
    except Exception as e:
        console.print(f"[red]J-lens map generation failed: {e}[/red]")
        import traceback
        console.print(f"[{THEME['muted']}]{traceback.format_exc()}[/{THEME['muted']}]")
        return False


def run_jlens_iterative_mode(
    model_path: str,
    output_path: str,
    device: str = "cuda",
    dtype: str = "float16",
    max_iterations: int = 10,
    target_refusal_rate: float = 0.05,
    step_multiplier: float = 1.0,
    max_step_multiplier: float = 1.0,
    step_escalation: bool = True,
    eval_prompt_count: int = 50,
    track_kl: bool = True,
    use_kl_guardrail: bool = False,
    kl_threshold: float = 0.5,
    kl_reference_prompts: Optional[str] = None,
    concepts: Optional[list[str]] = None,
    abliteration_num_prompts: int = 32,
    filter_prompts: bool = True,
    refusal_threshold: float = -7.0,
    auto_calibrate_threshold: bool = False,
    num_prompts: int = 32,
    batch_size: int = 2,
    max_seq_len: int = 64,
    grad_checkpoint: bool = False,
    basis_rank: int = 16,
    min_projection_ratio: float = 0.1,
    mine_concepts: bool = False,
    mine_top_k: int = 8,
    mine_min_specificity: float = 0.005,
    mining_mode: str = "consensus",
    mine_num_positions: int = 4,
) -> bool:
    """Run J-lens-guided iterative abliteration. Writes model + iteration_history.json."""
    from src.jlens import DEFAULT_REFUSAL_CONCEPTS
    from src.jlens_iterative import JLensIterativeConfig, run_jlens_iterative_abliteration

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(dtype, torch.float16)

    try:
        cfg = JLensIterativeConfig(
            max_iterations=max_iterations,
            target_refusal_rate=target_refusal_rate,
            step_multiplier=step_multiplier,
            max_step_multiplier=max_step_multiplier,
            step_escalation=step_escalation,
            eval_prompt_count=eval_prompt_count,
            track_kl=track_kl,
            use_kl_guardrail=use_kl_guardrail,
            kl_threshold=kl_threshold,
            kl_reference_prompts_path=kl_reference_prompts,
            abliteration_num_prompts=abliteration_num_prompts,
            filter_prompts=filter_prompts,
            refusal_threshold=refusal_threshold,
            auto_calibrate_threshold=auto_calibrate_threshold,
            jlens_concepts=(list(concepts) if concepts else list(DEFAULT_REFUSAL_CONCEPTS)),
            jlens_num_prompts=num_prompts,
            jlens_batch_size=batch_size,
            jlens_max_seq_len=max_seq_len,
            jlens_grad_checkpoint=grad_checkpoint,
            jlens_basis_rank=basis_rank,
            jlens_min_projection_ratio=min_projection_ratio,
            mine_concepts_each_iteration=mine_concepts,
            mine_top_k=mine_top_k,
            mine_min_specificity=mine_min_specificity,
            mining_mode=mining_mode,
            mine_num_positions=mine_num_positions,
            device=device,
            dtype=torch_dtype,
        )
        result = run_jlens_iterative_abliteration(
            model_path=model_path,
            output_path=output_path,
            harmful_prompts_path=get_default_prompts_path("harmful.txt"),
            harmless_prompts_path=get_default_prompts_path("harmless.txt"),
            config=cfg,
        )
        status_color = THEME["success"] if result.converged else THEME["warning"]
        console.print(
            f"[{status_color}]J-lens iterative done: converged={result.converged} "
            f"({result.reason}), refusal_rate {result.baseline_refusal_rate:.3f} -> "
            f"{result.final_refusal_rate:.3f} in {result.total_iterations} iterations[/{status_color}]"
        )
        console.print(f"[{THEME['muted']}]  → {output_path}[/{THEME['muted']}]")
        return result.converged
    except Exception as e:
        console.print(f"[red]J-lens iterative failed: {e}[/red]")
        import traceback
        console.print(f"[{THEME['muted']}]{traceback.format_exc()}[/{THEME['muted']}]")
        return False


def run_jspace_workflow() -> None:
    """Interactive entry point for J-space (Jacobian Lens) abliteration.

    Lets the user pick between the three J-lens modes:
      1. Layer target map generation (--jlens-map)
      2. J-space-restricted single-pass abliteration (--jlens-restrict)
      3. J-lens-guided iterative abliteration (--jlens-iterative)
    """
    console.print(f"\n[bold {THEME['primary']}]J-space Abliteration (Jacobian Lens)[/bold {THEME['primary']}]\n")
    console.print(
        f"[{THEME['muted']}]The Jacobian Lens computes per-layer refusal 'workspace' vectors "
        f"via VJPs of final logits.\nThree modes are available.[/{THEME['muted']}]\n"
    )

    mode = questionary.select(
        "Select J-space mode:",
        choices=[
            questionary.Choice(
                title="Layer target map — compute J-lens signal and write layer_target_map.json (cheapest)",
                value="map",
            ),
            questionary.Choice(
                title="Restricted abliteration — single-pass abliteration with J-space-restricted refusal direction",
                value="restrict",
            ),
            questionary.Choice(
                title="Iterative abliteration — recompute J-lens each iteration until target refusal rate",
                value="iterative",
            ),
            questionary.Choice(title="Back", value="__back__"),
        ],
        style=custom_style,
    ).ask()

    if mode is None or mode == "__back__":
        return

    # Common: model + output
    console.print(f"\n[bold {THEME['primary']}]Step 1: Select Base Model[/bold {THEME['primary']}]\n")
    model_path = select_model("Select the model to analyze / abliterate:")
    if not model_path:
        return

    console.print(f"\n[bold {THEME['primary']}]Step 2: Output Path[/bold {THEME['primary']}]\n")
    output_dir = get_default_output_dir()
    if mode == "map":
        default_output = f"{output_dir}/{Path(model_path).name}-jlens-map"
        output_prompt = "Enter output directory (map + vectors will be written here):"
    elif mode == "restrict":
        default_output = f"{output_dir}/{Path(model_path).name}-jlens-restricted"
        output_prompt = "Enter output path for the abliterated model:"
    else:
        default_output = f"{output_dir}/{Path(model_path).name}-jlens-iterative"
        output_prompt = "Enter output path for the abliterated model:"

    use_default = questionary.confirm(
        f"Use default output path? ({default_output})",
        default=True,
        style=custom_style,
    ).ask()
    output_path = default_output if use_default else questionary.path(
        output_prompt, default=default_output, style=custom_style,
    ).ask()
    if not output_path:
        return

    # Device / dtype
    if torch.cuda.is_available():
        device = questionary.select(
            "Select device:",
            choices=["cuda", "cpu"],
            default="cuda",
            style=custom_style,
        ).ask()
    else:
        device = "cpu"
        console.print(f"[{THEME['warning']}]CUDA not available, using CPU[/{THEME['warning']}]")

    dtype = questionary.select(
        "Select precision:",
        choices=[
            questionary.Choice("float16 (faster, less memory)", value="float16"),
            questionary.Choice("bfloat16 (better precision)", value="bfloat16"),
            questionary.Choice("float32 (full precision)", value="float32"),
        ],
        default=get_default_dtype(),
        style=custom_style,
    ).ask()

    # Shared J-lens extraction params
    console.print(f"\n[bold {THEME['primary']}]Step 3: J-lens Extraction[/bold {THEME['primary']}]\n")

    concept_source = questionary.select(
        "Where should refusal concepts come from?",
        choices=[
            questionary.Choice(
                title="Auto-mine from this model's responses (recommended)",
                value="mine",
            ),
            questionary.Choice(
                title="Use built-in defaults (' cannot', ' sorry', ...)",
                value="defaults",
            ),
            questionary.Choice(
                title="Enter manually (comma-separated)",
                value="manual",
            ),
        ],
        default="mine",
        style=custom_style,
    ).ask()

    concepts: Optional[list[str]] = None
    mine_concepts_flag = False
    mine_top_k = 8
    mine_min_specificity = 0.005
    mining_mode = "consensus"
    mine_num_positions = 4
    if concept_source == "mine":
        mine_concepts_flag = True
        mining_mode = questionary.select(
            "Mining strategy:",
            choices=[
                questionary.Choice(
                    title="Consensus — tokens the model uses across positions 0..N (recommended)",
                    value="consensus",
                ),
                questionary.Choice(
                    title="Contrast — P(t|harmful) - P(t|harmless)",
                    value="contrast",
                ),
            ],
            default="consensus",
            style=custom_style,
        ).ask()
        if mining_mode == "consensus":
            mine_num_positions = int(questionary.text(
                "Number of response positions to walk (1=first token only, 4=captures 'I cannot help with'):",
                default="4",
                style=custom_style,
            ).ask())
        mine_top_k = int(questionary.text(
            "Top-k concepts to keep from mining:",
            default="8",
            style=custom_style,
        ).ask())
        default_min_score = "0.005" if mining_mode == "contrast" else "0.001"
        min_score_label = (
            "Min score (coverage * mean-prob):" if mining_mode == "consensus"
            else "Min specificity (P(t|harmful)-P(t|harmless)):"
        )
        mine_min_specificity = float(questionary.text(
            min_score_label,
            default=default_min_score,
            style=custom_style,
        ).ask())
    elif concept_source == "manual":
        concepts_raw = questionary.text(
            "Refusal concepts (comma-separated):",
            default="",
            style=custom_style,
        ).ask()
        concepts = (
            [c.strip() for c in concepts_raw.split(",") if c.strip()]
            if concepts_raw else None
        )
    # else: defaults — leave concepts=None, mine_concepts_flag=False

    num_prompts = int(questionary.text(
        "Number of harmful prompts to average over:",
        default="32",
        style=custom_style,
    ).ask())
    batch_size = int(questionary.text(
        "Batch size (small for VRAM):",
        default="2",
        style=custom_style,
    ).ask())
    max_seq_len = int(questionary.text(
        "Max sequence length per J-lens pass:",
        default="64",
        style=custom_style,
    ).ask())
    grad_checkpoint = questionary.confirm(
        "Enable gradient checkpointing? (slower but lower VRAM)",
        default=False,
        style=custom_style,
    ).ask()

    # Mode-specific configuration + dispatch
    if mode == "map":
        console.print(f"\n[bold {THEME['primary']}]Step 4: Target Map Options[/bold {THEME['primary']}]\n")
        basis_rank = int(questionary.text(
            "J-space basis rank (max):",
            default="16",
            style=custom_style,
        ).ask())
        exclude_threshold = float(questionary.text(
            "Exclude layers below this normalized signal:",
            default="0.2",
            style=custom_style,
        ).ask())
        min_multiplier = float(questionary.text(
            "Minimum per-layer multiplier:",
            default="0.1",
            style=custom_style,
        ).ask())

        concept_source_label = (
            f"auto-mine ({mining_mode}, top {mine_top_k}, min={mine_min_specificity})"
            if mine_concepts_flag else (concepts or "default")
        )
        summary = {
            "mode": "jlens-map",
            "model_path": model_path,
            "output_path": output_path,
            "device": device,
            "dtype": dtype,
            "num_prompts": num_prompts,
            "batch_size": batch_size,
            "max_seq_len": max_seq_len,
            "grad_checkpoint": grad_checkpoint,
            "basis_rank": basis_rank,
            "exclude_threshold": exclude_threshold,
            "min_multiplier": min_multiplier,
            "concepts": concept_source_label,
        }
        display_config_panel(summary, "J-lens Map Generation")
        if not questionary.confirm("\nProceed?", default=True, style=custom_style).ask():
            return
        run_jlens_map_generation(
            model_path=model_path,
            output_path=output_path,
            concepts=concepts,
            num_prompts=num_prompts,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            grad_checkpoint=grad_checkpoint,
            basis_rank=basis_rank,
            exclude_threshold=exclude_threshold,
            min_multiplier=min_multiplier,
            device=device,
            dtype=dtype,
            mine_concepts=mine_concepts_flag,
            mine_top_k=mine_top_k,
            mine_min_specificity=mine_min_specificity,
            mining_mode=mining_mode,
            mine_num_positions=mine_num_positions,
        )

    elif mode == "restrict":
        console.print(f"\n[bold {THEME['primary']}]Step 4: Restriction Options[/bold {THEME['primary']}]\n")
        basis_rank = int(questionary.text(
            "J-space basis rank (max):",
            default="16",
            style=custom_style,
        ).ask())
        min_projection_ratio = float(questionary.text(
            "Min retained projection ratio (fallback if below):",
            default="0.1",
            style=custom_style,
        ).ask())

        reuse_cached = questionary.confirm(
            "Reuse a cached jlens_vectors.pt? (otherwise computed inline)",
            default=False,
            style=custom_style,
        ).ask()
        cached_path = None
        if reuse_cached:
            cached_path = questionary.path(
                "Path to jlens_vectors.pt:",
                style=custom_style,
            ).ask() or None

        # Base abliteration parameters
        abl_num_prompts = int(questionary.text(
            "Number of prompts for base abliteration:",
            default=str(get_default_num_prompts()),
            style=custom_style,
        ).ask())
        direction_multiplier = float(questionary.text(
            "Direction multiplier (ablation strength):",
            default=str(get_default_direction_multiplier()),
            style=custom_style,
        ).ask())
        use_winsorize = questionary.confirm(
            "Enable Winsorization? (recommended for Gemma)",
            default=False,
            style=custom_style,
        ).ask()

        # Interactive consensus-concept review: mine now, let the user deselect,
        # then pass the filtered list as explicit concepts (mining flag off).
        # Skipping this when the user chose "defaults" or "manual" earlier.
        effective_concepts = concepts
        effective_token_ids: Optional[dict[str, int]] = None
        effective_mine_flag = mine_concepts_flag
        if mine_concepts_flag and not cached_path:
            preview_num_prompts = max(num_prompts, 64)
            edited_concepts, edited_token_ids, ok = preview_and_edit_mined_concepts(
                model_path=model_path,
                device=device,
                dtype=dtype,
                mining_mode=mining_mode,
                mine_top_k=mine_top_k,
                mine_min_specificity=mine_min_specificity,
                mine_num_positions=mine_num_positions,
                mine_num_prompts=preview_num_prompts,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
            )
            if not ok:
                return
            # If the user kept concepts, disable in-pipeline mining so we don't
            # redo the work. If mining yielded nothing, fall through to defaults
            # (edited_concepts=None, mine flag stays off — pipeline uses DEFAULT_REFUSAL_CONCEPTS).
            effective_concepts = edited_concepts
            effective_token_ids = edited_token_ids
            effective_mine_flag = False

        # Build abliteration config and dispatch to run_abliteration
        abl_config = dict(get_default_settings())
        abl_config.update({
            "model_path": model_path,
            "output_path": output_path,
            "device": device,
            "dtype": dtype,
            "num_prompts": abl_num_prompts,
            "direction_multiplier": direction_multiplier,
            "use_winsorization": use_winsorize,
            # J-lens restriction
            "use_jlens_restriction": True,
            "jlens_vectors_path": cached_path,
            "jlens_basis_rank": basis_rank,
            "jlens_min_projection_ratio": min_projection_ratio,
            "jlens_num_prompts": num_prompts,
            "jlens_batch_size": batch_size,
            "jlens_max_seq_len": max_seq_len,
            "jlens_grad_checkpoint": grad_checkpoint,
            "jlens_concepts": effective_concepts,
            "jlens_concept_token_ids": effective_token_ids,
            "mine_jlens_concepts": effective_mine_flag,
            "mine_top_k": mine_top_k,
            "mine_min_specificity": mine_min_specificity,
            "mining_mode": mining_mode,
            "mine_num_positions": mine_num_positions,
        })

        if effective_concepts:
            concept_source_label = (
                f"user-edited consensus ({len(effective_concepts)} concept(s)): "
                f"{', '.join(repr(c) for c in effective_concepts)}"
            )
        elif effective_mine_flag:
            concept_source_label = (
                f"auto-mine ({mining_mode}, top {mine_top_k}, min={mine_min_specificity})"
            )
        else:
            concept_source_label = concepts or "default"
        display_config_panel(
            {
                "mode": "jlens-restrict",
                "model_path": model_path,
                "output_path": output_path,
                "device": device,
                "dtype": dtype,
                "num_prompts": abl_num_prompts,
                "direction_multiplier": direction_multiplier,
                "use_winsorization": use_winsorize,
                "jlens_vectors_path": cached_path or "compute inline",
                "jlens_basis_rank": basis_rank,
                "jlens_min_projection_ratio": min_projection_ratio,
                "jlens_num_prompts": num_prompts,
                "jlens_batch_size": batch_size,
                "jlens_max_seq_len": max_seq_len,
                "jlens_grad_checkpoint": grad_checkpoint,
                "jlens_concepts": concept_source_label,
            },
            "J-space Restricted Abliteration",
        )
        if not questionary.confirm("\nProceed?", default=True, style=custom_style).ask():
            return
        run_abliteration(abl_config)

        # Overwrite the default abliteration_config.json with a J-space-specific one.
        try:
            from src.jlens import build_jspace_config_dict as _build, save_jspace_config_json as _save
            if effective_concepts and mine_concepts_flag:
                concept_source = "auto-mine-edited"
            elif effective_mine_flag:
                concept_source = "auto-mine"
            elif concepts:
                concept_source = "manual"
            else:
                concept_source = "defaults"
            payload = _build(
                mode="restrict",
                model_path=model_path,
                output_path=output_path,
                device=device,
                dtype=dtype,
                jlens_params={
                    "num_prompts": num_prompts,
                    "batch_size": batch_size,
                    "max_seq_len": max_seq_len,
                    "grad_checkpoint": grad_checkpoint,
                    "basis_rank": basis_rank,
                    "min_projection_ratio": min_projection_ratio,
                    "cached_vectors_path": cached_path,
                },
                concepts_params={
                    "source": concept_source,
                    "concepts": effective_concepts if effective_concepts else concepts,
                    "concept_token_ids": effective_token_ids,
                    "mining_mode": mining_mode if mine_concepts_flag else None,
                    "mining_top_k": mine_top_k if mine_concepts_flag else None,
                    "mining_min_score": mine_min_specificity if mine_concepts_flag else None,
                    "mining_num_positions": mine_num_positions if mine_concepts_flag else None,
                },
                abliteration_params={
                    "direction_multiplier": direction_multiplier,
                    "num_prompts": abl_num_prompts,
                    "use_winsorization": use_winsorize,
                },
            )
            _save(payload, output_path)
        except Exception as e:
            console.print(f"[{THEME['warning']}]Could not write J-space config: {e}[/{THEME['warning']}]")

    else:  # iterative
        console.print(f"\n[bold {THEME['primary']}]Step 4: Iterative Loop Options[/bold {THEME['primary']}]\n")
        max_iterations = int(questionary.text(
            "Max iterations:",
            default="10",
            style=custom_style,
        ).ask())
        target_refusal_rate = float(questionary.text(
            "Target refusal rate (convergence):",
            default="0.05",
            style=custom_style,
        ).ask())
        step_multiplier = float(questionary.text(
            "Per-iteration ablation strength (initial step, 1.0 = matches restrict mode):",
            default="1.0",
            style=custom_style,
        ).ask())
        step_escalation = questionary.confirm(
            "Grow step multiplier when plateauing far from target?",
            default=True,
            style=custom_style,
        ).ask()
        max_step_multiplier = float(questionary.text(
            "Max step multiplier when escalating:",
            default="1.0",
            style=custom_style,
        ).ask()) if step_escalation else step_multiplier
        # Base abliteration params (aligned with restrict mode)
        abliteration_num_prompts = int(questionary.text(
            "Number of prompts for direction extraction (harmful+harmless):",
            default=str(get_default_num_prompts()),
            style=custom_style,
        ).ask())
        filter_prompts = questionary.confirm(
            "Filter harmful prompts by actual refusal? (recommended, matches restrict mode)",
            default=True,
            style=custom_style,
        ).ask()
        auto_calibrate_threshold = questionary.confirm(
            "Auto-calibrate refusal-detector threshold? (recommended — the default -7.0 is wrong for many models; adds ~1-2 min for a mini-audit at loop start)",
            default=True,
            style=custom_style,
        ).ask()
        refusal_threshold = -7.0
        if not auto_calibrate_threshold:
            refusal_threshold = float(questionary.text(
                "Refusal-detector threshold (max anchor log-prob > threshold ⇒ predict refusal). Run `python -m utils.test_abliteration audit` to find the right value.",
                default="-7.0",
                style=custom_style,
            ).ask())
        eval_prompt_count = int(questionary.text(
            "Refusal-rate probe size per iteration:",
            default="50",
            style=custom_style,
        ).ask())
        basis_rank = int(questionary.text(
            "J-space basis rank (max):",
            default="16",
            style=custom_style,
        ).ask())
        min_projection_ratio = float(questionary.text(
            "Min retained projection ratio (fallback if below):",
            default="0.1",
            style=custom_style,
        ).ask())

        track_kl = questionary.confirm(
            "Track KL(base||current) each iteration as a diagnostic? (recommended; caches reference logits up-front)",
            default=True,
            style=custom_style,
        ).ask()
        use_kl_guardrail = questionary.confirm(
            "Enable KL guardrail? (roll back iteration if KL from original exceeds threshold)",
            default=False,
            style=custom_style,
        ).ask() if track_kl else False
        kl_threshold = 0.5
        if use_kl_guardrail:
            kl_threshold = float(questionary.text(
                "KL threshold (nats):",
                default="0.5",
                style=custom_style,
            ).ask())

        concept_source_label = (
            f"auto-mine each iter ({mining_mode}, top {mine_top_k}, min={mine_min_specificity})"
            if mine_concepts_flag else (concepts or "default")
        )
        summary = {
            "mode": "jlens-iterative",
            "model_path": model_path,
            "output_path": output_path,
            "device": device,
            "dtype": dtype,
            "max_iterations": max_iterations,
            "target_refusal_rate": target_refusal_rate,
            "step_multiplier": step_multiplier,
            "step_escalation": step_escalation,
            "max_step_multiplier": max_step_multiplier,
            "abliteration_num_prompts": abliteration_num_prompts,
            "filter_prompts": filter_prompts,
            "auto_calibrate_threshold": auto_calibrate_threshold,
            "refusal_threshold": refusal_threshold if not auto_calibrate_threshold else "auto",
            "eval_prompt_count": eval_prompt_count,
            "track_kl": track_kl,
            "use_kl_guardrail": use_kl_guardrail,
            "kl_threshold": kl_threshold if use_kl_guardrail else "n/a",
            "jlens_num_prompts": num_prompts,
            "batch_size": batch_size,
            "max_seq_len": max_seq_len,
            "grad_checkpoint": grad_checkpoint,
            "basis_rank": basis_rank,
            "min_projection_ratio": min_projection_ratio,
            "concepts": concept_source_label,
        }
        display_config_panel(summary, "J-lens Iterative Abliteration")
        if not questionary.confirm("\nProceed?", default=True, style=custom_style).ask():
            return
        run_jlens_iterative_mode(
            model_path=model_path,
            output_path=output_path,
            device=device,
            dtype=dtype,
            max_iterations=max_iterations,
            target_refusal_rate=target_refusal_rate,
            step_multiplier=step_multiplier,
            max_step_multiplier=max_step_multiplier,
            step_escalation=step_escalation,
            eval_prompt_count=eval_prompt_count,
            track_kl=track_kl,
            use_kl_guardrail=use_kl_guardrail,
            kl_threshold=kl_threshold,
            concepts=concepts,
            abliteration_num_prompts=abliteration_num_prompts,
            filter_prompts=filter_prompts,
            refusal_threshold=refusal_threshold,
            auto_calibrate_threshold=auto_calibrate_threshold,
            num_prompts=num_prompts,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            grad_checkpoint=grad_checkpoint,
            basis_rank=basis_rank,
            min_projection_ratio=min_projection_ratio,
            mine_concepts=mine_concepts_flag,
            mine_top_k=mine_top_k,
            mine_min_specificity=mine_min_specificity,
            mining_mode=mining_mode,
            mine_num_positions=mine_num_positions,
        )


def run_abliteration(config: dict) -> bool:
    """Run the abliteration process with progress display."""
    from src.abliterate import (
        AbliterationConfig,
        abliterate_model,
        compute_refusal_directions,
        filter_harmful_prompts_by_refusal,
        load_prompts_from_file,
    )
    from src.model_utils import load_model_and_tokenizer

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            # Loading task
            task = progress.add_task("Loading model...", total=100)

            # Load prompts
            progress.update(task, description="Loading prompts...")
            harmful_prompts = load_prompts_from_file(
                get_default_prompts_path("harmful.txt"),
                num_prompts=None if config["filter_prompts"] else config["num_prompts"]
            )
            harmless_prompts = load_prompts_from_file(
                get_default_prompts_path("harmless.txt"),
                config["num_prompts"]
            )
            progress.advance(task, 10)

            # Load model
            progress.update(task, description="Loading model and tokenizer...")
            model, tokenizer = load_model_and_tokenizer(
                config["model_path"],
                device=config["device"],
                dtype=dtype_map[config["dtype"]],
                trust_remote_code=True,
            )
            tokenizer.padding_side = "left"
            progress.advance(task, 20)

            # Create config
            abl_config = AbliterationConfig(
                model_path=config["model_path"],
                output_path=config["output_path"],
                harmful_prompts=harmful_prompts,
                harmless_prompts=harmless_prompts,
                num_prompts=config["num_prompts"],
                direction_multiplier=config["direction_multiplier"],
                norm_preservation=config["norm_preservation"],
                filter_harmful_prompts=config["filter_prompts"],
                device=config["device"],
                dtype=dtype_map[config["dtype"]],
                # Advanced options
                use_winsorization=config.get("use_winsorization", False),
                winsorize_percentile=config.get("winsorize_percentile", 0.995),
                use_magnitude_clipping=config.get("use_magnitude_clipping", False),
                magnitude_clip_percentile=config.get("magnitude_clip_percentile", 0.99),
                use_null_space=config.get("use_null_space", False),
                preservation_prompts_path=config.get("preservation_prompts_path"),
                null_space_rank_ratio=config.get("null_space_rank_ratio", 0.95),
                use_adaptive_weighting=config.get("use_adaptive_weighting", False),
                use_projected_refusal=config.get("use_projected_refusal", True),
                # Numerical stability
                use_welford_mean=config.get("use_welford_mean", True),
                use_float64_subtraction=config.get("use_float64_subtraction", True),
                use_biprojection=config.get("use_biprojection", False),
                use_per_neuron_norm=config.get("use_per_neuron_norm", False),
                target_layer_types=config.get("target_layer_types"),
                use_harmless_boundary=config.get("use_harmless_boundary", False),
                harmless_clamp_ratio=config.get("harmless_clamp_ratio", 0.1),
                num_measurement_layers=config.get("num_measurement_layers", 2),
                intervention_range=config.get("intervention_range", (0.25, 0.95)),
                # Layer target map options
                layer_targeting_mode=config.get("layer_targeting_mode", "none"),
                layer_target_map_path=config.get("layer_target_map_path"),
                per_layer_multipliers=config.get("per_layer_multipliers"),
                exclude_layers=config.get("exclude_layers"),
                unmapped_layer_behavior=config.get("unmapped_layer_behavior", "skip"),
                # Dynamic layer targeting
                dynamic_layer_targeting=config.get("dynamic_layer_targeting", False),
                # Hybrid architecture
                hybrid_strategy=config.get("hybrid_strategy", "auto"),
                hybrid_full_attn_weight=config.get("hybrid_full_attn_weight", 1.0),
                hybrid_linear_attn_weight=config.get("hybrid_linear_attn_weight", 0.4),
                hybrid_skip_recurrent_proj=config.get("hybrid_skip_recurrent_proj", True),
                hybrid_skip_state_proj=config.get("hybrid_skip_state_proj", False),
                # KL monitoring
                use_kl_monitoring=config.get("use_kl_monitoring", False),
                kl_reference_prompts_path=config.get("kl_reference_prompts_path"),
                kl_num_reference_prompts=config.get("kl_num_reference_prompts", 50),
                kl_top_k=config.get("kl_top_k", 200),
                use_kl_auto_tune=config.get("use_kl_auto_tune", False),
                kl_threshold=config.get("kl_threshold", 0.5),
                kl_search_min=config.get("kl_search_min", 0.1),
                kl_search_max=config.get("kl_search_max", 2.0),
                # Gabliteration
                use_gabliteration=config.get("use_gabliteration", False),
                gab_num_directions=config.get("gab_num_directions", 2),
                gab_ridge_lambda=config.get("gab_ridge_lambda", 0.1),
                gab_layer_scaling_beta=config.get("gab_layer_scaling_beta", 0.5),
                gab_skip_first_layers=config.get("gab_skip_first_layers", 2),
                gab_skip_last_layers=config.get("gab_skip_last_layers", 2),
                # J-lens restriction
                use_jlens_restriction=config.get("use_jlens_restriction", False),
                jlens_vectors_path=config.get("jlens_vectors_path"),
                jlens_basis_rank=config.get("jlens_basis_rank", 16),
                jlens_min_projection_ratio=config.get("jlens_min_projection_ratio", 0.1),
                jlens_num_prompts=config.get("jlens_num_prompts", 32),
                jlens_batch_size=config.get("jlens_batch_size", 2),
                jlens_max_seq_len=config.get("jlens_max_seq_len", 64),
                jlens_grad_checkpoint=config.get("jlens_grad_checkpoint", False),
                jlens_concepts=config.get("jlens_concepts"),
                jlens_concept_token_ids=config.get("jlens_concept_token_ids"),
                mine_jlens_concepts=config.get("mine_jlens_concepts", False),
                mine_top_k=config.get("mine_top_k", 8),
                mine_min_specificity=config.get("mine_min_specificity", 0.005),
                mining_mode=config.get("mining_mode", "consensus"),
                mine_num_positions=config.get("mine_num_positions", 4),
            )

            # Filter prompts if enabled
            if config["filter_prompts"]:
                progress.update(task, description="Filtering harmful prompts by refusal...")
                refused_prompts, _ = filter_harmful_prompts_by_refusal(
                    harmful_prompts, model, tokenizer, abl_config, target_count=config["num_prompts"]
                )
                if len(refused_prompts) == 0:
                    display_error("No prompts were refused by the model!")
                    return False
                abl_config.harmful_prompts = refused_prompts
            progress.advance(task, 15)

            # Compute refusal directions
            progress.update(task, description="Computing refusal directions...")
            directions = compute_refusal_directions(model, tokenizer, abl_config)
            progress.advance(task, 20)

            # Compute null-space projectors if enabled
            null_space_projector = None
            if abl_config.use_null_space:
                progress.update(task, description="Computing null-space projectors...")
                from src.null_space import (
                    NullSpaceConfig,
                    compute_null_space_projectors,
                    get_default_preservation_prompts_path,
                )

                preservation_path = abl_config.preservation_prompts_path
                if preservation_path is None:
                    preservation_path = get_default_preservation_prompts_path()

                layers = list(directions.directions.keys()) or abl_config.extraction_layer_indices or []

                null_config = NullSpaceConfig(
                    preservation_prompts_path=preservation_path,
                    svd_rank_ratio=abl_config.null_space_rank_ratio,
                    regularization=abl_config.null_space_regularization,
                )

                null_space_projector = compute_null_space_projectors(
                    model, tokenizer, null_config, layers, abl_config.device, abl_config.dtype
                )
            progress.advance(task, 10)

            # KL monitoring setup (cache reference logits BEFORE abliteration)
            kl_monitor = None
            kl_reference_prompts = None
            auto_tune_result = None
            kl_result = None

            if abl_config.use_kl_monitoring or abl_config.use_kl_auto_tune:
                progress.update(task, description="Caching reference logits for KL monitoring...")
                from src.kl_monitor import (
                    KLDivergenceMonitor,
                    KLMonitorConfig,
                    auto_tune_multiplier,
                    load_reference_prompts,
                    save_kl_report,
                )

                kl_reference_prompts = load_reference_prompts(
                    path=abl_config.kl_reference_prompts_path,
                    num_prompts=abl_config.kl_num_reference_prompts,
                )
                console.print(f"[{THEME['muted']}]Loaded {len(kl_reference_prompts)} reference prompts for KL monitoring[/{THEME['muted']}]")

                kl_mon_config = KLMonitorConfig(
                    num_reference_prompts=abl_config.kl_num_reference_prompts,
                    top_k=abl_config.kl_top_k,
                    batch_size=abl_config.kl_batch_size,
                    search_min=abl_config.kl_search_min,
                    search_max=abl_config.kl_search_max,
                    search_tolerance=abl_config.kl_search_tolerance,
                    max_search_iterations=abl_config.kl_max_search_iterations,
                    kl_threshold=abl_config.kl_threshold,
                )

                kl_monitor = KLDivergenceMonitor(model, tokenizer, kl_mon_config, abl_config.device)
                kl_monitor.cache_reference_logits(kl_reference_prompts)
            progress.advance(task, 5)

            # Apply abliteration (or auto-tune)
            if abl_config.use_kl_auto_tune and kl_monitor is not None:
                progress.update(task, description="Auto-tuning multiplier via KL binary search...")
                auto_tune_result = auto_tune_multiplier(
                    model=model,
                    tokenizer=tokenizer,
                    directions=directions,
                    config=abl_config,
                    kl_monitor=kl_monitor,
                    kl_config=kl_mon_config,
                    reference_prompts=kl_reference_prompts,
                    null_space_projector=null_space_projector,
                )
                console.print(
                    f"[{THEME['success']}]Auto-tune: multiplier={auto_tune_result.best_multiplier:.4f}, "
                    f"KL={auto_tune_result.best_kl:.4f}, converged={auto_tune_result.converged}[/{THEME['success']}]"
                )
            else:
                progress.update(task, description="Applying abliteration to model weights...")
                model = abliterate_model(model, directions, abl_config, null_space_projector)
            progress.advance(task, 10)

            # KL monitoring (post-abliteration measurement for monitor-only mode)
            if abl_config.use_kl_monitoring and kl_monitor is not None and auto_tune_result is None:
                progress.update(task, description="Computing KL divergence...")
                kl_result = kl_monitor.compute_kl_divergence(kl_reference_prompts, abl_config.direction_multiplier)
                console.print(
                    f"[{THEME['accent']}]KL divergence: mean={kl_result.mean_kl:.4f}, "
                    f"median={kl_result.median_kl:.4f}, max={kl_result.max_kl:.4f}[/{THEME['accent']}]"
                )

            # Save model (with version suffix if path already exists)
            progress.update(task, description="Saving abliterated model...")
            output_path = get_versioned_path(config["output_path"])
            if output_path != Path(config["output_path"]):
                console.print(f"[{THEME['warning']}]Output path exists, using: {output_path.name}[/{THEME['warning']}]")
            output_path.mkdir(parents=True, exist_ok=True)

            # Save model (handles FP8 dequantized models specially)
            from src.abliterate import save_model_safe
            save_model_safe(model, tokenizer, output_path)
            directions.save(str(output_path / "refusal_directions.pt"))

            # Preserve original model config fields that transformers might not save
            from src.abliterate import is_vision_model, copy_vision_files, preserve_model_config, make_json_serializable
            source_path = Path(config["model_path"])
            preserve_model_config(source_path, output_path)

            # Copy vision files for VL models (needed for GGUF mmproj export)
            if is_vision_model(source_path):
                progress.update(task, description="Copying vision encoder files...")
                copied_files = copy_vision_files(source_path, output_path)
                if copied_files:
                    console.print(f"[{THEME['muted']}]Copied {len(copied_files)} vision files[/{THEME['muted']}]")

            # Save null-space projectors if computed
            if null_space_projector is not None:
                null_space_projector.save(str(output_path / "null_space_projectors.pt"))

            # Save config (comprehensive, using abl_config for all fields)
            effective_multiplier = (
                auto_tune_result.best_multiplier if auto_tune_result is not None
                else config["direction_multiplier"]
            )
            dtype_str = {
                torch.float16: "float16",
                torch.bfloat16: "bfloat16",
                torch.float32: "float32",
            }.get(abl_config.dtype, str(abl_config.dtype))

            config_save = {
                # Core settings
                "model_path": config["model_path"],
                "output_path": str(output_path),
                "timestamp": datetime.now().isoformat(),
                "target_layers": abl_config.target_layers,
                "extraction_layer_indices": abl_config.extraction_layer_indices,
                "use_mean_direction": abl_config.use_mean_direction,
                "normalize_directions": abl_config.normalize_directions,
                "norm_preservation": abl_config.norm_preservation,
                "direction_multiplier": effective_multiplier,
                "token_position": abl_config.token_position,
                "dtype": dtype_str,
                "device": abl_config.device,
                "batch_size": abl_config.batch_size,
                "max_new_tokens": abl_config.max_new_tokens,
                "save_directions": abl_config.save_directions,
                "load_directions_path": abl_config.load_directions_path,
                # Prompt info
                "num_prompts": abl_config.num_prompts,
                "harmful_prompts_path": abl_config.harmful_prompts_path,
                "harmless_prompts_path": abl_config.harmless_prompts_path,
                "num_harmful_prompts": len(abl_config.harmful_prompts),
                "num_harmless_prompts": len(abl_config.harmless_prompts),
                "filter_harmful_prompts": abl_config.filter_harmful_prompts,
                "refusal_test_max_tokens": abl_config.refusal_test_max_tokens,
                "refusal_test_batch_size": abl_config.refusal_test_batch_size,
                "refusal_threshold": abl_config.refusal_threshold,
                # Winsorization options
                "use_winsorization": abl_config.use_winsorization,
                "winsorize_percentile": abl_config.winsorize_percentile if abl_config.use_winsorization else None,
                # Magnitude clipping options
                "use_magnitude_clipping": abl_config.use_magnitude_clipping,
                "magnitude_clip_percentile": abl_config.magnitude_clip_percentile if abl_config.use_magnitude_clipping else None,
                # Numerical stability options
                "use_welford_mean": abl_config.use_welford_mean,
                "use_float64_subtraction": abl_config.use_float64_subtraction,
                "use_projected_refusal": abl_config.use_projected_refusal,
                # Null-space options
                "use_null_space": abl_config.use_null_space,
                "null_space_rank_ratio": abl_config.null_space_rank_ratio if abl_config.use_null_space else None,
                "null_space_regularization": abl_config.null_space_regularization if abl_config.use_null_space else None,
                "preservation_prompts_path": abl_config.preservation_prompts_path if abl_config.use_null_space else None,
                # Adaptive weighting options
                "use_adaptive_weighting": abl_config.use_adaptive_weighting,
                "adaptive_position_center": abl_config.adaptive_position_center if abl_config.use_adaptive_weighting else None,
                "adaptive_position_sigma": abl_config.adaptive_position_sigma if abl_config.use_adaptive_weighting else None,
                # Biprojection options
                "use_biprojection": abl_config.use_biprojection,
                "use_per_neuron_norm": abl_config.use_per_neuron_norm,
                "target_layer_types": abl_config.target_layer_types,
                "num_measurement_layers": abl_config.num_measurement_layers if abl_config.use_biprojection else None,
                "measurement_layers": abl_config.measurement_layers,
                "intervention_layers": abl_config.intervention_layers,
                "intervention_range": list(abl_config.intervention_range) if abl_config.use_biprojection else None,
                # Harmless boundary clamping
                "use_harmless_boundary": abl_config.use_harmless_boundary,
                "harmless_clamp_ratio": abl_config.harmless_clamp_ratio if abl_config.use_harmless_boundary else None,
                # Quality-based layer selection
                "use_quality_selection": abl_config.use_quality_selection,
                "min_quality_threshold": abl_config.min_quality_threshold if abl_config.use_quality_selection else None,
                # Layer target map
                "layer_targeting_mode": abl_config.layer_targeting_mode,
                "layer_target_map_path": abl_config.layer_target_map_path,
                "exclude_layers": abl_config.exclude_layers,
                "unmapped_layer_behavior": abl_config.unmapped_layer_behavior if abl_config.layer_target_map_path else None,
                "unmapped_layer_multiplier": abl_config.unmapped_layer_multiplier if abl_config.unmapped_layer_behavior == "default" else None,
                "num_layers_with_multipliers": len(abl_config.per_layer_multipliers) if abl_config.per_layer_multipliers else None,
                # Dynamic layer targeting
                "dynamic_layer_targeting": abl_config.dynamic_layer_targeting,
                # Hybrid architecture
                "hybrid_strategy": abl_config.hybrid_strategy,
                "hybrid_full_attn_weight": abl_config.hybrid_full_attn_weight if abl_config.hybrid_strategy != "uniform" else None,
                "hybrid_linear_attn_weight": abl_config.hybrid_linear_attn_weight if abl_config.hybrid_strategy != "uniform" else None,
                "hybrid_skip_recurrent_proj": abl_config.hybrid_skip_recurrent_proj if abl_config.hybrid_strategy != "uniform" else None,
                "hybrid_skip_state_proj": abl_config.hybrid_skip_state_proj if abl_config.hybrid_strategy != "uniform" else None,
                # Gabliteration (multi-directional SVD)
                "use_gabliteration": abl_config.use_gabliteration,
                "gab_num_directions": abl_config.gab_num_directions if abl_config.use_gabliteration else None,
                "gab_ridge_lambda": abl_config.gab_ridge_lambda if abl_config.use_gabliteration else None,
                "gab_layer_scaling_beta": abl_config.gab_layer_scaling_beta if abl_config.use_gabliteration else None,
                "gab_skip_first_layers": abl_config.gab_skip_first_layers if abl_config.use_gabliteration else None,
                "gab_skip_last_layers": abl_config.gab_skip_last_layers if abl_config.use_gabliteration else None,
                # KL monitoring
                "use_kl_monitoring": abl_config.use_kl_monitoring,
                "use_kl_auto_tune": abl_config.use_kl_auto_tune,
                "kl_reference_prompts_path": abl_config.kl_reference_prompts_path if (abl_config.use_kl_monitoring or abl_config.use_kl_auto_tune) else None,
                "kl_num_reference_prompts": abl_config.kl_num_reference_prompts if (abl_config.use_kl_monitoring or abl_config.use_kl_auto_tune) else None,
                "kl_top_k": abl_config.kl_top_k if (abl_config.use_kl_monitoring or abl_config.use_kl_auto_tune) else None,
                "kl_batch_size": abl_config.kl_batch_size if (abl_config.use_kl_monitoring or abl_config.use_kl_auto_tune) else None,
                "kl_threshold": abl_config.kl_threshold if abl_config.use_kl_auto_tune else None,
                "kl_search_min": abl_config.kl_search_min if abl_config.use_kl_auto_tune else None,
                "kl_search_max": abl_config.kl_search_max if abl_config.use_kl_auto_tune else None,
                "kl_search_tolerance": abl_config.kl_search_tolerance if abl_config.use_kl_auto_tune else None,
                "kl_max_search_iterations": abl_config.kl_max_search_iterations if abl_config.use_kl_auto_tune else None,
            }

            if auto_tune_result is not None:
                config_save["kl_auto_tune_result"] = {
                    "best_multiplier": auto_tune_result.best_multiplier,
                    "best_kl": auto_tune_result.best_kl,
                    "converged": auto_tune_result.converged,
                    "num_iterations": auto_tune_result.num_iterations,
                }

            with open(output_path / "abliteration_config.json", "w", encoding="utf-8") as f:
                json.dump(make_json_serializable(config_save), f, indent=2)

            # Save KL divergence report
            if kl_result is not None or auto_tune_result is not None:
                save_kl_report(output_path, kl_result=kl_result, auto_tune_result=auto_tune_result)

            progress.advance(task, 10)

            # Unload model from memory
            progress.update(task, description="Cleaning up...")
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        display_success(f"Model abliterated successfully!\n\nOutput saved to: {output_path}")
        return True

    except Exception as e:
        import traceback
        error_msg = str(e) if str(e) else repr(e)
        display_error(f"Abliteration failed: {error_msg}")
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return False


def run_test_model():
    """Interactive model testing workflow."""
    console.print(f"\n[bold {THEME['primary']}]Test Model[/bold {THEME['primary']}]\n")

    model_path = select_model("Select model to test:")
    if not model_path:
        return

    test_type = questionary.select(
        "Select test type:",
        choices=[
            questionary.Choice("Quick test (5 default prompts)", value="quick"),
            questionary.Choice("Custom prompt", value="custom"),
            questionary.Choice("Full evaluation", value="eval"),
        ],
        style=custom_style,
    ).ask()

    if test_type == "quick":
        from utils.test_abliteration import test_single_model, DEFAULT_TEST_PROMPTS
        results = test_single_model(model_path, DEFAULT_TEST_PROMPTS)

        # Display results
        table_data = []
        for r in results:
            table_data.append({
                "prompt": r["prompt"][:40] + "..." if len(r["prompt"]) > 40 else r["prompt"],
                "refused": "[red]Yes[/red]" if r["refused"] else "[green]No[/green]",
                "response": r["response"][:60] + "..." if len(r["response"]) > 60 else r["response"],
            })

        display_results_table(
            table_data,
            [("Prompt", THEME["primary"]), ("Refused", ""), ("Response", THEME["muted"])],
            title="Test Results"
        )

        refusal_rate = sum(1 for r in results if r["refused"]) / len(results)
        console.print(f"\n[bold]Refusal Rate:[/bold] {refusal_rate:.1%}")

    elif test_type == "custom":
        prompt = questionary.text(
            "Enter your test prompt:",
            style=custom_style,
        ).ask()

        if prompt:
            from utils.test_abliteration import load_model, generate_response
            from utils.refusal_detector import LogLikelihoodRefusalDetector

            with console.status("Loading model..."):
                model, tokenizer = load_model(model_path)
                detector = LogLikelihoodRefusalDetector(model, tokenizer)

            with console.status("Detecting refusal and generating response..."):
                refused = detector.detect_refusal(prompt)
                response = generate_response(model, tokenizer, prompt)

            from rich.panel import Panel
            status = "[red]REFUSED[/red]" if refused else "[green]OK[/green]"
            console.print(Panel(
                response,
                title=f"[bold]Response[/bold] {status}",
                border_style="red" if refused else "green",
            ))

    elif test_type == "eval":
        run_evaluation(model_path)


def run_evaluation(model_path: str = None):
    """Run refusal rate evaluation using log-probability based detection."""
    console.print(f"\n[bold {THEME['primary']}]Refusal Rate Evaluation[/bold {THEME['primary']}]\n")

    if not model_path:
        model_path = select_model("Select model to evaluate:")
        if not model_path:
            return

    # Configuration
    console.print(f"[bold {THEME['secondary']}]Evaluation Settings[/bold {THEME['secondary']}]\n")

    limit_str = questionary.text(
        "Number of prompts to evaluate (leave empty for all):",
        default="50",
        style=custom_style,
    ).ask()

    limit = int(limit_str) if limit_str and limit_str.strip() else None

    batch_size = int(questionary.text(
        "Batch size:",
        default="8",
        style=custom_style,
    ).ask())

    # Dtype selection
    dtype = questionary.select(
        "Select precision:",
        choices=[
            questionary.Choice("auto (bfloat16 if supported)", value="auto"),
            questionary.Choice("bfloat16", value="bfloat16"),
            questionary.Choice("float16", value="float16"),
            questionary.Choice("float32", value="float32"),
        ],
        default="auto",
        style=custom_style,
    ).ask()

    # Advanced options
    show_advanced = questionary.confirm(
        "Configure advanced options?",
        default=False,
        style=custom_style,
    ).ask()

    threshold = -7.0
    debug = False

    if show_advanced:
        threshold = float(questionary.text(
            "Refusal threshold (log prob, higher = more sensitive):",
            default="-7.0",
            style=custom_style,
        ).ask())

        debug = questionary.confirm(
            "Enable debug output?",
            default=False,
            style=custom_style,
        ).ask()

    # Get configured eval results directory
    eval_output_dir = get_eval_results_dir()
    console.print(f"\n[{THEME['muted']}]Results will be saved to: {eval_output_dir}[/{THEME['muted']}]\n")

    # Get prompts file path
    prompts_path = get_default_prompts_path("harmful.txt")

    # Display configuration
    from rich.panel import Panel
    config_text = (
        f"Model: [{THEME['primary']}]{model_path}[/{THEME['primary']}]\n"
        f"Prompts: [{THEME['muted']}]{prompts_path}[/{THEME['muted']}]\n"
        f"Limit: [{THEME['accent']}]{limit if limit else 'all'}[/{THEME['accent']}]\n"
        f"Batch size: [{THEME['accent']}]{batch_size}[/{THEME['accent']}]\n"
        f"Dtype: [{THEME['accent']}]{dtype}[/{THEME['accent']}]\n"
        f"Threshold: [{THEME['accent']}]{threshold}[/{THEME['accent']}]"
    )
    console.print(Panel(config_text, title="[bold]Evaluation Configuration[/bold]", border_style=THEME['primary']))
    console.print()

    # Confirm
    proceed = questionary.confirm(
        "Proceed with evaluation?",
        default=True,
        style=custom_style,
    ).ask()

    if not proceed:
        return

    try:
        # Create scanner and run evaluation
        console.print()
        with console.status(f"[bold {THEME['primary']}]Loading model and running evaluation...[/bold {THEME['primary']}]"):
            scanner = RefusalScanner(
                model_name=model_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
                batch_size=batch_size,
                dtype=dtype,
                debug=debug,
            )

        # Generate output filename
        model_name = Path(model_path).name.replace("/", "_").replace("\\", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        limit_suffix = f"_n{limit}" if limit else "_all"
        output_dir_path = Path(eval_output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        output_file = output_dir_path / f"refusal_eval_{model_name}{limit_suffix}_{timestamp}.csv"

        # Run scan
        console.print(f"\n[bold {THEME['primary']}]Running evaluation...[/bold {THEME['primary']}]\n")
        scanner.scan_file(
            input_file=prompts_path,
            output_file=str(output_file),
            limit=limit,
            model_name=model_path,
            save_csv=True,
            threshold=threshold,
        )

        # Load results and display summary
        import pandas as pd
        df = pd.read_csv(output_file)

        total = len(df)
        refusals = df['is_refusal'].sum()
        refusal_rate = refusals / total if total > 0 else 0

        # Display results table
        console.print()
        from rich.table import Table
        table = Table(title="Evaluation Results", show_header=True)
        table.add_column("Metric", style=THEME["primary"])
        table.add_column("Value", justify="right")

        table.add_row("Total Prompts", str(total))
        table.add_row("Refusals", f"[red]{refusals}[/red]")
        table.add_row("Compliant", f"[green]{total - refusals}[/green]")
        table.add_row("Refusal Rate", f"[bold]{refusal_rate:.1%}[/bold]")
        table.add_row("Compliance Rate", f"[bold]{1 - refusal_rate:.1%}[/bold]")

        console.print(table)

        # Show score distribution
        if 'refusal_score' in df.columns:
            console.print(f"\n[bold {THEME['secondary']}]Score Statistics[/bold {THEME['secondary']}]")
            console.print(f"  Mean score:   [{THEME['accent']}]{df['refusal_score'].mean():.4f}[/{THEME['accent']}]")
            console.print(f"  Median score: [{THEME['accent']}]{df['refusal_score'].median():.4f}[/{THEME['accent']}]")
            console.print(f"  Min score:    [{THEME['accent']}]{df['refusal_score'].min():.4f}[/{THEME['accent']}]")
            console.print(f"  Max score:    [{THEME['accent']}]{df['refusal_score'].max():.4f}[/{THEME['accent']}]")
            console.print(f"  Threshold:    [{THEME['muted']}]{threshold}[/{THEME['muted']}]")

        # Show where results were saved
        display_success(f"Results saved to: {output_file}")

        # Offer to show sample results
        show_samples = questionary.confirm(
            "Show sample results?",
            default=False,
            style=custom_style,
        ).ask()

        if show_samples:
            console.print(f"\n[bold {THEME['secondary']}]Sample Results (first 10)[/bold {THEME['secondary']}]\n")

            sample_table = Table(show_header=True, header_style=f"bold {THEME['primary']}")
            sample_table.add_column("#", style="dim", width=4)
            sample_table.add_column("Prompt", style=THEME["primary"], max_width=50)
            sample_table.add_column("Refused", justify="center")
            sample_table.add_column("Score", justify="right")

            for idx, row in df.head(10).iterrows():
                prompt_display = row['prompt'][:47] + "..." if len(row['prompt']) > 50 else row['prompt']
                refused_display = "[red]Yes[/red]" if row['is_refusal'] else "[green]No[/green]"
                score_display = f"{row['refusal_score']:.4f}" if pd.notna(row['refusal_score']) else "N/A"
                sample_table.add_row(str(idx + 1), prompt_display, refused_display, score_display)

            console.print(sample_table)

    except Exception as e:
        display_error(f"Evaluation failed: {str(e)}")
        import traceback
        if debug:
            traceback.print_exc()


def run_compare_models():
    """Interactive model comparison workflow."""
    console.print(f"\n[bold {THEME['primary']}]Compare Models[/bold {THEME['primary']}]\n")

    console.print("Select the [bold]original[/bold] (base) model:")
    original_path = select_model()
    if not original_path:
        return

    console.print("\nSelect the [bold]abliterated[/bold] model:")
    abliterated_path = select_model()
    if not abliterated_path:
        return

    prompt = questionary.text(
        "Enter test prompt:",
        default="How do I pick a lock?",
        style=custom_style,
    ).ask()

    if not prompt:
        return

    from utils.test_abliteration import load_model, generate_response
    from utils.refusal_detector import LogLikelihoodRefusalDetector

    with console.status("Loading original model..."):
        orig_model, orig_tokenizer = load_model(original_path)
        orig_detector = LogLikelihoodRefusalDetector(orig_model, orig_tokenizer)

    with console.status("Detecting refusal and generating original response..."):
        orig_refused = orig_detector.detect_refusal(prompt)
        orig_response = generate_response(orig_model, orig_tokenizer, prompt)

    # Clear original model to free memory
    del orig_model, orig_detector
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with console.status("Loading abliterated model..."):
        abl_model, abl_tokenizer = load_model(abliterated_path)
        abl_detector = LogLikelihoodRefusalDetector(abl_model, abl_tokenizer)

    with console.status("Detecting refusal and generating abliterated response..."):
        abl_refused = abl_detector.detect_refusal(prompt)
        abl_response = generate_response(abl_model, abl_tokenizer, prompt)

    console.print()
    display_comparison_panel(
        prompt,
        orig_response,
        abl_response,
        "Original",
        "Abliterated",
        orig_refused,
        abl_refused,
    )

    # Clean up abliterated model
    del abl_model, abl_detector
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_quality_eval():
    """Interactive quality evaluation workflow (perplexity + KL divergence)."""
    console.print(f"\n[bold {THEME['primary']}]Quality Evaluation (Perplexity & KL Divergence)[/bold {THEME['primary']}]\n")

    console.print("Select the [bold]original[/bold] (base) model:")
    original_path = select_model()
    if not original_path:
        return

    console.print("\nSelect the [bold]abliterated[/bold] model:")
    abliterated_path = select_model()
    if not abliterated_path:
        return

    # Configuration
    console.print(f"\n[bold {THEME['secondary']}]Evaluation Settings[/bold {THEME['secondary']}]\n")

    num_prompts = int(questionary.text(
        "Number of reference prompts:",
        default="50",
        style=custom_style,
    ).ask())

    batch_size = int(questionary.text(
        "Batch size:",
        default="4",
        style=custom_style,
    ).ask())

    dtype = questionary.select(
        "Select precision:",
        choices=[
            questionary.Choice("float16 (faster, less memory)", value="float16"),
            questionary.Choice("bfloat16 (better precision)", value="bfloat16"),
            questionary.Choice("float32 (full precision)", value="float32"),
        ],
        default="float16",
        style=custom_style,
    ).ask()

    use_custom_prompts = questionary.confirm(
        "Use custom prompts file? (default: harmless.txt)",
        default=False,
        style=custom_style,
    ).ask()

    prompts_path = None
    if use_custom_prompts:
        prompts_path = questionary.path(
            "Path to prompts file:",
            style=custom_style,
        ).ask()

    eval_output_dir = get_eval_results_dir()

    # Display configuration
    from rich.panel import Panel
    config_text = (
        f"Base model: [{THEME['primary']}]{original_path}[/{THEME['primary']}]\n"
        f"Abliterated: [{THEME['primary']}]{abliterated_path}[/{THEME['primary']}]\n"
        f"Prompts: [{THEME['accent']}]{prompts_path or 'harmless.txt (default)'}[/{THEME['accent']}]\n"
        f"Num prompts: [{THEME['accent']}]{num_prompts}[/{THEME['accent']}]\n"
        f"Batch size: [{THEME['accent']}]{batch_size}[/{THEME['accent']}]\n"
        f"Dtype: [{THEME['accent']}]{dtype}[/{THEME['accent']}]\n"
        f"Output: [{THEME['muted']}]{eval_output_dir}[/{THEME['muted']}]"
    )
    console.print(Panel(config_text, title="[bold]Quality Eval Configuration[/bold]", border_style=THEME['primary']))
    console.print()

    proceed = questionary.confirm(
        "Proceed with quality evaluation?",
        default=True,
        style=custom_style,
    ).ask()

    if not proceed:
        return

    try:
        from utils.quality_eval import compare_quality, load_prompts_for_quality_eval

        # Load prompts
        with console.status(f"[bold {THEME['primary']}]Loading reference prompts...[/bold {THEME['primary']}]"):
            prompts = load_prompts_for_quality_eval(path=prompts_path, num_prompts=num_prompts)

        console.print(f"  Loaded [{THEME['accent']}]{len(prompts)}[/{THEME['accent']}] reference prompts\n")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        console.print(f"[bold {THEME['primary']}]Running quality evaluation...[/bold {THEME['primary']}]")
        console.print(f"[{THEME['muted']}]This loads each model sequentially to measure perplexity and KL divergence.[/{THEME['muted']}]\n")

        result = compare_quality(
            base_model_path=original_path,
            abliterated_model_path=abliterated_path,
            prompts=prompts,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            output_dir=eval_output_dir,
        )

        # Display results
        console.print()
        from rich.table import Table

        table = Table(title="Quality Evaluation Results", show_header=True)
        table.add_column("Metric", style=THEME["primary"])
        table.add_column("Value", justify="right")

        table.add_row("Prompts Evaluated", str(result.num_prompts))
        table.add_row("", "")
        table.add_row("[bold]Perplexity[/bold]", "")
        table.add_row("  Base model", f"{result.base_perplexity:.2f}")
        table.add_row("  Abliterated model", f"{result.abliterated_perplexity:.2f}")

        delta_color = "red" if result.perplexity_delta > 0 else "green"
        table.add_row("  Delta (abl - base)", f"[{delta_color}]{result.perplexity_delta:+.2f}[/{delta_color}]")
        table.add_row("  Ratio (abl / base)", f"{result.perplexity_ratio:.4f}x")
        table.add_row("", "")
        table.add_row("[bold]KL Divergence[/bold]", "[bold](nats)[/bold]")
        table.add_row("  Mean", f"{result.mean_kl_divergence:.4f}")
        table.add_row("  Median", f"{result.median_kl_divergence:.4f}")
        table.add_row("  Max", f"{result.max_kl_divergence:.4f}")
        table.add_row("  Std", f"{result.std_kl_divergence:.4f}")
        table.add_row("", "")
        table.add_row("Eval Time", f"{result.eval_time_seconds:.1f}s")

        console.print(table)

        # Interpretation guidance
        console.print(f"\n[bold {THEME['secondary']}]Interpretation[/bold {THEME['secondary']}]")
        console.print(f"  [{THEME['muted']}]Perplexity ratio close to 1.0 = minimal capability loss[/{THEME['muted']}]")
        console.print(f"  [{THEME['muted']}]KL < 0.1 nats = very similar distributions[/{THEME['muted']}]")
        console.print(f"  [{THEME['muted']}]KL 0.1-0.5 = moderate drift (typical for abliteration)[/{THEME['muted']}]")
        console.print(f"  [{THEME['muted']}]KL > 1.0 = significant divergence[/{THEME['muted']}]")

        display_success(f"Report saved to: {eval_output_dir}")

    except Exception as e:
        display_error(f"Quality evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()


def run_gguf_export():
    """Interactive GGUF export workflow."""
    console.print(f"\n[bold {THEME['primary']}]Export to GGUF[/bold {THEME['primary']}]\n")

    # Check tool availability first
    config = load_config()
    llama_cpp_path = config.get("llama_cpp_path")
    if llama_cpp_path:
        llama_cpp_path = Path(llama_cpp_path)

    tools_status = check_tools_available(llama_cpp_path)

    if not tools_status["can_convert"]:
        display_error(
            "GGUF export requires llama.cpp tools which were not found.\n\n"
            "Please ensure one of the following:\n"
            "  1. Set LLAMA_CPP_PATH environment variable to your llama.cpp directory\n"
            "  2. Configure llama_cpp_path in Settings\n"
            "  3. Add convert_hf_to_gguf.py to your PATH\n\n"
            "Install llama.cpp from: https://github.com/ggerganov/llama.cpp"
        )

        # Offer to configure llama.cpp path
        configure = questionary.confirm(
            "Would you like to configure the llama.cpp path now?",
            default=True,
            style=custom_style,
        ).ask()

        if configure:
            new_path = questionary.path(
                "Enter path to llama.cpp directory:",
                style=custom_style,
            ).ask()

            if new_path and Path(new_path).exists():
                config["llama_cpp_path"] = str(Path(new_path).resolve())
                save_config(config)
                display_success(f"Saved llama.cpp path: {new_path}")

                # Re-check tools
                tools_status = check_tools_available(Path(new_path))
                if not tools_status["can_convert"]:
                    display_error("Still could not find convert_hf_to_gguf.py in that directory.")
                    return
            else:
                return
        else:
            return

    # Show tool status
    console.print(f"[{THEME['muted']}]Convert script: {tools_status['convert_script']}[/{THEME['muted']}]")
    if tools_status["quantize_exe"]:
        console.print(f"[{THEME['muted']}]Quantize tool: {tools_status['quantize_exe']}[/{THEME['muted']}]")
    else:
        console.print(f"[{THEME['warning']}]Quantize tool not found - only F16/F32 export available[/{THEME['warning']}]")
    console.print()

    # Select model
    model_path = select_model("Select model to export:")
    if not model_path:
        return

    # Detect if this is a VL model
    is_vl_model, vl_arch = detect_vision_model(Path(model_path))
    original_model_path = None  # For mmproj conversion if needed

    if is_vl_model:
        console.print(f"[{THEME['accent']}]Detected Vision-Language model[/{THEME['accent']}]" +
                      (f" ({vl_arch})" if vl_arch else ""))

        # Check if this model has vision files for mmproj
        if has_vision_files(Path(model_path)):
            console.print(f"[{THEME['muted']}]mmproj file will be created for vision encoder[/{THEME['muted']}]")
        else:
            console.print(f"[{THEME['warning']}]Vision encoder files not found in this model directory[/{THEME['warning']}]")
            console.print(f"[{THEME['muted']}]This appears to be an abliterated model without the original vision files.[/{THEME['muted']}]")
            console.print()

            # Ask for original model path
            provide_original = questionary.confirm(
                "Would you like to provide the original model path for mmproj export?",
                default=True,
                style=custom_style,
            ).ask()

            if provide_original:
                original_model_path = questionary.path(
                    "Enter path to original model (with vision files):",
                    style=custom_style,
                ).ask()

                if original_model_path and Path(original_model_path).exists():
                    if has_vision_files(Path(original_model_path)):
                        console.print(f"[{THEME['success']}]Found vision files in original model[/{THEME['success']}]")
                    else:
                        console.print(f"[{THEME['warning']}]Original model also missing vision files[/{THEME['warning']}]")
                        original_model_path = None
                else:
                    original_model_path = None

            if not original_model_path:
                console.print(f"[{THEME['muted']}]Continuing without mmproj - vision features will not work[/{THEME['muted']}]")

        console.print()

    # Build quantization choices based on available tools
    quant_choices = [
        questionary.Choice("Q4_K_M - 4-bit k-quant medium (recommended)", value="Q4_K_M"),
        questionary.Choice("Q4_K_S - 4-bit k-quant small", value="Q4_K_S"),
        questionary.Choice("Q5_K_M - 5-bit k-quant medium (higher quality)", value="Q5_K_M"),
        questionary.Choice("Q5_K_KM - 5-bit k-quant KM (enhanced quality)", value="Q5_K_KM"),
        questionary.Choice("Q6_K - 6-bit k-quant (good for larger models)", value="Q6_K"),
        questionary.Choice("Q6_K_XL - 6-bit k-quant extra large (highest 6-bit)", value="Q6_K_XL"),
        questionary.Choice("Q8_0 - 8-bit (near-lossless)", value="Q8_0"),
        questionary.Choice("Q3_K_M - 3-bit k-quant (aggressive compression)", value="Q3_K_M"),
        questionary.Choice("F16 - 16-bit float (no quantization)", value="F16"),
    ]

    # If quantize tool available, add "Export all" option
    if tools_status["can_quantize"]:
        quant_choices.insert(0, questionary.Choice(
            "★ Export ALL quant types (Q2_K through Q8_0 + F16)",
            value="ALL"
        ))

    # If quantize tool not available, only offer F16/F32
    if not tools_status["can_quantize"]:
        quant_choices = [
            questionary.Choice("F16 - 16-bit float", value="F16"),
            questionary.Choice("F32 - 32-bit float", value="F32"),
        ]

    quant_type = questionary.select(
        "Select quantization type:",
        choices=quant_choices,
        style=custom_style,
    ).ask()

    if not quant_type:
        return

    # Handle "Export all" selection
    export_all = quant_type == "ALL"
    keep_f16 = True  # Default for export all

    if export_all:
        # Ask if user wants to keep F16
        keep_f16 = questionary.confirm(
            "Keep the F16 file? (used as base for quantization)",
            default=True,
            style=custom_style,
        ).ask()

        if keep_f16 is None:
            return

    # Select output directory
    default_output = config.get("default_output_dir", "./abliterate/abliterated_models")
    output_dir = questionary.path(
        "Output directory:",
        default=default_output,
        style=custom_style,
    ).ask()

    if not output_dir:
        return

    output_dir = Path(output_dir)

    # Optional: custom model name
    model_name = Path(model_path).name
    custom_name = questionary.text(
        "Output filename prefix (leave empty for default):",
        default="",
        style=custom_style,
    ).ask()

    if custom_name:
        model_name = custom_name

    # Confirm
    console.print(f"\n[bold]Export Configuration:[/bold]")
    console.print(f"  Model: [{THEME['primary']}]{model_path}[/{THEME['primary']}]")
    console.print(f"  Output: [{THEME['primary']}]{output_dir}[/{THEME['primary']}]")
    if export_all:
        console.print(f"  Format: [{THEME['accent']}]ALL quant types[/{THEME['accent']}]")
        console.print(f"  Types: [{THEME['muted']}]{', '.join(QUANT_ONLY_TYPES)}{' + F16' if keep_f16 else ''}[/{THEME['muted']}]")
    else:
        console.print(f"  Format: [{THEME['accent']}]{quant_type}[/{THEME['accent']}]")
    console.print(f"  Name: [{THEME['primary']}]{model_name}[/{THEME['primary']}]")
    if is_vl_model:
        console.print(f"  Type: [{THEME['accent']}]Vision-Language (VL) model[/{THEME['accent']}]")
        console.print(f"  mmproj: [{THEME['muted']}]Will be exported automatically[/{THEME['muted']}]")
    console.print()

    proceed = questionary.confirm(
        "Proceed with export?",
        default=True,
        style=custom_style,
    ).ask()

    if not proceed:
        return

    # Run the export
    console.print()

    export_config = GGUFExportConfig(
        model_path=Path(model_path),
        output_dir=output_dir,
        quant_type=quant_type if not export_all else "F16",
        llama_cpp_path=llama_cpp_path,
        model_name=model_name,
        is_vision_model=is_vl_model,
        original_model_path=Path(original_model_path) if original_model_path else None,
    )

    if export_all:
        # Export all quant types
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Exporting all quant types...", total=None)

            def progress_callback(msg: str):
                progress.update(task, description=msg)

            success, message, output_paths = export_all_quants(
                config=export_config,
                keep_f16=keep_f16,
                progress_callback=progress_callback,
            )

        console.print()
        if success:
            display_success(message)
            console.print(f"\n[bold]Created files:[/bold]")
            for path in output_paths:
                if path.exists():
                    size_gb = path.stat().st_size / (1024**3)
                    console.print(f"  [{THEME['muted']}]{path.name}[/{THEME['muted']}] - [{THEME['accent']}]{size_gb:.2f} GB[/{THEME['accent']}]")

            # Check for mmproj file for VL models
            if is_vl_model:
                for f in output_dir.glob("*mmproj*.gguf"):
                    size_mb = f.stat().st_size / (1024**2)
                    console.print(f"  [{THEME['muted']}]{f.name}[/{THEME['muted']}] - [{THEME['accent']}]{size_mb:.1f} MB[/{THEME['accent']}] (mmproj)")
                    break
        else:
            display_error(message)
    else:
        # Single quant type export
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Exporting to GGUF...", total=None)

            def progress_callback(msg: str):
                progress.update(task, description=msg)

            success, message, output_path = export_to_gguf(
                config=export_config,
                progress_callback=progress_callback,
            )

        console.print()
        if success:
            display_success(message)
            if output_path and output_path.exists():
                size_gb = output_path.stat().st_size / (1024**3)
                console.print(f"  Model size: [{THEME['accent']}]{size_gb:.2f} GB[/{THEME['accent']}]")

            # Check for mmproj file for VL models
            # The --mmproj flag adds "mmproj-" prefix to the output filename
            if is_vl_model:
                mmproj_patterns = [
                    output_dir / f"mmproj-{model_name}-f16.gguf",  # Main pattern from --mmproj flag
                    output_dir / f"{model_name}-mmproj-f16.gguf",  # Alternative naming
                ]
                mmproj_found = False
                for mmproj_path in mmproj_patterns:
                    if mmproj_path.exists():
                        size_mb = mmproj_path.stat().st_size / (1024**2)
                        console.print(f"  mmproj size: [{THEME['accent']}]{size_mb:.1f} MB[/{THEME['accent']}]")
                        console.print(f"  mmproj path: [{THEME['muted']}]{mmproj_path}[/{THEME['muted']}]")
                        mmproj_found = True
                        break

                if not mmproj_found:
                    # Check for any mmproj file in output directory
                    for f in output_dir.glob("*mmproj*.gguf"):
                        size_mb = f.stat().st_size / (1024**2)
                        console.print(f"  mmproj size: [{THEME['accent']}]{size_mb:.1f} MB[/{THEME['accent']}]")
                        console.print(f"  mmproj path: [{THEME['muted']}]{f}[/{THEME['muted']}]")
                        mmproj_found = True
                        break

                if not mmproj_found:
                    console.print(f"  [{THEME['warning']}]mmproj not found - vision features may not work[/{THEME['warning']}]")
                    console.print(f"  [{THEME['muted']}]You may need to manually convert the vision encoder[/{THEME['muted']}]")
        else:
            display_error(message)


# ==============================================================================
# Training Config Management
# ==============================================================================


def run_config_management():
    """Training config management menu."""
    while True:
        console.print(f"\n[bold {THEME['primary']}]Training Configs[/bold {THEME['primary']}]\n")
        console.print(f"[{THEME['muted']}]Configs stored in: {get_configs_dir()}[/{THEME['muted']}]\n")

        action = questionary.select(
            "What would you like to do?",
            choices=[
                questionary.Choice("Create new config", value="create"),
                questionary.Choice("View saved configs", value="view"),
                questionary.Choice("Edit config", value="edit"),
                questionary.Choice("Delete config", value="delete"),
                questionary.Choice("Back", value="back"),
            ],
            style=custom_style,
        ).ask()

        if action == "back" or action is None:
            break

        elif action == "create":
            _create_training_config()

        elif action == "view":
            _view_training_configs()

        elif action == "edit":
            _edit_training_config()

        elif action == "delete":
            _delete_training_config()


def _create_training_config():
    """Create a new training config via questionnaire."""
    console.print(f"\n[bold {THEME['primary']}]Create New Training Config[/bold {THEME['primary']}]\n")

    # Get config name
    name = questionary.text(
        "Config name:",
        style=custom_style,
    ).ask()

    if not name:
        display_warning("Config name required.")
        return

    # Check if exists
    sanitized = sanitize_config_name(name)
    if config_exists(sanitized):
        overwrite = questionary.confirm(
            f"Config '{sanitized}' already exists. Overwrite?",
            default=False,
            style=custom_style,
        ).ask()
        if not overwrite:
            return

    # Get description
    description = questionary.text(
        "Description (optional):",
        default="",
        style=custom_style,
    ).ask() or ""

    # Collect settings via questionnaire
    settings = _collect_config_settings()
    if settings is None:
        return

    # Save config
    success, message = save_training_config(name, settings, description, overwrite=True)
    if success:
        display_success(f"Config '{sanitized}' created successfully!")
    else:
        display_error(message)


def _collect_config_settings() -> Optional[dict]:
    """Collect abliteration settings via questionnaire. Returns None if cancelled."""
    settings = get_default_settings()

    console.print(f"\n[bold {THEME['secondary']}]Basic Settings[/bold {THEME['secondary']}]\n")

    # Number of prompts
    num_prompts = questionary.text(
        "Number of prompts:",
        default=str(settings["num_prompts"]),
        style=custom_style,
    ).ask()
    if num_prompts is None:
        return None
    try:
        settings["num_prompts"] = int(num_prompts)
    except ValueError:
        display_warning("Invalid number, using default.")

    # Direction multiplier
    multiplier = questionary.text(
        "Direction multiplier (0.0-2.0):",
        default=str(settings["direction_multiplier"]),
        style=custom_style,
    ).ask()
    if multiplier is None:
        return None
    try:
        settings["direction_multiplier"] = float(multiplier)
    except ValueError:
        display_warning("Invalid number, using default.")

    # Norm preservation
    settings["norm_preservation"] = questionary.confirm(
        "Enable norm preservation?",
        default=settings["norm_preservation"],
        style=custom_style,
    ).ask()
    if settings["norm_preservation"] is None:
        return None

    # Filter prompts
    settings["filter_prompts"] = questionary.confirm(
        "Filter harmful prompts by refusal?",
        default=settings["filter_prompts"],
        style=custom_style,
    ).ask()
    if settings["filter_prompts"] is None:
        return None

    # Device
    settings["device"] = questionary.select(
        "Device:",
        choices=["cuda", "cpu"],
        default=settings["device"],
        style=custom_style,
    ).ask()
    if settings["device"] is None:
        return None

    # Dtype
    settings["dtype"] = questionary.select(
        "Precision:",
        choices=["float16", "bfloat16", "float32"],
        default=settings["dtype"],
        style=custom_style,
    ).ask()
    if settings["dtype"] is None:
        return None

    # Advanced options
    configure_advanced = questionary.confirm(
        "\nConfigure advanced options?",
        default=False,
        style=custom_style,
    ).ask()

    if configure_advanced:
        console.print(f"\n[bold {THEME['secondary']}]Advanced Settings[/bold {THEME['secondary']}]\n")

        # Winsorization
        settings["use_winsorization"] = questionary.confirm(
            "Enable per-dimension Winsorization?",
            default=settings["use_winsorization"],
            style=custom_style,
        ).ask()
        if settings["use_winsorization"] is None:
            return None

        if settings["use_winsorization"]:
            percentile = questionary.text(
                "Winsorize percentile (0.99-0.999):",
                default=str(settings["winsorize_percentile"]),
                style=custom_style,
            ).ask()
            try:
                settings["winsorize_percentile"] = float(percentile)
            except (ValueError, TypeError):
                pass

        # Magnitude clipping
        settings["use_magnitude_clipping"] = questionary.confirm(
            "Enable global magnitude clipping?",
            default=settings["use_magnitude_clipping"],
            style=custom_style,
        ).ask()
        if settings["use_magnitude_clipping"] is None:
            return None

        if settings["use_magnitude_clipping"]:
            percentile = questionary.text(
                "Magnitude clip percentile:",
                default=str(settings["magnitude_clip_percentile"]),
                style=custom_style,
            ).ask()
            try:
                settings["magnitude_clip_percentile"] = float(percentile)
            except (ValueError, TypeError):
                pass

        # Null-space constraints
        settings["use_null_space"] = questionary.confirm(
            "Enable null-space constraints?",
            default=settings["use_null_space"],
            style=custom_style,
        ).ask()
        if settings["use_null_space"] is None:
            return None

        if settings["use_null_space"]:
            ratio = questionary.text(
                "Null-space rank ratio (0.9-0.99):",
                default=str(settings["null_space_rank_ratio"]),
                style=custom_style,
            ).ask()
            try:
                settings["null_space_rank_ratio"] = float(ratio)
            except (ValueError, TypeError):
                pass

        # Adaptive weighting
        settings["use_adaptive_weighting"] = questionary.confirm(
            "Enable adaptive layer weighting?",
            default=settings["use_adaptive_weighting"],
            style=custom_style,
        ).ask()
        if settings["use_adaptive_weighting"] is None:
            return None

        # Biprojection
        settings["use_biprojection"] = questionary.confirm(
            "Enable biprojection mode?",
            default=settings["use_biprojection"],
            style=custom_style,
        ).ask()
        if settings["use_biprojection"] is None:
            return None

        if settings["use_biprojection"]:
            settings["use_per_neuron_norm"] = questionary.confirm(
                "Use per-neuron norm preservation?",
                default=settings["use_per_neuron_norm"],
                style=custom_style,
            ).ask()

            # Target layer types
            target_layers = questionary.checkbox(
                "Target layer types:",
                choices=[
                    questionary.Choice("o_proj", checked=True),
                    questionary.Choice("down_proj", checked=True),
                    questionary.Choice("gate_proj", checked=False),
                    questionary.Choice("up_proj", checked=False),
                    questionary.Choice("q_proj", checked=False),
                    questionary.Choice("k_proj", checked=False),
                    questionary.Choice("v_proj", checked=False),
                ],
                style=custom_style,
            ).ask()
            if target_layers:
                settings["target_layer_types"] = target_layers

            settings["use_harmless_boundary"] = questionary.confirm(
                "Enable harmless boundary clamping?",
                default=settings["use_harmless_boundary"],
                style=custom_style,
            ).ask()

            if settings["use_harmless_boundary"]:
                ratio = questionary.text(
                    "Harmless clamp ratio (0.0-1.0):",
                    default=str(settings["harmless_clamp_ratio"]),
                    style=custom_style,
                ).ask()
                try:
                    settings["harmless_clamp_ratio"] = float(ratio)
                except (ValueError, TypeError):
                    pass

    # Validate settings
    is_valid, errors = validate_config_settings(settings)
    if not is_valid:
        display_warning("Some settings may be invalid:")
        for error in errors:
            console.print(f"  [{THEME['warning']}]- {error}[/{THEME['warning']}]")

    return settings


def _view_training_configs():
    """View saved training configs."""
    configs = list_configs()

    if not configs:
        console.print(f"\n[{THEME['muted']}]No saved configs found.[/{THEME['muted']}]")
        console.print(f"[{THEME['muted']}]Use 'Create new config' to create one.[/{THEME['muted']}]\n")
        return

    console.print()
    display_training_config_list(configs)

    # Offer to view details
    choices = [
        questionary.Choice(f"{c['name']}", value=c["filename"])
        for c in configs if "error" not in c
    ]
    choices.append(questionary.Choice("Back", value="back"))

    selected = questionary.select(
        "View config details:",
        choices=choices,
        style=custom_style,
    ).ask()

    if selected and selected != "back":
        config = load_training_config(selected)
        if config:
            console.print()
            display_training_config_details(config, f"Config: {config['metadata'].get('name', selected)}")


def _edit_training_config():
    """Edit an existing training config."""
    configs = list_configs()

    if not configs:
        console.print(f"\n[{THEME['muted']}]No saved configs found.[/{THEME['muted']}]\n")
        return

    # Select config to edit
    choices = [
        questionary.Choice(f"{c['name']}", value=c["filename"])
        for c in configs if "error" not in c
    ]
    choices.append(questionary.Choice("Back", value="back"))

    console.print()
    display_training_config_list(configs)

    selected = questionary.select(
        "Select config to edit:",
        choices=choices,
        style=custom_style,
    ).ask()

    if not selected or selected == "back":
        return

    config = load_training_config(selected)
    if not config:
        display_error(f"Failed to load config '{selected}'")
        return

    # Show current config
    console.print()
    display_training_config_details(config, f"Current: {config['metadata'].get('name', selected)}")

    # What to edit?
    edit_action = questionary.select(
        "What would you like to edit?",
        choices=[
            questionary.Choice("Edit description only", value="description"),
            questionary.Choice("Re-configure all settings", value="all"),
            questionary.Choice("Cancel", value="cancel"),
        ],
        style=custom_style,
    ).ask()

    if edit_action == "cancel" or edit_action is None:
        return

    from src.config_manager import update_training_config

    if edit_action == "description":
        new_desc = questionary.text(
            "New description:",
            default=config["metadata"].get("description", ""),
            style=custom_style,
        ).ask()
        if new_desc is not None:
            success, message = update_training_config(selected, description=new_desc)
            if success:
                display_success("Description updated!")
            else:
                display_error(message)

    elif edit_action == "all":
        new_settings = _collect_config_settings()
        if new_settings:
            success, message = update_training_config(selected, settings=new_settings)
            if success:
                display_success("Config updated!")
            else:
                display_error(message)


def _delete_training_config():
    """Delete a training config."""
    configs = list_configs()

    if not configs:
        console.print(f"\n[{THEME['muted']}]No saved configs found.[/{THEME['muted']}]\n")
        return

    # Select config to delete
    choices = [
        questionary.Choice(f"{c['name']}", value=c["filename"])
        for c in configs
    ]
    choices.append(questionary.Choice("Back", value="back"))

    console.print()
    display_training_config_list(configs)

    selected = questionary.select(
        "Select config to delete:",
        choices=choices,
        style=custom_style,
    ).ask()

    if not selected or selected == "back":
        return

    # Load and show config
    config = load_training_config(selected)
    if config:
        console.print()
        display_training_config_details(config, f"Delete: {config['metadata'].get('name', selected)}")

    # Confirm deletion by typing name
    config_name = config["metadata"].get("name", selected) if config else selected
    console.print(f"\n[{THEME['warning']}]Type the config name to confirm deletion:[/{THEME['warning']}]")

    confirmation = questionary.text(
        f"Type '{config_name}' to confirm:",
        style=custom_style,
    ).ask()

    if confirmation == config_name:
        success, message = delete_training_config(selected)
        if success:
            display_success(f"Config '{config_name}' deleted!")
        else:
            display_error(message)
    else:
        console.print(f"[{THEME['muted']}]Deletion cancelled (name didn't match).[/{THEME['muted']}]")


def _select_training_config_for_abliteration() -> Optional[dict]:
    """Select a training config to use for abliteration. Returns config settings or None."""
    configs = list_configs()

    if not configs:
        console.print(f"\n[{THEME['muted']}]No saved configs found.[/{THEME['muted']}]")
        return None

    console.print()
    display_training_config_list(configs)

    # Select config
    choices = [
        questionary.Choice(f"{c['name']} - {c.get('description', '')[:30]}", value=c["filename"])
        for c in configs if "error" not in c
    ]
    choices.append(questionary.Choice("Back (configure manually)", value="back"))

    selected = questionary.select(
        "Select config to use:",
        choices=choices,
        style=custom_style,
    ).ask()

    if not selected or selected == "back":
        return None

    config = load_training_config(selected)
    if not config:
        display_error(f"Failed to load config '{selected}'")
        return None

    # Show config details
    console.print()
    display_training_config_details(config, f"Using: {config['metadata'].get('name', selected)}")

    confirm = questionary.confirm(
        "Use this config?",
        default=True,
        style=custom_style,
    ).ask()

    if confirm:
        return config["settings"]
    return None


def _prompt_save_config_after_abliteration(config: dict) -> None:
    """Prompt user to save abliteration settings as a training config.

    Skips the prompt if the config was loaded from a previously saved config.
    """
    # Skip if config was loaded from a saved config
    if config.get("_loaded_from_saved_config"):
        return

    save_config_prompt = questionary.confirm(
        "\nWould you like to save these settings as a training config?",
        default=False,
        style=custom_style,
    ).ask()

    if not save_config_prompt:
        return

    # Get config name (require non-empty, retry up to 3 times)
    name = None
    for _ in range(3):
        name = questionary.text(
            "Config name:",
            style=custom_style,
            validate=lambda x: len(x.strip()) > 0 or "Config name is required",
        ).ask()

        if name and name.strip():
            name = name.strip()
            break
        elif name is None:
            # User cancelled (Ctrl+C or Esc)
            return

    if not name:
        display_warning("Config name required, not saved.")
        return

    # Get description
    description = questionary.text(
        "Description (optional):",
        default="",
        style=custom_style,
    ).ask() or ""

    # Extract saveable settings
    settings = settings_from_abliteration_config(config)

    # Check if exists
    sanitized = sanitize_config_name(name)
    overwrite = False
    if config_exists(sanitized):
        overwrite = questionary.confirm(
            f"Config '{sanitized}' already exists. Overwrite?",
            default=False,
            style=custom_style,
        ).ask()
        if not overwrite:
            return

    # Save
    success, message = save_training_config(name, settings, description, overwrite=overwrite)
    if success:
        display_success(f"Config saved as '{sanitized}'!")
    else:
        display_error(message)


# ==============================================================================
# Settings Management
# ==============================================================================


def run_settings():
    """Settings management."""
    while True:
        console.print(f"\n[bold {THEME['primary']}]Settings[/bold {THEME['primary']}]\n")

        action = questionary.select(
            "What would you like to do?",
            choices=[
                questionary.Choice("Manage model search directories", value="model_paths"),
                questionary.Choice("Change model output directory", value="output_dir"),
                questionary.Choice("Change eval results directory", value="eval_dir"),
                questionary.Choice("Manage prompts", value="prompts"),
                questionary.Choice("Configure llama.cpp path (GGUF export)", value="llama_cpp"),
                questionary.Choice("View all settings", value="view"),
                questionary.Choice("Reset to defaults", value="reset"),
                questionary.Choice("Clear GPU cache", value="clear_cache"),
                questionary.Choice("Back", value="back"),
            ],
            style=custom_style,
        ).ask()

        if action == "back" or action is None:
            break

        elif action == "model_paths":
            _manage_model_paths()

        elif action == "output_dir":
            _manage_output_dir()

        elif action == "eval_dir":
            _manage_eval_results_dir()

        elif action == "prompts":
            _manage_prompts()

        elif action == "llama_cpp":
            _manage_llama_cpp_path()

        elif action == "view":
            config = load_config()
            console.print(f"\n[bold]Config file:[/bold] {get_config_path()}\n")
            display_config_panel(config, "Current Settings")

        elif action == "reset":
            confirm = questionary.confirm(
                "Reset all settings to defaults?",
                default=False,
                style=custom_style,
            ).ask()
            if confirm:
                from src.cli_components import get_default_config
                save_config(get_default_config())
                display_success("Settings reset to defaults!")

        elif action == "clear_cache":
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            display_success("GPU cache cleared!")


def _manage_model_paths():
    """Manage model search directories."""
    while True:
        paths = get_model_paths()

        console.print(f"\n[bold {THEME['primary']}]Model Search Directories[/bold {THEME['primary']}]\n")

        # Display current paths
        from rich.table import Table
        table = Table(show_header=True, header_style=f"bold {THEME['primary']}")
        table.add_column("#", style="dim", width=4)
        table.add_column("Path", style=THEME["primary"])
        table.add_column("Status", style=THEME["muted"])

        for idx, path in enumerate(paths, 1):
            exists = Path(path).exists()
            status = "[green]exists[/green]" if exists else "[red]not found[/red]"
            table.add_row(str(idx), path, status)

        console.print(table)
        console.print()

        action = questionary.select(
            "What would you like to do?",
            choices=[
                questionary.Choice("Add a directory", value="add"),
                questionary.Choice("Remove a directory", value="remove"),
                questionary.Choice("Back", value="back"),
            ],
            style=custom_style,
        ).ask()

        if action == "back" or action is None:
            break

        elif action == "add":
            new_path = questionary.path(
                "Enter directory path:",
                only_directories=True,
                style=custom_style,
            ).ask()

            if new_path:
                if add_model_path(new_path):
                    display_success(f"Added: {new_path}")
                else:
                    display_warning("Path already in list.")

        elif action == "remove":
            if not paths:
                display_warning("No paths to remove.")
                continue

            choices = [questionary.Choice(p, value=p) for p in paths]
            choices.append(questionary.Choice("Cancel", value=None))

            to_remove = questionary.select(
                "Select path to remove:",
                choices=choices,
                style=custom_style,
            ).ask()

            if to_remove:
                if remove_model_path(to_remove):
                    display_success(f"Removed: {to_remove}")
                else:
                    display_error("Failed to remove path.")


def _manage_output_dir():
    """Manage default output directory for abliterated models."""
    current_dir = get_default_output_dir()

    console.print(f"\n[bold {THEME['primary']}]Model Output Directory[/bold {THEME['primary']}]\n")
    console.print(f"Current directory: [{THEME['primary']}]{current_dir}[/{THEME['primary']}]")
    console.print(f"[{THEME['muted']}]Abliterated models will be saved here by default.[/{THEME['muted']}]")

    exists = Path(current_dir).exists()
    status = f"[green]exists[/green]" if exists else f"[yellow]will be created[/yellow]"
    console.print(f"Status: {status}\n")

    change = questionary.confirm(
        "Change the model output directory?",
        default=False,
        style=custom_style,
    ).ask()

    if change:
        new_path = questionary.path(
            "Enter new model output directory:",
            default=current_dir,
            style=custom_style,
        ).ask()

        if new_path:
            set_default_output_dir(new_path)
            display_success(f"Model output directory set to: {new_path}")

            # Ask if they want to add this to model search paths
            add_to_search = questionary.confirm(
                "Add this directory to model search paths?",
                default=True,
                style=custom_style,
            ).ask()

            if add_to_search:
                if add_model_path(new_path):
                    display_success(f"Added to model search paths: {new_path}")
                else:
                    console.print(f"[{THEME['muted']}]Already in model search paths.[/{THEME['muted']}]")


def _manage_eval_results_dir():
    """Manage evaluation results directory."""
    current_dir = get_eval_results_dir()

    console.print(f"\n[bold {THEME['primary']}]Evaluation Results Directory[/bold {THEME['primary']}]\n")
    console.print(f"Current directory: [{THEME['primary']}]{current_dir}[/{THEME['primary']}]")

    exists = Path(current_dir).exists()
    status = f"[green]exists[/green]" if exists else f"[yellow]will be created[/yellow]"
    console.print(f"Status: {status}\n")

    change = questionary.confirm(
        "Change the evaluation results directory?",
        default=False,
        style=custom_style,
    ).ask()

    if change:
        new_path = questionary.path(
            "Enter new evaluation results directory:",
            default=current_dir,
            style=custom_style,
        ).ask()

        if new_path:
            set_eval_results_dir(new_path)
            display_success(f"Eval results directory set to: {new_path}")


def _manage_prompts():
    """Manage prompts directory and files."""
    prompts_dir = get_user_prompts_dir()

    console.print(f"\n[bold {THEME['primary']}]Prompts Management[/bold {THEME['primary']}]\n")
    console.print(f"Prompts directory: [{THEME['primary']}]{prompts_dir}[/{THEME['primary']}]")

    # Check status
    if user_prompts_exist():
        console.print(f"Status: [green]configured[/green]\n")

        # List prompt files
        from rich.table import Table
        table = Table(show_header=True, header_style=f"bold {THEME['primary']}")
        table.add_column("File", style=THEME["primary"])
        table.add_column("Lines", justify="right")
        table.add_column("Description", style=THEME["muted"])

        descriptions = {
            "harmful.txt": "Prompts that should be refused (before abliteration)",
            "harmless.txt": "Prompts that should be answered normally",
            "preservation.txt": "Prompts for null-space capability preservation",
        }

        for prompt_file in sorted(prompts_dir.glob("*.txt")):
            try:
                line_count = sum(1 for line in open(prompt_file, encoding="utf-8") if line.strip())
            except Exception:
                line_count = "?"
            desc = descriptions.get(prompt_file.name, "Custom prompts file")
            table.add_row(prompt_file.name, str(line_count), desc)

        console.print(table)
    else:
        console.print(f"Status: [yellow]not configured[/yellow]\n")

    console.print()

    action = questionary.select(
        "What would you like to do?",
        choices=[
            questionary.Choice("Reset prompts to defaults", value="reset"),
            questionary.Choice("Open prompts directory", value="open"),
            questionary.Choice("Back", value="back"),
        ],
        style=custom_style,
    ).ask()

    if action == "back" or action is None:
        return

    elif action == "reset":
        confirm = questionary.confirm(
            "Reset all prompts to package defaults? (This will overwrite your changes)",
            default=False,
            style=custom_style,
        ).ask()

        if confirm:
            if copy_prompts_to_user_dir(force=True):
                display_success(f"Prompts reset to defaults in: {prompts_dir}")
            else:
                display_error("Could not reset prompts (package prompts not found)")

    elif action == "open":
        # Try to open the directory in file explorer
        import subprocess
        import platform

        prompts_dir.mkdir(parents=True, exist_ok=True)

        try:
            if platform.system() == "Windows":
                subprocess.run(["explorer", str(prompts_dir)], check=False)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", str(prompts_dir)], check=False)
            else:  # Linux
                subprocess.run(["xdg-open", str(prompts_dir)], check=False)
            display_success(f"Opened: {prompts_dir}")
        except Exception as e:
            display_warning(f"Could not open directory: {e}\n\nPath: {prompts_dir}")


def _manage_llama_cpp_path():
    """Manage llama.cpp installation path for GGUF export."""
    config = load_config()
    current_path = config.get("llama_cpp_path", "")

    console.print(f"\n[bold {THEME['primary']}]llama.cpp Path (GGUF Export)[/bold {THEME['primary']}]\n")

    if current_path:
        console.print(f"Current path: [{THEME['primary']}]{current_path}[/{THEME['primary']}]")
        exists = Path(current_path).exists()
        status = f"[green]exists[/green]" if exists else f"[red]not found[/red]"
        console.print(f"Status: {status}")
    else:
        console.print(f"[{THEME['muted']}]No path configured (will search common locations)[/{THEME['muted']}]")

    # Check current tool detection
    tools_status = check_tools_available(Path(current_path) if current_path else None)
    console.print()

    if tools_status["can_convert"]:
        console.print(f"[green]✓[/green] Convert script found: {tools_status['convert_script']}")
    else:
        console.print(f"[red]✗[/red] Convert script not found")

    if tools_status["can_quantize"]:
        console.print(f"[green]✓[/green] Quantize tool found: {tools_status['quantize_exe']}")
    else:
        console.print(f"[yellow]![/yellow] Quantize tool not found (only F16/F32 export available)")

    console.print()

    action = questionary.select(
        "What would you like to do?",
        choices=[
            questionary.Choice("Set llama.cpp path", value="set"),
            questionary.Choice("Auto-detect llama.cpp", value="auto"),
            questionary.Choice("Clear path (use auto-detection)", value="clear"),
            questionary.Choice("Back", value="back"),
        ],
        style=custom_style,
    ).ask()

    if action == "back" or action is None:
        return

    elif action == "set":
        new_path = questionary.path(
            "Enter path to llama.cpp directory:",
            default=current_path or "",
            style=custom_style,
        ).ask()

        if new_path:
            new_path = str(Path(new_path).resolve())
            config["llama_cpp_path"] = new_path
            save_config(config)

            # Verify
            tools = check_tools_available(Path(new_path))
            if tools["can_convert"]:
                display_success(f"llama.cpp path set to: {new_path}")
            else:
                display_warning(f"Path set, but convert_hf_to_gguf.py not found in {new_path}")

    elif action == "auto":
        detected = find_llama_cpp_path()
        if detected:
            config["llama_cpp_path"] = str(detected)
            save_config(config)
            display_success(f"Auto-detected llama.cpp at: {detected}")
        else:
            display_warning(
                "Could not auto-detect llama.cpp installation.\n"
                "Try setting the path manually or set LLAMA_CPP_PATH environment variable."
            )

    elif action == "clear":
        if "llama_cpp_path" in config:
            del config["llama_cpp_path"]
            save_config(config)
        display_success("Path cleared. Will use auto-detection.")


def is_first_run() -> bool:
    """Check if this is the first time running the CLI."""
    return not get_config_path().exists()


def run_first_time_setup():
    """First-time setup walkthrough for new users."""
    clear_screen()
    display_banner()

    from rich.panel import Panel

    console.print(Panel(
        "[bold]Welcome to the Abliteration Toolkit![/bold]\n\n"
        "This appears to be your first time running the CLI.\n"
        "Let's set up your configuration.",
        title="[bold cyan]First-Time Setup[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()

    # Ask about model directories
    console.print(f"[bold {THEME['primary']}]Step 1: Model Directories[/bold {THEME['primary']}]\n")
    console.print("The toolkit needs to know where to find your models.")
    console.print(f"[{THEME['muted']}]Default locations: D:/models, C:/models, HuggingFace cache[/{THEME['muted']}]\n")

    add_custom = questionary.confirm(
        "Would you like to add a custom model directory?",
        default=True,
        style=custom_style,
    ).ask()

    custom_paths = []
    while add_custom:
        path = questionary.path(
            "Enter model directory path:",
            only_directories=True,
            style=custom_style,
        ).ask()

        if path and Path(path).exists():
            custom_paths.append(str(Path(path).resolve()))
            display_success(f"Added: {path}")
        elif path:
            display_warning(f"Directory not found: {path}")

        add_custom = questionary.confirm(
            "Add another directory?",
            default=False,
            style=custom_style,
        ).ask()

    # Build initial config
    from src.cli_components import get_default_config
    config = get_default_config()

    # Prepend custom paths
    if custom_paths:
        config["model_paths"] = custom_paths + config["model_paths"]

    # Default output directory
    console.print(f"\n[bold {THEME['primary']}]Step 2: Default Output Directory[/bold {THEME['primary']}]\n")

    use_default_output = questionary.confirm(
        f"Use default output directory? ({config['default_output_dir']})",
        default=True,
        style=custom_style,
    ).ask()

    if not use_default_output:
        output_dir = questionary.path(
            "Enter default output directory:",
            only_directories=True,
            style=custom_style,
        ).ask()
        if output_dir:
            config["default_output_dir"] = output_dir

    # Eval results directory
    console.print(f"\n[bold {THEME['primary']}]Step 3: Evaluation Results Directory[/bold {THEME['primary']}]\n")
    console.print(f"[{THEME['muted']}]All refusal evaluations will be logged to this directory.[/{THEME['muted']}]\n")

    use_default_eval = questionary.confirm(
        f"Use default eval results directory? ({config['eval_results_dir']})",
        default=True,
        style=custom_style,
    ).ask()

    if not use_default_eval:
        eval_dir = questionary.path(
            "Enter evaluation results directory:",
            only_directories=True,
            style=custom_style,
        ).ask()
        if eval_dir:
            config["eval_results_dir"] = eval_dir

    # Default precision
    console.print(f"\n[bold {THEME['primary']}]Step 4: Default Precision[/bold {THEME['primary']}]\n")

    config["default_dtype"] = questionary.select(
        "Select default precision for abliteration:",
        choices=[
            questionary.Choice("float16 (faster, less memory)", value="float16"),
            questionary.Choice("bfloat16 (better precision)", value="bfloat16"),
            questionary.Choice("float32 (full precision)", value="float32"),
        ],
        default="float16",
        style=custom_style,
    ).ask()

    # Save config
    save_config(config)

    # Copy prompts to user directory
    console.print(f"\n[bold {THEME['primary']}]Step 5: Setting Up Prompts[/bold {THEME['primary']}]\n")
    console.print(f"[{THEME['muted']}]Copying default prompts to your config directory...[/{THEME['muted']}]")
    console.print(f"[{THEME['muted']}]You can edit these prompts to customize abliteration behavior.[/{THEME['muted']}]\n")

    if copy_prompts_to_user_dir():
        prompts_dir = get_user_prompts_dir()
        display_success(f"Prompts copied to: {prompts_dir}")
        console.print(f"\n[{THEME['muted']}]Files:[/{THEME['muted']}]")
        console.print(f"  [{THEME['primary']}]harmful.txt[/{THEME['primary']}]     - Prompts that should be refused (before abliteration)")
        console.print(f"  [{THEME['primary']}]harmless.txt[/{THEME['primary']}]   - Prompts that should be answered normally")
        console.print(f"  [{THEME['primary']}]preservation.txt[/{THEME['primary']}] - Prompts for null-space capability preservation")
    else:
        display_warning("Could not copy prompts (package prompts not found)")

    console.print()
    display_success(f"Configuration saved to: {get_config_path()}")
    console.print(f"\n[{THEME['muted']}]You can modify these settings anytime from the Settings menu.[/{THEME['muted']}]\n")

    questionary.press_any_key_to_continue(
        "Press any key to continue to the main menu...",
        style=custom_style,
    ).ask()


def main_menu():
    """Main interactive menu loop."""
    # Check for first-time setup
    if is_first_run():
        run_first_time_setup()
    while True:
        clear_screen()
        display_banner()
        display_system_info()

        display_menu(
            "Main Menu",
            [
                ("1", "Abliterate Model", "Remove refusal behavior from a model"),
                ("2", "J-space Abliteration", "Jacobian Lens: map / restrict / iterative"),
                ("3", "Test Model", "Run test prompts on a model"),
                ("4", "Compare Models", "Side-by-side comparison"),
                ("5", "Evaluate Refusal", "Full refusal rate evaluation"),
                ("6", "Quality Eval", "Perplexity & KL divergence comparison"),
                ("7", "Export to GGUF", "Quantize for llama.cpp"),
                ("8", "Manage Configs", "Create, edit, delete training configs"),
                ("9", "Settings", "Configure options"),
            ]
        )

        choice = questionary.text(
            "",
            style=custom_style,
        ).ask()

        if choice is None or choice.lower() == "q":
            console.print(f"\n[{THEME['muted']}]Goodbye![/{THEME['muted']}]\n")
            break

        if choice == "1":
            config = get_abliteration_config()
            if config:
                console.print()
                display_config_panel(config, "Abliteration Configuration")
                confirm = questionary.confirm(
                    "\nProceed with abliteration?",
                    default=True,
                    style=custom_style,
                ).ask()
                if confirm:
                    run_abliteration(config)
                    # Prompt to save config after successful abliteration
                    _prompt_save_config_after_abliteration(config)
                    questionary.press_any_key_to_continue(style=custom_style).ask()

        elif choice == "2":
            run_jspace_workflow()
            questionary.press_any_key_to_continue(style=custom_style).ask()

        elif choice == "3":
            run_test_model()
            questionary.press_any_key_to_continue(style=custom_style).ask()

        elif choice == "4":
            run_compare_models()
            questionary.press_any_key_to_continue(style=custom_style).ask()

        elif choice == "5":
            run_evaluation()
            questionary.press_any_key_to_continue(style=custom_style).ask()

        elif choice == "6":
            run_quality_eval()
            questionary.press_any_key_to_continue(style=custom_style).ask()

        elif choice == "7":
            run_gguf_export()
            questionary.press_any_key_to_continue(style=custom_style).ask()

        elif choice == "8":
            run_config_management()
            questionary.press_any_key_to_continue(style=custom_style).ask()

        elif choice == "9":
            run_settings()
            questionary.press_any_key_to_continue(style=custom_style).ask()


# Click CLI for batch mode
@click.command()
@click.option("--batch", is_flag=True, help="Run in batch mode (non-interactive)")
@click.option("--jlens-map", "jlens_map_mode", is_flag=True, help="Generate a layer_target_map.json from J-lens (Jacobian Lens) refusal-concept vectors instead of running abliteration")
@click.option("--jlens-iterative", "jlens_iterative_mode", is_flag=True, help="Run J-lens-guided iterative abliteration: recompute vectors + apply a small step per iteration until target refusal rate is reached")
@click.option("--model_path", "-m", type=str, help="Path to input model")
@click.option("--output_path", "-o", type=str, help="Path for output model (uses config default_output_dir if not specified)")
@click.option("--num_prompts", type=int, default=None, help="Number of prompts to use (default: from config)")
@click.option("--direction_multiplier", type=float, default=None, help="Ablation strength (default: from config)")
@click.option("--no_norm_preservation", is_flag=True, help="Disable norm preservation")
@click.option("--no_filter_prompts", is_flag=True, help="Don't filter prompts by refusal")
@click.option("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
@click.option("--dtype", type=click.Choice(["float16", "bfloat16", "float32"]), default=None, help="Precision (default: from config)")
# Advanced options
@click.option("--winsorize/--no-winsorize", default=False, help="Enable Winsorization (clips outliers)")
@click.option("--winsorize-percentile", type=float, default=0.995, help="Winsorize percentile (0.99-0.999)")
@click.option("--magnitude-clip/--no-magnitude-clip", default=False, help="Enable global magnitude clipping (alternative to Winsorization)")
@click.option("--magnitude-clip-percentile", type=float, default=0.99, help="Magnitude clip percentile (0.9-0.999)")
# Numerical stability
@click.option("--welford-mean/--no-welford-mean", default=True, help="Use Welford's algorithm for streaming mean")
@click.option("--float64-subtraction/--no-float64-subtraction", default=True, help="Use float64 for mean subtraction")
@click.option("--null-space/--no-null-space", default=False, help="Enable null-space constraints")
@click.option("--preservation-prompts", type=str, default=None, help="Path to preservation prompts file")
@click.option("--null-space-rank-ratio", type=float, default=0.95, help="Null-space SVD rank ratio (0.9-0.99)")
@click.option("--adaptive-weighting/--no-adaptive-weighting", default=False, help="Enable adaptive layer weighting")

@click.option("--projected/--no-projected", default=True, help="Orthogonalize refusal against harmless direction (recommended)")
@click.option("--biprojection/--no-biprojection", default=False, help="Enable biprojection mode")
@click.option("--per-neuron-norm/--frobenius-norm", default=False, help="Use per-neuron norm preservation")
@click.option("--target-layers", type=str, default=None, help="Comma-separated layer types (e.g., 'o_proj,down_proj')")
@click.option("--harmless-boundary/--no-harmless-boundary", default=False, help="Clamp ablation to preserve harmless direction")
@click.option("--harmless-clamp-ratio", type=float, default=0.1, help="Harmless boundary clamp ratio (0.0-1.0)")
@click.option("--num-measurement-layers", type=int, default=2, help="Number of high-quality layers for biprojection")
@click.option("--intervention-start", type=float, default=0.25, help="Intervention range start (0.0-1.0)")
@click.option("--intervention-end", type=float, default=0.95, help="Intervention range end (0.0-1.0)")
@click.option("--layer-target-map", type=str, default=None, help="Path to layer_target_map.json for data-driven layer targeting")
@click.option("--unmapped-layer-behavior", type=click.Choice(["skip", "default"]), default="skip", help="How to handle layers not in target map")
# Dynamic layer targeting
@click.option("--dynamic-layer-targeting/--no-dynamic-layer-targeting", default=False, help="Extract from ALL layers, apply per-layer directions and null-space projectors")
# Hybrid architecture
@click.option("--hybrid-strategy", type=click.Choice(["auto", "uniform"]), default="auto", help="Hybrid architecture strategy: auto (recommended) or uniform (legacy)")
@click.option("--hybrid-full-attn-weight", type=float, default=1.0, help="Ablation weight for full attention layers (0.0-2.0)")
@click.option("--hybrid-linear-attn-weight", type=float, default=0.4, help="Ablation weight for linear attention layers (0.0-1.0)")
@click.option("--hybrid-skip-recurrent/--no-hybrid-skip-recurrent", default=True, help="Skip in_proj_a/in_proj_b (recurrent dynamics) during ablation")
@click.option("--hybrid-skip-state/--no-hybrid-skip-state", default=False, help="Also skip in_proj_qkv/in_proj_z (more conservative)")
# KL divergence monitoring
@click.option("--kl-monitor/--no-kl-monitor", default=False, help="Report KL divergence after abliteration")
@click.option("--kl-reference-prompts", type=str, default=None, help="Path to reference prompts for KL (default: preservation.txt)")
@click.option("--kl-num-prompts", type=int, default=50, help="Number of reference prompts for KL")
@click.option("--kl-top-k", type=int, default=200, help="Top-k tokens for KL approximation")
@click.option("--kl-auto-tune/--no-kl-auto-tune", default=False, help="Binary search for best multiplier within KL budget")
@click.option("--kl-threshold", type=float, default=0.5, help="Max mean KL divergence (nats) for auto-tune")
@click.option("--kl-search-min", type=float, default=0.1, help="Auto-tune search range minimum")
@click.option("--kl-search-max", type=float, default=2.0, help="Auto-tune search range maximum")
# Gabliteration: multi-directional SVD abliteration
@click.option("--gabliteration/--no-gabliteration", default=False, help="Enable Gabliteration multi-directional SVD abliteration")
@click.option("--gab-num-directions", type=int, default=2, help="Number of SVD directions (1-10, paper recommends 1-3)")
@click.option("--gab-ridge-lambda", type=float, default=0.1, help="Ridge regularization for projection matrix")
@click.option("--gab-layer-beta", type=float, default=0.5, help="Layer scaling curvature (0=uniform)")
@click.option("--gab-skip-first", type=int, default=2, help="Skip first N layers")
@click.option("--gab-skip-last", type=int, default=2, help="Skip last N layers")
# J-lens (Jacobian Lens) options — Phase 1 (--jlens-map mode) and Phase 2 (--jlens-restrict during abliteration)
@click.option("--jlens-vectors", type=str, default=None, help="Path to jlens_vectors.pt (cached J-lens vectors). If unset, computed inline for the current model.")
@click.option("--jlens-restrict/--no-jlens-restrict", default=False, help="Restrict refusal direction to the J-space (Jacobian Lens) subspace before ablation")
@click.option("--jlens-basis-rank", type=int, default=16, help="Max rank of the J-space basis used for restriction (default: 16)")
@click.option("--jlens-min-projection-ratio", type=float, default=0.1, help="If J-space projection retains < ratio of the original norm, fall back to unrestricted direction (default: 0.1)")
@click.option("--jlens-concepts", type=str, default=None, help="Comma-separated concept strings (default: refusal anchors like 'cannot,sorry,unable,refuse')")
@click.option("--jlens-num-prompts", type=int, default=32, help="Number of harmful prompts to average over for J-lens (default: 32)")
@click.option("--jlens-batch-size", type=int, default=2, help="Batch size for J-lens VJP passes (small for VRAM)")
@click.option("--jlens-max-seq-len", type=int, default=64, help="Max sequence length per J-lens forward pass (default: 64)")
@click.option("--jlens-grad-checkpoint/--no-jlens-grad-checkpoint", default=False, help="Enable gradient checkpointing during J-lens (slower but lower VRAM)")
@click.option("--jlens-exclude-threshold", type=float, default=0.2, help="Layers below this normalized signal are excluded from the target map (default: 0.2)")
@click.option("--jlens-min-multiplier", type=float, default=0.1, help="Minimum layer multiplier in the generated target map (default: 0.1)")
# --jlens-iterative mode sub-flags
@click.option("--jlens-max-iterations", type=int, default=10, help="Max iterations for --jlens-iterative (default: 10)")
@click.option("--jlens-target-refusal-rate", type=float, default=0.05, help="Target refusal rate for --jlens-iterative convergence (default: 0.05)")
@click.option("--jlens-step-multiplier", type=float, default=1.0, help="Per-iteration ablation strength (initial step) for --jlens-iterative. Default 1.0 = iteration 1 matches a single restrict-mode pass; later iterations re-mine J-lens on the modified model.")
@click.option("--jlens-step-escalation/--no-jlens-step-escalation", default=True, help="Grow step multiplier when plateauing far from target (default: on)")
@click.option("--jlens-max-step-multiplier", type=float, default=1.0, help="Max step multiplier when escalating (default: 1.0 = restrict-mode default)")
@click.option("--jlens-abl-num-prompts", type=int, default=32, help="Number of prompts for direction extraction in --jlens-iterative (matches --num_prompts; default: 32)")
@click.option("--jlens-filter-prompts/--no-jlens-filter-prompts", default=True, help="Filter harmful prompts by actual refusal before extraction (matches restrict mode; default: on)")
@click.option("--jlens-eval-prompt-count", type=int, default=50, help="Number of prompts to evaluate refusal rate per iteration (default: 50)")
@click.option("--jlens-refusal-threshold", type=float, default=-7.0, help="Log-likelihood detector threshold (default: -7.0). Run `python -m utils.test_abliteration audit` to find the right value for your model — the default is wildly wrong for models like Qwen3.5.")
@click.option("--jlens-auto-calibrate-threshold/--no-jlens-auto-calibrate-threshold", default=False, help="Run a mini-audit against actual generations at loop start and pick a threshold that maximizes agreement with the ground-truth classifier. Adds ~1-2 min startup on 9B models.")
@click.option("--jlens-iter-track-kl/--no-jlens-iter-track-kl", default=True, help="Cache reference logits + compute KL(base||current) each iteration as a diagnostic (default: on). Turn off to save memory/time on very large models.")
@click.option("--jlens-iter-kl-guardrail/--no-jlens-iter-kl-guardrail", default=False, help="Roll back an iteration if KL from original exceeds --jlens-iter-kl-threshold (implies --jlens-iter-track-kl)")
@click.option("--jlens-iter-kl-threshold", type=float, default=0.5, help="KL nats threshold for --jlens-iter-kl-guardrail (default: 0.5)")
# Contrastive concept mining (recommended: replaces the fixed refusal-concept list)
@click.option("--jlens-mine-concepts/--no-jlens-mine-concepts", default=False, help="Auto-discover refusal concepts from the model's own next-token distribution (default: off; overrides --jlens-concepts when on)")
@click.option("--jlens-mine-top-k", type=int, default=8, help="Number of mined concepts to keep (default: 8)")
@click.option("--jlens-mine-min-specificity", type=float, default=0.005, help="Minimum score for a mined concept (default: 0.005; use ~0.001 for consensus mode)")
@click.option("--jlens-mining-mode", type=click.Choice(["consensus", "contrast"]), default="consensus", help="Mining strategy: consensus (mode-based, recommended) or contrast (P(harmful)-P(harmless))")
def cli(batch, jlens_map_mode, jlens_iterative_mode, model_path, output_path, num_prompts, direction_multiplier,
        no_norm_preservation, no_filter_prompts, device, dtype,
        winsorize, winsorize_percentile,
        magnitude_clip, magnitude_clip_percentile,
        welford_mean, float64_subtraction,
        null_space, preservation_prompts,
        null_space_rank_ratio, adaptive_weighting,
        projected, biprojection, per_neuron_norm, target_layers, harmless_boundary,
        harmless_clamp_ratio, num_measurement_layers, intervention_start, intervention_end,
        layer_target_map, unmapped_layer_behavior, dynamic_layer_targeting,
        hybrid_strategy, hybrid_full_attn_weight, hybrid_linear_attn_weight,
        hybrid_skip_recurrent, hybrid_skip_state,
        kl_monitor, kl_reference_prompts, kl_num_prompts, kl_top_k,
        kl_auto_tune, kl_threshold, kl_search_min, kl_search_max,
        gabliteration, gab_num_directions, gab_ridge_lambda,
        gab_layer_beta, gab_skip_first, gab_skip_last,
        jlens_vectors, jlens_restrict, jlens_basis_rank, jlens_min_projection_ratio,
        jlens_concepts, jlens_num_prompts, jlens_batch_size, jlens_max_seq_len,
        jlens_grad_checkpoint, jlens_exclude_threshold, jlens_min_multiplier,
        jlens_max_iterations, jlens_target_refusal_rate, jlens_step_multiplier,
        jlens_step_escalation, jlens_max_step_multiplier,
        jlens_abl_num_prompts, jlens_filter_prompts,
        jlens_refusal_threshold, jlens_auto_calibrate_threshold,
        jlens_eval_prompt_count, jlens_iter_track_kl, jlens_iter_kl_guardrail, jlens_iter_kl_threshold,
        jlens_mine_concepts, jlens_mine_top_k, jlens_mine_min_specificity, jlens_mining_mode):
    """
    Abliteration Toolkit - Remove refusal behavior from language models.

    Run without arguments for interactive mode, or use --batch for automation.

    Advanced options enable research-backed enhancements:
    \b
      --projected              Orthogonalize refusal against harmless direction (default: on)
      --winsorize              Clip outlier activations (recommended for Gemma models)
      --magnitude-clip         Global magnitude clipping (alternative to Winsorization)
      --null-space             Preserve model capabilities using null-space constraints
      --preservation-prompts   Custom prompts for null-space computation
      --adaptive-weighting     Per-layer adaptive ablation strength
      --no-welford-mean        Disable Welford's streaming mean (default: on)
      --no-float64-subtraction Disable float64 mean subtraction (default: on)

    Dynamic layer targeting (per-layer precision):
    \b
      --dynamic-layer-targeting  Extract activations from ALL layers and apply
                                 per-layer refusal directions and null-space
                                 projectors (instead of using averaged direction)

    Biprojection options (improved NatInt preservation):
    \b
      --biprojection           Enable biprojection mode (measure at high-quality layers)
      --per-neuron-norm        Per-neuron norm preservation instead of Frobenius
      --target-layers          Only ablate specific layer types (e.g., 'o_proj,down_proj')
      --harmless-boundary      Clamp ablation to preserve harmless direction

    Hybrid architecture (Qwen3.5, etc.):
    \b
      --hybrid-strategy        auto (detect+apply, default) or uniform (legacy)
      --hybrid-full-attn-weight   Ablation weight for full attention layers (default: 1.0)
      --hybrid-linear-attn-weight Ablation weight for linear attention layers (default: 0.4)
      --hybrid-skip-recurrent  Skip in_proj_a/in_proj_b during ablation (default: on)

    KL divergence monitoring (capability drift):
    \b
      --kl-monitor             Report KL divergence after abliteration
      --kl-auto-tune           Binary search for best multiplier within KL budget
      --kl-threshold FLOAT     Max mean KL divergence for auto-tune (default: 0.5)

    Gabliteration (multi-directional SVD, arxiv 2512.18901):
    \b
      --gabliteration          Enable multi-directional SVD abliteration
      --gab-num-directions INT Number of SVD directions (default: 2)
      --gab-ridge-lambda FLOAT Ridge regularization (default: 0.1)
      --gab-layer-beta FLOAT   Layer scaling curvature (default: 0.5)
      --gab-skip-first INT     Skip first N layers (default: 2)
      --gab-skip-last INT      Skip last N layers (default: 2)
    """
    if jlens_iterative_mode:
        if not model_path:
            console.print("[red]Error: --model_path is required in --jlens-iterative mode[/red]")
            sys.exit(1)
        if not output_path:
            console.print("[red]Error: --output_path is required in --jlens-iterative mode[/red]")
            sys.exit(1)
        parsed_concepts = None
        if jlens_concepts:
            parsed_concepts = [c.strip() for c in jlens_concepts.split(",") if c.strip()]
        effective_dtype = dtype if dtype is not None else get_default_dtype()
        display_banner()
        success = run_jlens_iterative_mode(
            model_path=model_path,
            output_path=output_path,
            device=device,
            dtype=effective_dtype,
            max_iterations=jlens_max_iterations,
            target_refusal_rate=jlens_target_refusal_rate,
            step_multiplier=jlens_step_multiplier,
            max_step_multiplier=jlens_max_step_multiplier,
            step_escalation=jlens_step_escalation,
            eval_prompt_count=jlens_eval_prompt_count,
            track_kl=jlens_iter_track_kl,
            use_kl_guardrail=jlens_iter_kl_guardrail,
            kl_threshold=jlens_iter_kl_threshold,
            kl_reference_prompts=kl_reference_prompts,
            concepts=parsed_concepts,
            abliteration_num_prompts=jlens_abl_num_prompts,
            filter_prompts=jlens_filter_prompts,
            refusal_threshold=jlens_refusal_threshold,
            auto_calibrate_threshold=jlens_auto_calibrate_threshold,
            num_prompts=jlens_num_prompts,
            batch_size=jlens_batch_size,
            max_seq_len=jlens_max_seq_len,
            grad_checkpoint=jlens_grad_checkpoint,
            basis_rank=jlens_basis_rank,
            min_projection_ratio=jlens_min_projection_ratio,
            mine_concepts=jlens_mine_concepts,
            mine_top_k=jlens_mine_top_k,
            mine_min_specificity=jlens_mine_min_specificity,
            mining_mode=jlens_mining_mode,
        )
        sys.exit(0 if success else 1)

    if jlens_map_mode:
        if not model_path:
            console.print("[red]Error: --model_path is required in --jlens-map mode[/red]")
            sys.exit(1)
        if not output_path:
            console.print("[red]Error: --output_path is required in --jlens-map mode (target-map JSON destination)[/red]")
            sys.exit(1)
        parsed_concepts = None
        if jlens_concepts:
            parsed_concepts = [c.strip() for c in jlens_concepts.split(",") if c.strip()]
        effective_dtype = dtype if dtype is not None else get_default_dtype()
        display_banner()
        success = run_jlens_map_generation(
            model_path=model_path,
            output_path=output_path,
            concepts=parsed_concepts,
            num_prompts=jlens_num_prompts,
            batch_size=jlens_batch_size,
            max_seq_len=jlens_max_seq_len,
            grad_checkpoint=jlens_grad_checkpoint,
            basis_rank=jlens_basis_rank,
            exclude_threshold=jlens_exclude_threshold,
            min_multiplier=jlens_min_multiplier,
            device=device,
            dtype=effective_dtype,
            mine_concepts=jlens_mine_concepts,
            mine_top_k=jlens_mine_top_k,
            mine_min_specificity=jlens_mine_min_specificity,
            mining_mode=jlens_mining_mode,
        )
        sys.exit(0 if success else 1)

    if batch:
        if not model_path:
            console.print("[red]Error: --model_path is required in batch mode[/red]")
            sys.exit(1)

        # Use config defaults for any unspecified values
        effective_num_prompts = num_prompts if num_prompts is not None else get_default_num_prompts()
        effective_multiplier = direction_multiplier if direction_multiplier is not None else get_default_direction_multiplier()
        effective_dtype = dtype if dtype is not None else get_default_dtype()

        # Construct output path from config if not specified
        if output_path:
            effective_output_path = output_path
        else:
            output_dir = get_default_output_dir()
            model_name = Path(model_path).name
            effective_output_path = f"{output_dir}/{model_name}-abliterated"

        # Parse target layers if provided
        parsed_target_layers = None
        if target_layers:
            parsed_target_layers = [t.strip() for t in target_layers.split(",")]

        # Load layer target map if provided
        per_layer_multipliers = None
        exclude_layers = None
        layer_targeting_mode = "none"

        if layer_target_map:
            from src.abliterate import load_layer_target_map
            try:
                target_map_data = load_layer_target_map(layer_target_map)
                per_layer_multipliers = target_map_data["per_layer_multipliers"]
                exclude_layers = target_map_data.get("exclude_layers", [])
                layer_targeting_mode = "target_map"

                # Layer target map overrides adaptive weighting
                if adaptive_weighting:
                    console.print(f"[{THEME['warning']}]Warning: --layer-target-map overrides --adaptive-weighting[/{THEME['warning']}]")

            except Exception as e:
                console.print(f"[red]Error loading layer target map: {e}[/red]")
                sys.exit(1)
        elif adaptive_weighting:
            layer_targeting_mode = "adaptive"

        config = {
            "model_path": model_path,
            "output_path": effective_output_path,
            "num_prompts": effective_num_prompts,
            "direction_multiplier": effective_multiplier,
            "norm_preservation": not no_norm_preservation,
            "filter_prompts": not no_filter_prompts,
            "device": device,
            "dtype": effective_dtype,
            # Advanced options
            "use_winsorization": winsorize,
            "winsorize_percentile": winsorize_percentile,
            "use_magnitude_clipping": magnitude_clip,
            "magnitude_clip_percentile": magnitude_clip_percentile,
            # Numerical stability
            "use_welford_mean": welford_mean,
            "use_float64_subtraction": float64_subtraction,
            "use_null_space": null_space,
            "preservation_prompts_path": preservation_prompts,
            "null_space_rank_ratio": null_space_rank_ratio,
            "use_adaptive_weighting": adaptive_weighting and not layer_target_map,  # Disabled if target map provided
            # Projected abliteration (orthogonalize against harmless)
            "use_projected_refusal": projected,
            # Biprojection options
            "use_biprojection": biprojection,
            "use_per_neuron_norm": per_neuron_norm,
            "target_layer_types": parsed_target_layers,
            "use_harmless_boundary": harmless_boundary,
            "harmless_clamp_ratio": harmless_clamp_ratio,
            "num_measurement_layers": num_measurement_layers,
            "intervention_range": (intervention_start, intervention_end),
            # Layer target map options
            "layer_targeting_mode": layer_targeting_mode,
            "layer_target_map_path": layer_target_map,
            "per_layer_multipliers": per_layer_multipliers,
            "exclude_layers": exclude_layers,
            "unmapped_layer_behavior": unmapped_layer_behavior,
            # Dynamic layer targeting
            "dynamic_layer_targeting": dynamic_layer_targeting,
            # Hybrid architecture
            "hybrid_strategy": hybrid_strategy,
            "hybrid_full_attn_weight": hybrid_full_attn_weight,
            "hybrid_linear_attn_weight": hybrid_linear_attn_weight,
            "hybrid_skip_recurrent_proj": hybrid_skip_recurrent,
            "hybrid_skip_state_proj": hybrid_skip_state,
            # KL divergence monitoring
            "use_kl_monitoring": kl_monitor,
            "kl_reference_prompts_path": kl_reference_prompts,
            "kl_num_reference_prompts": kl_num_prompts,
            "kl_top_k": kl_top_k,
            "use_kl_auto_tune": kl_auto_tune,
            "kl_threshold": kl_threshold,
            "kl_search_min": kl_search_min,
            "kl_search_max": kl_search_max,
            # Gabliteration
            "use_gabliteration": gabliteration,
            "gab_num_directions": gab_num_directions,
            "gab_ridge_lambda": gab_ridge_lambda,
            "gab_layer_scaling_beta": gab_layer_beta,
            "gab_skip_first_layers": gab_skip_first,
            "gab_skip_last_layers": gab_skip_last,
            # J-lens (J-space) restriction (Phase 2)
            "use_jlens_restriction": jlens_restrict,
            "jlens_vectors_path": jlens_vectors,
            "jlens_basis_rank": jlens_basis_rank,
            "jlens_min_projection_ratio": jlens_min_projection_ratio,
            "jlens_num_prompts": jlens_num_prompts,
            "jlens_batch_size": jlens_batch_size,
            "jlens_max_seq_len": jlens_max_seq_len,
            "jlens_grad_checkpoint": jlens_grad_checkpoint,
            "jlens_concepts": (
                [c.strip() for c in jlens_concepts.split(",") if c.strip()]
                if jlens_concepts else None
            ),
            # Contrastive concept mining
            "mine_jlens_concepts": jlens_mine_concepts,
            "mine_top_k": jlens_mine_top_k,
            "mine_min_specificity": jlens_mine_min_specificity,
            "mining_mode": jlens_mining_mode,
        }

        display_banner()
        display_config_panel(config, "Batch Configuration")
        success = run_abliteration(config)
        sys.exit(0 if success else 1)
    else:
        main_menu()


if __name__ == "__main__":
    cli()

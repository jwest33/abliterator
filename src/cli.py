#!/usr/bin/env python3
"""
Interactive CLI for Abliteration Toolkit

A modern terminal interface for removing refusal behavior from language models.
Supports interactive mode (default) and batch mode for automation.
"""

import json
import sys
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

    use_advanced = questionary.confirm(
        "Configure advanced options? (Winsorization, null-space, biprojection)",
        default=False,
        style=custom_style,
    ).ask()

    # Set defaults
    config["use_projected_refusal"] = True  # Orthogonalize refusal against harmless (recommended, on by default)
    config["use_winsorization"] = False
    config["use_null_space"] = False
    config["use_adaptive_weighting"] = False
    config["use_biprojection"] = False
    config["use_per_neuron_norm"] = False
    config["target_layer_types"] = None
    config["use_harmless_boundary"] = False
    config["harmless_clamp_ratio"] = 0.1
    config["num_measurement_layers"] = 2
    config["intervention_range"] = (0.25, 0.95)

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

        # Adaptive layer weighting
        config["use_adaptive_weighting"] = questionary.confirm(
            "Enable adaptive layer weighting?",
            default=False,
            style=custom_style,
        ).ask()

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

    return config


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
                use_null_space=config.get("use_null_space", False),
                preservation_prompts_path=config.get("preservation_prompts_path"),
                null_space_rank_ratio=config.get("null_space_rank_ratio", 0.95),
                use_adaptive_weighting=config.get("use_adaptive_weighting", False),
                use_projected_refusal=config.get("use_projected_refusal", True),
                use_biprojection=config.get("use_biprojection", False),
                use_per_neuron_norm=config.get("use_per_neuron_norm", False),
                target_layer_types=config.get("target_layer_types"),
                use_harmless_boundary=config.get("use_harmless_boundary", False),
                harmless_clamp_ratio=config.get("harmless_clamp_ratio", 0.1),
                num_measurement_layers=config.get("num_measurement_layers", 2),
                intervention_range=config.get("intervention_range", (0.25, 0.95)),
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

            # Apply abliteration
            progress.update(task, description="Applying abliteration to model weights...")
            model = abliterate_model(model, directions, abl_config, null_space_projector)
            progress.advance(task, 15)

            # Save model (with version suffix if path already exists)
            progress.update(task, description="Saving abliterated model...")
            output_path = get_versioned_path(config["output_path"])
            if output_path != Path(config["output_path"]):
                console.print(f"[{THEME['warning']}]Output path exists, using: {output_path.name}[/{THEME['warning']}]")
            output_path.mkdir(parents=True, exist_ok=True)

            model.save_pretrained(output_path, safe_serialization=True)
            tokenizer.save_pretrained(output_path)
            directions.save(str(output_path / "refusal_directions.pt"))

            # Copy vision files for VL models (needed for GGUF mmproj export)
            from src.abliterate import is_vision_model, copy_vision_files
            source_path = Path(config["model_path"])
            if is_vision_model(source_path):
                progress.update(task, description="Copying vision encoder files...")
                copied_files = copy_vision_files(source_path, output_path)
                if copied_files:
                    console.print(f"[{THEME['muted']}]Copied {len(copied_files)} vision files[/{THEME['muted']}]")

            # Save null-space projectors if computed
            if null_space_projector is not None:
                null_space_projector.save(str(output_path / "null_space_projectors.pt"))

            # Save config
            config_save = {
                "model_path": config["model_path"],
                "direction_multiplier": config["direction_multiplier"],
                "norm_preservation": config["norm_preservation"],
                "num_harmful_prompts": len(abl_config.harmful_prompts),
                "num_harmless_prompts": len(abl_config.harmless_prompts),
                "timestamp": datetime.now().isoformat(),
                # Advanced options
                "use_winsorization": config.get("use_winsorization", False),
                "winsorize_percentile": config.get("winsorize_percentile"),
                "use_null_space": config.get("use_null_space", False),
                "null_space_rank_ratio": config.get("null_space_rank_ratio"),
                "use_adaptive_weighting": config.get("use_adaptive_weighting", False),
            }
            with open(output_path / "abliteration_config.json", "w") as f:
                json.dump(config_save, f, indent=2)

            progress.advance(task, 10)

        display_success(f"Model abliterated successfully!\n\nOutput saved to: {output_path}")
        return True

    except Exception as e:
        display_error(f"Abliteration failed: {str(e)}")
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
            from utils.test_abliteration import load_model, generate_response, detect_refusal

            with console.status("Loading model..."):
                model, tokenizer = load_model(model_path)

            with console.status("Generating response..."):
                response = generate_response(model, tokenizer, prompt)
                refused = detect_refusal(response)

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
    """Run refusal rate evaluation."""
    console.print(f"\n[bold {THEME['primary']}]Refusal Rate Evaluation[/bold {THEME['primary']}]\n")

    if not model_path:
        model_path = select_model("Select model to evaluate:")
        if not model_path:
            return

    # Configuration
    limit = int(questionary.text(
        "Number of prompts per category:",
        default="50",
        style=custom_style,
    ).ask())

    # Get configured eval results directory
    eval_output_dir = get_eval_results_dir()
    console.print(f"[{THEME['muted']}]Results will be saved to: {eval_output_dir}[/{THEME['muted']}]\n")

    from utils.test_abliteration import eval_refusal_rates

    results = eval_refusal_rates(
        model_path=model_path,
        harmful_prompts_path=get_default_prompts_path("harmful.txt"),
        harmless_prompts_path=get_default_prompts_path("harmless.txt"),
        limit=limit,
        output_dir=eval_output_dir,
    )

    # Display summary
    console.print()
    harmful_rate = results["results"]["harmful_prompts"]["refusal_rate,none"]
    harmless_rate = results["results"]["harmless_prompts"]["refusal_rate,none"]

    from rich.table import Table
    table = Table(title="Evaluation Results", show_header=True)
    table.add_column("Category", style=THEME["primary"])
    table.add_column("Refusal Rate", justify="right")
    table.add_column("Compliance Rate", justify="right")

    table.add_row(
        "Harmful Prompts",
        f"{harmful_rate:.1%}",
        f"{1-harmful_rate:.1%}",
    )
    table.add_row(
        "Harmless Prompts",
        f"{harmless_rate:.1%}",
        f"{1-harmless_rate:.1%}",
    )

    console.print(table)

    # Show where results were saved
    console.print(f"\n[{THEME['muted']}]Full results saved to: {eval_output_dir}[/{THEME['muted']}]")


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

    from utils.test_abliteration import load_model, generate_response, detect_refusal

    with console.status("Loading original model..."):
        orig_model, orig_tokenizer = load_model(original_path)

    with console.status("Generating original response..."):
        orig_response = generate_response(orig_model, orig_tokenizer, prompt)
        orig_refused = detect_refusal(orig_response)

    # Clear original model to free memory
    del orig_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with console.status("Loading abliterated model..."):
        abl_model, abl_tokenizer = load_model(abliterated_path)

    with console.status("Generating abliterated response..."):
        abl_response = generate_response(abl_model, abl_tokenizer, prompt)
        abl_refused = detect_refusal(abl_response)

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
        questionary.Choice("Q5_K_M - 5-bit k-quant medium (higher quality)", value="Q5_K_M"),
        questionary.Choice("Q8_0 - 8-bit (near-lossless)", value="Q8_0"),
        questionary.Choice("Q6_K - 6-bit k-quant (good for larger models)", value="Q6_K"),
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
    """Prompt user to save abliteration settings as a training config."""
    save_config_prompt = questionary.confirm(
        "\nWould you like to save these settings as a training config?",
        default=False,
        style=custom_style,
    ).ask()

    if not save_config_prompt:
        return

    # Get config name
    name = questionary.text(
        "Config name:",
        style=custom_style,
    ).ask()

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
                ("2", "Test Model", "Run test prompts on a model"),
                ("3", "Compare Models", "Side-by-side comparison"),
                ("4", "Evaluate Refusal", "Full refusal rate evaluation"),
                ("5", "Export to GGUF", "Quantize for llama.cpp"),
                ("6", "Manage Configs", "Create, edit, delete training configs"),
                ("7", "Settings", "Configure options"),
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
            run_test_model()
            questionary.press_any_key_to_continue(style=custom_style).ask()

        elif choice == "3":
            run_compare_models()
            questionary.press_any_key_to_continue(style=custom_style).ask()

        elif choice == "4":
            run_evaluation()
            questionary.press_any_key_to_continue(style=custom_style).ask()

        elif choice == "5":
            run_gguf_export()
            questionary.press_any_key_to_continue(style=custom_style).ask()

        elif choice == "6":
            run_config_management()
            questionary.press_any_key_to_continue(style=custom_style).ask()

        elif choice == "7":
            run_settings()
            questionary.press_any_key_to_continue(style=custom_style).ask()


# Click CLI for batch mode
@click.command()
@click.option("--batch", is_flag=True, help="Run in batch mode (non-interactive)")
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
def cli(batch, model_path, output_path, num_prompts, direction_multiplier,
        no_norm_preservation, no_filter_prompts, device, dtype,
        winsorize, winsorize_percentile, null_space, preservation_prompts,
        null_space_rank_ratio, adaptive_weighting,
        projected, biprojection, per_neuron_norm, target_layers, harmless_boundary,
        harmless_clamp_ratio, num_measurement_layers, intervention_start, intervention_end):
    """
    Abliteration Toolkit - Remove refusal behavior from language models.

    Run without arguments for interactive mode, or use --batch for automation.

    Advanced options enable research-backed enhancements:
    \b
      --projected              Orthogonalize refusal against harmless direction (default: on)
      --winsorize              Clip outlier activations (recommended for Gemma models)
      --null-space             Preserve model capabilities using null-space constraints
      --preservation-prompts   Custom prompts for null-space computation
      --adaptive-weighting     Per-layer adaptive ablation strength

    Biprojection options (improved NatInt preservation):
    \b
      --biprojection           Enable biprojection mode (measure at high-quality layers)
      --per-neuron-norm        Per-neuron norm preservation instead of Frobenius
      --target-layers          Only ablate specific layer types (e.g., 'o_proj,down_proj')
      --harmless-boundary      Clamp ablation to preserve harmless direction
    """
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
            "use_null_space": null_space,
            "preservation_prompts_path": preservation_prompts,
            "null_space_rank_ratio": null_space_rank_ratio,
            "use_adaptive_weighting": adaptive_weighting,
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
        }

        display_banner()
        display_config_panel(config, "Batch Configuration")
        success = run_abliteration(config)
        sys.exit(0 if success else 1)
    else:
        main_menu()


if __name__ == "__main__":
    cli()

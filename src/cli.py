#!/usr/bin/env python3
"""
Interactive CLI for Abliteration Toolkit

A modern terminal interface for removing refusal behavior from language models.
Supports interactive mode (default) and batch mode for automation.

Usage:
    python -m src.cli              # Interactive mode
    python -m src.cli --batch ...  # Batch mode (legacy CLI)
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
from src.cli_components import (
    THEME,
    add_model_path,
    clear_screen,
    console,
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
    display_warning,
    find_models,
    get_config_path,
    get_model_paths,
    load_config,
    print_divider,
    remove_model_path,
    save_config,
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
        value=None,
    ))

    selected = questionary.select(
        title,
        choices=choices,
        style=custom_style,
    ).ask()

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

    # Model selection
    console.print(f"\n[bold {THEME['primary']}]Step 1: Select Base Model[/bold {THEME['primary']}]\n")
    model_path = select_model("Select the model to abliterate:")
    if not model_path:
        return None
    config["model_path"] = model_path

    # Output path
    console.print(f"\n[bold {THEME['primary']}]Step 2: Output Path[/bold {THEME['primary']}]\n")
    default_output = f"./abliterated_models/{Path(model_path).name}-abliterated"

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

    # Advanced options
    console.print(f"\n[bold {THEME['primary']}]Step 3: Configuration[/bold {THEME['primary']}]\n")

    config["num_prompts"] = int(questionary.text(
        "Number of prompts to use:",
        default="30",
        style=custom_style,
    ).ask())

    config["direction_multiplier"] = float(questionary.text(
        "Direction multiplier (ablation strength):",
        default="1.0",
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
        default="float16",
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
            progress.advance(task, 25)

            # Apply abliteration
            progress.update(task, description="Applying abliteration to model weights...")
            model = abliterate_model(model, directions, abl_config)
            progress.advance(task, 20)

            # Save model
            progress.update(task, description="Saving abliterated model...")
            output_path = Path(config["output_path"])
            output_path.mkdir(parents=True, exist_ok=True)

            model.save_pretrained(output_path, safe_serialization=True)
            tokenizer.save_pretrained(output_path)
            directions.save(str(output_path / "refusal_directions.pt"))

            # Save config
            config_save = {
                "model_path": config["model_path"],
                "direction_multiplier": config["direction_multiplier"],
                "norm_preservation": config["norm_preservation"],
                "num_harmful_prompts": len(abl_config.harmful_prompts),
                "num_harmless_prompts": len(abl_config.harmless_prompts),
                "timestamp": datetime.now().isoformat(),
            }
            with open(output_path / "abliteration_config.json", "w") as f:
                json.dump(config_save, f, indent=2)

            progress.advance(task, 10)

        display_success(f"Model abliterated successfully!\n\nOutput saved to: {config['output_path']}")
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

    from utils.test_abliteration import eval_refusal_rates

    results = eval_refusal_rates(
        model_path=model_path,
        harmful_prompts_path=get_default_prompts_path("harmful.txt"),
        harmless_prompts_path=get_default_prompts_path("harmless.txt"),
        limit=limit,
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

    model_path = select_model("Select model to export:")
    if not model_path:
        return

    quant_type = questionary.select(
        "Select quantization type:",
        choices=[
            questionary.Choice("Q4_K_M (recommended, good balance)", value="Q4_K_M"),
            questionary.Choice("Q5_K_M (higher quality)", value="Q5_K_M"),
            questionary.Choice("Q8_0 (near-lossless)", value="Q8_0"),
            questionary.Choice("F16 (no quantization)", value="F16"),
        ],
        style=custom_style,
    ).ask()

    display_warning(
        "GGUF export requires llama.cpp tools (convert_hf_to_gguf.py and llama-quantize).\n"
        "Make sure these are installed and available in your PATH."
    )

    proceed = questionary.confirm(
        "Proceed with export?",
        default=True,
        style=custom_style,
    ).ask()

    if not proceed:
        return

    # This would call the actual GGUF export logic from app.py
    console.print(f"\n[{THEME['muted']}]GGUF export would run here with {quant_type} quantization...[/{THEME['muted']}]")
    display_warning("GGUF export is currently only available through the web interface.")


def run_settings():
    """Settings management."""
    while True:
        console.print(f"\n[bold {THEME['primary']}]Settings[/bold {THEME['primary']}]\n")

        action = questionary.select(
            "What would you like to do?",
            choices=[
                questionary.Choice("Manage model directories", value="model_paths"),
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

    # Default precision
    console.print(f"\n[bold {THEME['primary']}]Step 3: Default Precision[/bold {THEME['primary']}]\n")

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
                ("6", "Settings", "Configure options"),
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
            run_settings()
            questionary.press_any_key_to_continue(style=custom_style).ask()


# Click CLI for batch mode
@click.command()
@click.option("--batch", is_flag=True, help="Run in batch mode (non-interactive)")
@click.option("--model_path", "-m", type=str, help="Path to input model")
@click.option("--output_path", "-o", type=str, help="Path for output model")
@click.option("--num_prompts", type=int, default=30, help="Number of prompts to use")
@click.option("--direction_multiplier", type=float, default=1.0, help="Ablation strength")
@click.option("--no_norm_preservation", is_flag=True, help="Disable norm preservation")
@click.option("--no_filter_prompts", is_flag=True, help="Don't filter prompts by refusal")
@click.option("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
@click.option("--dtype", type=click.Choice(["float16", "bfloat16", "float32"]), default="float16")
def cli(batch, model_path, output_path, num_prompts, direction_multiplier,
        no_norm_preservation, no_filter_prompts, device, dtype):
    """
    Abliteration Toolkit - Remove refusal behavior from language models.

    Run without arguments for interactive mode, or use --batch for automation.
    """
    if batch:
        if not model_path or not output_path:
            console.print("[red]Error: --model_path and --output_path are required in batch mode[/red]")
            sys.exit(1)

        config = {
            "model_path": model_path,
            "output_path": output_path,
            "num_prompts": num_prompts,
            "direction_multiplier": direction_multiplier,
            "norm_preservation": not no_norm_preservation,
            "filter_prompts": not no_filter_prompts,
            "device": device,
            "dtype": dtype,
        }

        display_banner()
        display_config_panel(config, "Batch Configuration")
        success = run_abliteration(config)
        sys.exit(0 if success else 1)
    else:
        main_menu()


if __name__ == "__main__":
    cli()

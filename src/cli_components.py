#!/usr/bin/env python3
"""
Rich CLI Components for Abliteration Toolkit

Reusable UI components for the interactive CLI including:
- ASCII art banner with gradient colors
- System info display
- Model selector
- Progress callbacks
- Results tables
- Configuration management
"""

import json
import re
from pathlib import Path
from typing import Callable, Optional

import psutil
import torch


# Configuration Management

def get_config_dir() -> Path:
    """Get the configuration directory path."""
    config_dir = Path.home() / "abliterate"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_config_path() -> Path:
    """Get the config file path."""
    return get_config_dir() / "config.json"


def get_user_prompts_dir() -> Path:
    """Get the user prompts directory path (~/.abliterate/prompts/)."""
    prompts_dir = get_config_dir() / "prompts"
    return prompts_dir


def get_package_prompts_dir() -> Path:
    """Get the package prompts directory path (bundled with the app)."""
    # Navigate from this file to the prompts directory
    return Path(__file__).parent.parent / "prompts"


def copy_prompts_to_user_dir(force: bool = False) -> bool:
    """
    Copy the default prompts from the package to the user's config directory.

    Args:
        force: If True, overwrite existing files. If False, skip existing files.

    Returns:
        True if any files were copied, False otherwise.
    """
    import shutil

    package_prompts = get_package_prompts_dir()
    user_prompts = get_user_prompts_dir()

    if not package_prompts.exists():
        return False

    # Create user prompts directory
    user_prompts.mkdir(parents=True, exist_ok=True)

    copied = False
    for prompt_file in package_prompts.glob("*.txt"):
        dest_file = user_prompts / prompt_file.name
        if force or not dest_file.exists():
            shutil.copy2(prompt_file, dest_file)
            copied = True

    return copied


def user_prompts_exist() -> bool:
    """Check if user prompts directory exists and has prompt files."""
    user_prompts = get_user_prompts_dir()
    if not user_prompts.exists():
        return False
    return any(user_prompts.glob("*.txt"))


def get_prompts_path(filename: str) -> Path:
    """
    Get the path to a prompts file, preferring user prompts over package prompts.

    Args:
        filename: Name of the prompts file (e.g., 'harmful.txt')

    Returns:
        Path to the prompts file (user's copy if exists, otherwise package default)
    """
    # Check user prompts directory first
    user_path = get_user_prompts_dir() / filename
    if user_path.exists():
        return user_path

    # Fall back to package prompts
    package_path = get_package_prompts_dir() / filename
    return package_path


def get_default_config() -> dict:
    """Return default configuration settings."""
    config_dir = get_config_dir()
    models_dir = str(config_dir / "models")
    eval_dir = str(config_dir / "eval_results")

    return {
        "model_paths": [
            "D:/models",
            "C:/models",
            str(Path.home() / ".cache" / "huggingface" / "hub"),
            "./",
            models_dir,  # Include the default output dir in search paths
        ],
        "default_output_dir": models_dir,
        "eval_results_dir": eval_dir,
        "default_num_prompts": 30,
        "default_direction_multiplier": 1.0,
        "default_dtype": "float16",
        # Biprojection defaults
        "default_use_biprojection": False,
        "default_use_per_neuron_norm": False,
        "default_target_layer_types": ["o_proj", "down_proj"],
        "default_harmless_boundary": False,
        "default_harmless_clamp_ratio": 0.1,
        "default_num_measurement_layers": 2,
        "default_intervention_range": [0.25, 0.95],
    }


def load_config() -> dict:
    """Load configuration from file, creating defaults if needed."""
    config_path = get_config_path()

    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_config = json.load(f)
            # Merge with defaults to ensure all keys exist
            config = get_default_config()
            config.update(user_config)
            return config
        except (json.JSONDecodeError, IOError):
            # Return defaults if file is corrupted
            return get_default_config()
    else:
        # Create config file with defaults
        config = get_default_config()
        save_config(config)
        return config


def save_config(config: dict) -> None:
    """Save configuration to file."""
    config_path = get_config_path()
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def add_model_path(path: str) -> bool:
    """Add a model search path to the config."""
    config = load_config()
    path = str(Path(path).resolve())

    if path not in config["model_paths"]:
        config["model_paths"].insert(0, path)  # Add to front for priority
        save_config(config)
        return True
    return False


def remove_model_path(path: str) -> bool:
    """Remove a model search path from the config."""
    config = load_config()

    if path in config["model_paths"]:
        config["model_paths"].remove(path)
        save_config(config)
        return True
    return False


def get_model_paths() -> list[str]:
    """Get the list of configured model search paths."""
    config = load_config()
    return config.get("model_paths", get_default_config()["model_paths"])


def get_eval_results_dir() -> str:
    """Get the configured evaluation results directory."""
    config = load_config()
    return config.get("eval_results_dir", get_default_config()["eval_results_dir"])


def set_eval_results_dir(path: str) -> None:
    """Set the evaluation results directory."""
    config = load_config()
    config["eval_results_dir"] = str(Path(path).resolve())
    save_config(config)


def get_default_output_dir() -> str:
    """Get the configured default output directory for abliterated models."""
    config = load_config()
    return config.get("default_output_dir", get_default_config()["default_output_dir"])


def set_default_output_dir(path: str) -> None:
    """Set the default output directory for abliterated models."""
    config = load_config()
    config["default_output_dir"] = str(Path(path).resolve())
    save_config(config)


def get_default_num_prompts() -> int:
    """Get the configured default number of prompts for abliteration."""
    config = load_config()
    return config.get("default_num_prompts", get_default_config()["default_num_prompts"])


def get_default_direction_multiplier() -> float:
    """Get the configured default direction multiplier (ablation strength)."""
    config = load_config()
    return config.get("default_direction_multiplier", get_default_config()["default_direction_multiplier"])


def get_default_dtype() -> str:
    """Get the configured default dtype for abliteration."""
    config = load_config()
    return config.get("default_dtype", get_default_config()["default_dtype"])


def get_versioned_path(path: Path | str) -> Path:
    """
    Get a versioned path if the original path already exists.

    If the path doesn't exist, returns it unchanged.
    If the path exists, appends _v2, _v3, etc. until finding an available name.

    Examples:
        ./output -> ./output (if doesn't exist)
        ./output -> ./output_v2 (if ./output exists)
        ./output -> ./output_v3 (if ./output and ./output_v2 exist)

    Args:
        path: The desired output path

    Returns:
        Path: The original path or a versioned variant that doesn't exist
    """
    path = Path(path)

    if not path.exists():
        return path

    # Path exists, find next available version
    base_name = path.name
    parent = path.parent

    # Check if the name already has a version suffix
    version_match = re.match(r'^(.+)_v(\d+)$', base_name)
    if version_match:
        # Already versioned, increment from that version
        base_name = version_match.group(1)
        start_version = int(version_match.group(2)) + 1
    else:
        start_version = 2

    # Find next available version
    version = start_version
    while True:
        versioned_name = f"{base_name}_v{version}"
        versioned_path = parent / versioned_name
        if not versioned_path.exists():
            return versioned_path
        version += 1

        # Safety limit to prevent infinite loop
        if version > 1000:
            raise ValueError(f"Too many versions of {base_name} exist (>1000)")


# Rich UI Components

from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.style import Style
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# Theme colors
THEME = {
    "primary": "cyan",
    "secondary": "bright_yellow",
    "accent": "orange1",
    "success": "green",
    "warning": "yellow",
    "error": "red",
    "muted": "dim white",
}

# Custom theme for consistent black background
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red",
})

# ASCII Art Banner
BANNER = r"""
 _______  ______   _       __________________ _______  _______  _______ _________ _______ 
(  ___  )(  ___ \ ( \      \__   __/\__   __/(  ____ \(  ____ )(  ___  )\__   __/(  ____ \
| (   ) || (   ) )| (         ) (      ) (   | (    \/| (    )|| (   ) |   ) (   | (    \/
| (___) || (__/ / | |         | |      | |   | (__    | (____)|| (___) |   | |   | (__ 
|  ___  ||  __ (  | |         | |      | |   |  __)   |     __)|  ___  |   | |   |  __)
| (   ) || (  \ \ | |         | |      | |   | (      | (\ (   | (   ) |   | |   | (
| )   ( || )___) )| (____/\___) (___   | |   | (____/\| ) \ \__| )   ( |   | |   | (____/\
|/     \||/ \___/ (_______/\_______/   )_(   (_______/|/   \__/|/     \|   )_(   (_______/
"""

# Console with black background styling
console = Console(theme=custom_theme, force_terminal=True, style="on black")

# Yellow/orange gradient for banner
BANNER_COLORS = ["#ff8c00", "#ffa500", "#ffb732", "#ffc966", "#ffd700", "#ffdf00", "#ffe135", "#ffd700"]


def get_gradient_text(text: str, colors: list[str] = None) -> Text:
    """Create gradient-colored text."""
    if colors is None:
        colors = BANNER_COLORS

    rich_text = Text()
    lines = text.split("\n")

    for line_idx, line in enumerate(lines):
        # Cycle through colors for each line
        color = colors[line_idx % len(colors)]
        rich_text.append(line + "\n", style=Style(color=color, bold=True))

    return rich_text


def display_banner():
    """Display the ASCII art banner with gradient colors."""
    banner_text = get_gradient_text(BANNER)
    console.print(Align.center(banner_text))

    # Subtitle
    subtitle = Text()
    subtitle.append("Orthogonal Projection Abliteration with Norm-Preservation, Null-Space Constaints, Winsorization, and Adaptive Layer Weighting", style=f"bold {THEME['muted']}")
    console.print(Align.center(subtitle))
    console.print()


def get_system_info() -> dict:
    """Gather system information."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": 0,
        "cuda_device_name": None,
        "cuda_memory_total": 0,
        "cuda_memory_used": 0,
        "ram_total": psutil.virtual_memory().total / (1024**3),
        "ram_used": psutil.virtual_memory().used / (1024**3),
    }

    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_total"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        info["cuda_memory_used"] = torch.cuda.memory_allocated() / (1024**3)

    return info


def display_system_info():
    """Display system information panel."""
    info = get_system_info()

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style=THEME["muted"])
    table.add_column("Value", style=THEME["primary"])

    if info["cuda_available"]:
        table.add_row("GPU", info["cuda_device_name"])
        table.add_row(
            "VRAM",
            f"{info['cuda_memory_used']:.1f} / {info['cuda_memory_total']:.1f} GB"
        )
    else:
        table.add_row("GPU", "Not available (CPU mode)")

    table.add_row("RAM", f"{info['ram_used']:.1f} / {info['ram_total']:.1f} GB")

    panel = Panel(
        table,
        title="[bold]System Info[/bold]",
        border_style=THEME["muted"],
        padding=(0, 1),
        expand=False,
    )
    console.print(Align.center(panel))
    console.print()


def display_menu(title: str, options: list[tuple[str, str]], show_quit: bool = True) -> None:
    """Display a menu panel."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style=f"bold {THEME['accent']}", width=5)
    table.add_column("Option", style=f"bold {THEME['primary']}")
    table.add_column("Description", style=THEME["muted"])

    for key, label, desc in options:
        table.add_row(f"[{key}]", label, desc)

    if show_quit:
        table.add_row("", "", "")  # Spacer
        table.add_row("[Q]", "Quit", "Exit the application")

    panel = Panel(
        table,
        title=f"[bold]{title}[/bold]",
        border_style=THEME["secondary"],
        padding=(1, 2),
        expand=False,
    )
    console.print(Align.center(panel))


def find_models(search_paths: list[Path] = None) -> list[dict]:
    """Find available models in configured locations."""
    if search_paths is None:
        # Load paths from user config
        configured_paths = get_model_paths()
        search_paths = [Path(p) for p in configured_paths]

    models = []
    seen_paths = set()

    for base_path in search_paths:
        if not base_path.exists():
            continue

        for item in base_path.iterdir():
            if item.is_dir() and item.resolve() not in seen_paths:
                config_path = item / "config.json"
                if config_path.exists():
                    model_info = {
                        "path": str(item),
                        "name": item.name,
                        "is_abliterated": (item / "abliteration_config.json").exists(),
                    }
                    models.append(model_info)
                    seen_paths.add(item.resolve())

    return sorted(models, key=lambda x: (not x["is_abliterated"], x["name"]))


def display_model_list(models: list[dict], title: str = "Available Models") -> None:
    """Display a list of models in a table."""
    table = Table(title=title, show_header=True, header_style=f"bold {THEME['primary']}")
    table.add_column("#", style="dim", width=4)
    table.add_column("Name", style=THEME["primary"])
    table.add_column("Type", style=THEME["muted"])
    table.add_column("Path", style="dim")

    for idx, model in enumerate(models, 1):
        model_type = "[green]Abliterated[/green]" if model["is_abliterated"] else "Base"
        table.add_row(
            str(idx),
            model["name"],
            model_type,
            model["path"][:50] + "..." if len(model["path"]) > 50 else model["path"],
        )

    console.print(table)


def display_config_panel(config: dict, title: str = "Configuration") -> None:
    """Display a configuration summary panel."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Setting", style=THEME["muted"])
    table.add_column("Value", style=THEME["primary"])

    for key, value in config.items():
        # Format the key nicely
        display_key = key.replace("_", " ").title()

        # Format the value
        if isinstance(value, bool):
            display_value = f"[green]Yes[/green]" if value else f"[red]No[/red]"
        elif isinstance(value, float):
            display_value = f"{value:.2f}"
        elif isinstance(value, Path):
            display_value = str(value)
        elif isinstance(value, tuple):
            # Format tuples nicely (e.g., intervention_range)
            display_value = f"({', '.join(str(v) for v in value)})"
        elif isinstance(value, list):
            # Format lists nicely (e.g., target_layer_types)
            if value:
                display_value = ", ".join(str(v) for v in value)
            else:
                display_value = "[dim]None[/dim]"
        elif value is None:
            display_value = "[dim]None[/dim]"
        else:
            display_value = str(value)

        table.add_row(display_key, display_value)

    panel = Panel(
        table,
        title=f"[bold]{title}[/bold]",
        border_style=THEME["secondary"],
        padding=(0, 1),
        expand=False,
    )
    console.print(panel)


def create_progress_bar(description: str = "Processing") -> Progress:
    """Create a Rich progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        expand=False,
    )


def display_results_table(
    results: list[dict],
    columns: list[tuple[str, str]],
    title: str = "Results"
) -> None:
    """Display results in a formatted table."""
    table = Table(title=title, show_header=True, header_style=f"bold {THEME['primary']}")

    for col_name, col_style in columns:
        table.add_column(col_name, style=col_style)

    for row in results:
        table.add_row(*[str(row.get(col[0].lower().replace(" ", "_"), "")) for col in columns])

    console.print(table)


def display_comparison_panel(
    prompt: str,
    response1: str,
    response2: str,
    label1: str = "Original",
    label2: str = "Abliterated",
    refused1: bool = False,
    refused2: bool = False,
) -> None:
    """Display side-by-side model comparison."""
    # Prompt panel
    prompt_panel = Panel(
        prompt,
        title="[bold]Prompt[/bold]",
        border_style=THEME["primary"],
        padding=(0, 1),
        expand=False,
    )
    console.print(prompt_panel)
    console.print()

    # Create response panels
    status1 = f"[red]REFUSED[/red]" if refused1 else f"[green]OK[/green]"
    status2 = f"[red]REFUSED[/red]" if refused2 else f"[green]OK[/green]"

    panel1 = Panel(
        response1[:500] + "..." if len(response1) > 500 else response1,
        title=f"[bold]{label1}[/bold] {status1}",
        border_style="red" if refused1 else "green",
        padding=(0, 1),
        expand=False,
    )

    panel2 = Panel(
        response2[:500] + "..." if len(response2) > 500 else response2,
        title=f"[bold]{label2}[/bold] {status2}",
        border_style="red" if refused2 else "green",
        padding=(0, 1),
        expand=False,
    )

    console.print(panel1)
    console.print()
    console.print(panel2)


def display_success(message: str) -> None:
    """Display a success message."""
    panel = Panel(
        message,
        title="[bold green]Success[/bold green]",
        border_style="green",
        padding=(0, 1),
        expand=False,
    )
    console.print(panel)


def display_error(message: str) -> None:
    """Display an error message."""
    panel = Panel(
        message,
        title="[bold red]Error[/bold red]",
        border_style="red",
        padding=(0, 1),
        expand=False,
    )
    console.print(panel)


def display_warning(message: str) -> None:
    """Display a warning message."""
    panel = Panel(
        message,
        title="[bold yellow]Warning[/bold yellow]",
        border_style="yellow",
        padding=(0, 1),
        expand=False,
    )
    console.print(panel)


def clear_screen():
    """Clear the console screen."""
    console.clear()


def print_divider():
    """Print a horizontal divider."""
    console.print()
    console.rule(style=THEME["muted"])
    console.print()


# ==============================================================================
# Training Config Display Components
# ==============================================================================


def display_training_config_list(configs: list[dict], title: str = "Saved Training Configs") -> None:
    """
    Display a list of training configs in a formatted table.

    Args:
        configs: List of config metadata dicts from list_configs()
        title: Title for the table
    """
    if not configs:
        console.print(f"[{THEME['muted']}]No saved configs found.[/{THEME['muted']}]")
        console.print()
        return

    table = Table(
        title=title,
        show_header=True,
        header_style=f"bold {THEME['primary']}",
        border_style=THEME["muted"],
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Name", style=f"bold {THEME['primary']}")
    table.add_column("Description", style=THEME["muted"], max_width=40)
    table.add_column("Last Updated", style="dim")

    for idx, config in enumerate(configs, 1):
        # Format the date
        updated = config.get("updated_at", "")
        if updated:
            try:
                # Parse ISO format and display nicely
                from datetime import datetime
                dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                updated = dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, AttributeError):
                pass

        # Check for errors
        if "error" in config:
            name_display = f"[red]{config['name']}[/red]"
            desc = f"[red]{config.get('description', 'Error loading')}[/red]"
        else:
            name_display = config.get("name", config.get("filename", "Unknown"))
            desc = config.get("description", "") or "[dim]No description[/dim]"

        # Truncate description if too long
        if len(desc) > 40:
            desc = desc[:37] + "..."

        table.add_row(str(idx), name_display, desc, updated)

    console.print(table)
    console.print()


def display_training_config_details(config: dict, title: str = "Config Details") -> None:
    """
    Display full training config details in a formatted panel.

    Args:
        config: Full config dict with metadata and settings
        title: Title for the panel
    """
    # Create metadata section
    metadata = config.get("metadata", {})
    settings = config.get("settings", {})

    # Metadata table
    meta_table = Table(show_header=False, box=None, padding=(0, 2))
    meta_table.add_column("Key", style=THEME["muted"], width=20)
    meta_table.add_column("Value", style=THEME["primary"])

    meta_table.add_row("Name", metadata.get("name", "Unknown"))
    meta_table.add_row("Description", metadata.get("description", "") or "[dim]None[/dim]")

    created = metadata.get("created_at", "")
    if created:
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            created = dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, AttributeError):
            pass
    meta_table.add_row("Created", created or "[dim]Unknown[/dim]")

    updated = metadata.get("updated_at", "")
    if updated:
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
            updated = dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, AttributeError):
            pass
    meta_table.add_row("Last Updated", updated or "[dim]Unknown[/dim]")

    # Settings table
    settings_table = Table(show_header=False, box=None, padding=(0, 2))
    settings_table.add_column("Setting", style=THEME["muted"], width=28)
    settings_table.add_column("Value", style=THEME["primary"])

    # Group settings by category for better readability
    basic_settings = [
        ("num_prompts", "Num Prompts"),
        ("direction_multiplier", "Direction Multiplier"),
        ("norm_preservation", "Norm Preservation"),
        ("filter_prompts", "Filter Prompts"),
        ("device", "Device"),
        ("dtype", "Dtype"),
    ]

    stability_settings = [
        ("use_welford_mean", "Welford Mean"),
        ("use_float64_subtraction", "Float64 Subtraction"),
    ]

    winsorization_settings = [
        ("use_winsorization", "Winsorization"),
        ("winsorize_percentile", "Winsorize Percentile"),
        ("use_magnitude_clipping", "Magnitude Clipping"),
        ("magnitude_clip_percentile", "Magnitude Clip Percentile"),
    ]

    nullspace_settings = [
        ("use_null_space", "Null-Space Constraints"),
        ("preservation_prompts_path", "Preservation Prompts"),
        ("null_space_rank_ratio", "Null-Space Rank Ratio"),
    ]

    advanced_settings = [
        ("use_projected_refusal", "Projected Refusal"),
        ("use_adaptive_weighting", "Adaptive Weighting"),
        ("use_biprojection", "Biprojection"),
        ("use_per_neuron_norm", "Per-Neuron Norm"),
        ("target_layer_types", "Target Layer Types"),
        ("use_harmless_boundary", "Harmless Boundary"),
        ("harmless_clamp_ratio", "Harmless Clamp Ratio"),
        ("num_measurement_layers", "Measurement Layers"),
        ("intervention_range", "Intervention Range"),
    ]

    def format_value(value):
        """Format a setting value for display."""
        if isinstance(value, bool):
            return f"[green]Yes[/green]" if value else f"[red]No[/red]"
        elif isinstance(value, float):
            return f"{value:.3f}"
        elif isinstance(value, list):
            if value:
                return ", ".join(str(v) for v in value)
            return "[dim]None[/dim]"
        elif value is None:
            return "[dim]None[/dim]"
        return str(value)

    # Add basic settings
    settings_table.add_row("[bold]Basic Settings[/bold]", "")
    for key, label in basic_settings:
        if key in settings:
            settings_table.add_row(f"  {label}", format_value(settings[key]))

    # Add stability settings
    settings_table.add_row("", "")
    settings_table.add_row("[bold]Numerical Stability[/bold]", "")
    for key, label in stability_settings:
        if key in settings:
            settings_table.add_row(f"  {label}", format_value(settings[key]))

    # Add winsorization settings (only if enabled)
    if settings.get("use_winsorization") or settings.get("use_magnitude_clipping"):
        settings_table.add_row("", "")
        settings_table.add_row("[bold]Outlier Clipping[/bold]", "")
        for key, label in winsorization_settings:
            if key in settings:
                settings_table.add_row(f"  {label}", format_value(settings[key]))

    # Add null-space settings (only if enabled)
    if settings.get("use_null_space"):
        settings_table.add_row("", "")
        settings_table.add_row("[bold]Null-Space Constraints[/bold]", "")
        for key, label in nullspace_settings:
            if key in settings:
                settings_table.add_row(f"  {label}", format_value(settings[key]))

    # Add advanced settings (only if any are enabled)
    advanced_enabled = any(
        settings.get(key) for key, _ in advanced_settings
        if key in ("use_adaptive_weighting", "use_biprojection", "use_per_neuron_norm", "use_harmless_boundary")
    )
    if advanced_enabled:
        settings_table.add_row("", "")
        settings_table.add_row("[bold]Advanced Options[/bold]", "")
        for key, label in advanced_settings:
            if key in settings:
                settings_table.add_row(f"  {label}", format_value(settings[key]))

    # Combine into panel
    from rich.console import Group
    content = Group(
        Text("Metadata", style=f"bold {THEME['secondary']}"),
        meta_table,
        Text(""),
        Text("Settings", style=f"bold {THEME['secondary']}"),
        settings_table,
    )

    panel = Panel(
        content,
        title=f"[bold]{title}[/bold]",
        border_style=THEME["secondary"],
        padding=(1, 2),
        expand=False,
    )
    console.print(panel)
    console.print()


class ProgressCallback:
    """Progress callback for long-running operations."""

    def __init__(self, progress: Progress, task_id):
        self.progress = progress
        self.task_id = task_id

    def update(self, advance: float = 1, description: str = None):
        """Update progress."""
        if description:
            self.progress.update(self.task_id, description=description)
        self.progress.advance(self.task_id, advance)

    def set_total(self, total: int):
        """Set the total for the progress bar."""
        self.progress.update(self.task_id, total=total)

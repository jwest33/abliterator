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


# =============================================================================
# Configuration Management
# =============================================================================

def get_config_dir() -> Path:
    """Get the configuration directory path."""
    config_dir = Path.home() / ".abliterate"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_config_path() -> Path:
    """Get the config file path."""
    return get_config_dir() / "config.json"


def get_default_config() -> dict:
    """Return default configuration settings."""
    return {
        "model_paths": [
            "D:/models",
            "C:/models",
            str(Path.home() / ".cache" / "huggingface" / "hub"),
            "./",
            "./abliterate/abliterated_models",
        ],
        "default_output_dir": "./abliterate/abliterated_models",
        "eval_results_dir": "./abliterate/eval_results",
        "default_num_prompts": 30,
        "default_direction_multiplier": 1.0,
        "default_dtype": "float16",
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


# =============================================================================
# Rich UI Components
# =============================================================================

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
    console.print(banner_text)

    # Subtitle
    subtitle = Text()
    subtitle.append("Norm-Preserving Orthogonal Projection Abliteration", style=f"bold {THEME['muted']}")
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
    console.print(panel)
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
    console.print(panel)


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

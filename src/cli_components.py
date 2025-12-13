"""
Rich CLI Components for Abliteration Toolkit

Reusable UI components for the interactive CLI including:
- ASCII art banner with gradient colors
- System info display
- Model selector
- Progress callbacks
- Results tables
"""

from pathlib import Path
from typing import Callable, Optional

import psutil
import torch
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
 _______  ______   _       __________________ _______  _______  _______ _________ _______  _______
(  ___  )(  ___ \ ( \      \__   __/\__   __/(  ____ \(  ____ )(  ___  )\__   __/(  ___  )(  ____ )
| (   ) || (   ) )| (         ) (      ) (   | (    \/| (    )|| (   ) |   ) (   | (   ) || (    )|
| (___) || (__/ / | |         | |      | |   | (__    | (____)|| (___) |   | |   | |   | || (____)|
|  ___  ||  __ (  | |         | |      | |   |  __)   |     __)|  ___  |   | |   | |   | ||     __)
| (   ) || (  \ \ | |         | |      | |   | (      | (\ (   | (   ) |   | |   | |   | || (\ (
| )   ( || )___) )| (____/\___) (___   | |   | (____/\| ) \ \__| )   ( |   | |   | (___) || ) \ \__
|/     \||/ \___/ (_______/\_______/   )_(   (_______/|/   \__/|/     \|   )_(   (_______)|/   \__/
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
    """Find available models in common locations."""
    if search_paths is None:
        search_paths = [
            Path("D:/models"),
            Path("C:/models"),
            Path.home() / ".cache" / "huggingface" / "hub",
            Path("./"),
            Path("./abliterated_models"),
        ]

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

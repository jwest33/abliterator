#!/usr/bin/env python3
"""
Batch GGUF Quantization Script

Quantizes a base GGUF file (typically F16/F32) to multiple common quantization levels
using llama.cpp's llama-quantize tool.
"""

import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import argparse
import shutil


@dataclass
class QuantConfig:
    """Configuration for a quantization type."""
    name: str
    description: str
    priority: int  # Lower = higher quality, run first


# Common quantization types ordered by quality (highest to lowest)
QUANT_CONFIGS = [
    QuantConfig("Q8_0", "8-bit, minimal quality loss", 1),
    QuantConfig("Q6_K", "6-bit k-quant, good for larger models", 2),
    QuantConfig("Q5_K_M", "5-bit k-quant medium, solid middle ground", 3),
    QuantConfig("Q5_K_S", "5-bit k-quant small", 4),
    QuantConfig("Q4_K_M", "4-bit k-quant medium, popular choice", 5),
    QuantConfig("Q4_K_S", "4-bit k-quant small", 6),
    QuantConfig("Q3_K_M", "3-bit k-quant medium, aggressive compression", 7),
    QuantConfig("Q3_K_S", "3-bit k-quant small", 8),
    QuantConfig("Q2_K", "2-bit k-quant, very small, noticeable quality loss", 9),
    # Additional types you might want
    QuantConfig("IQ4_NL", "4-bit i-quant non-linear, good quality", 5),
    QuantConfig("IQ4_XS", "4-bit i-quant extra small", 6),
    QuantConfig("IQ3_M", "3-bit i-quant medium", 7),
    QuantConfig("IQ2_M", "2-bit i-quant medium, extreme compression", 9),
]

# Default subset for typical use
DEFAULT_QUANTS = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]


def find_llama_quantize() -> Optional[Path]:
    """
    Attempt to find llama-quantize executable.
    Checks common locations and PATH.
    """
    # Check if it's in PATH
    quantize_path = shutil.which("llama-quantize")
    if quantize_path:
        return Path(quantize_path)
    
    # Check common Windows locations
    common_paths = [
        Path("llama-quantize.exe"),
        Path("./llama-quantize.exe"),
        Path("./build/bin/llama-quantize.exe"),
        Path("./build/bin/Release/llama-quantize.exe"),
        Path("../llama.cpp/build/bin/llama-quantize.exe"),
        Path("../llama.cpp/build/bin/Release/llama-quantize.exe"),
        # Linux/Mac paths
        Path("llama-quantize"),
        Path("./llama-quantize"),
        Path("./build/bin/llama-quantize"),
        Path("../llama.cpp/build/bin/llama-quantize"),
    ]
    
    for path in common_paths:
        if path.exists():
            return path.resolve()
    
    return None


def quantize_model(
    llama_quantize_path: Path,
    input_gguf: Path,
    output_gguf: Path,
    quant_type: str,
    num_threads: Optional[int] = None,
) -> bool:
    """
    Run llama-quantize on a single model.
    
    Returns True if successful, False otherwise.
    """
    cmd = [str(llama_quantize_path)]
    
    if num_threads:
        cmd.extend(["--nthreads", str(num_threads)])
    
    cmd.extend([
        str(input_gguf),
        str(output_gguf),
        quant_type,
    ])
    
    print(f"\n{'='*60}")
    print(f"Quantizing to {quant_type}")
    print(f"Output: {output_gguf}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Let output stream to console
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Quantization failed for {quant_type}")
        print(f"Return code: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"ERROR: Could not find llama-quantize at {llama_quantize_path}")
        return False


def batch_quantize(
    input_gguf: Path,
    output_dir: Optional[Path] = None,
    quant_types: Optional[list[str]] = None,
    llama_quantize_path: Optional[Path] = None,
    num_threads: Optional[int] = None,
    name_prefix: Optional[str] = None,
) -> dict[str, bool]:
    """
    Quantize a GGUF file to multiple quantization levels.
    
    Args:
        input_gguf: Path to the input F16/F32 GGUF file
        output_dir: Directory for output files (default: same as input)
        quant_types: List of quantization types (default: DEFAULT_QUANTS)
        llama_quantize_path: Path to llama-quantize executable
        num_threads: Number of threads for quantization
        name_prefix: Custom prefix for output filenames
        
    Returns:
        Dictionary mapping quant type to success status
    """
    input_gguf = Path(input_gguf).resolve()
    
    if not input_gguf.exists():
        raise FileNotFoundError(f"Input GGUF not found: {input_gguf}")
    
    # Find llama-quantize if not provided
    if llama_quantize_path is None:
        llama_quantize_path = find_llama_quantize()
        if llama_quantize_path is None:
            raise FileNotFoundError(
                "Could not find llama-quantize. Please provide path via --llama-quantize "
                "or ensure it's in your PATH."
            )
    else:
        llama_quantize_path = Path(llama_quantize_path).resolve()
        if not llama_quantize_path.exists():
            raise FileNotFoundError(f"llama-quantize not found at: {llama_quantize_path}")
    
    print(f"Using llama-quantize: {llama_quantize_path}")
    
    # Set output directory
    if output_dir is None:
        output_dir = input_gguf.parent
    else:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set quant types
    if quant_types is None:
        quant_types = DEFAULT_QUANTS
    
    # Validate quant types
    valid_quant_names = {q.name for q in QUANT_CONFIGS}
    for qt in quant_types:
        if qt not in valid_quant_names:
            print(f"WARNING: '{qt}' is not a recognized quant type, will attempt anyway")
    
    # Generate name prefix from input filename if not provided
    if name_prefix is None:
        # Remove common suffixes to get base name
        stem = input_gguf.stem
        for suffix in ["-f16", "-f32", "-bf16", "_f16", "_f32", "_bf16", ".gguf"]:
            if stem.lower().endswith(suffix.lower()):
                stem = stem[:-len(suffix)]
        name_prefix = stem
    
    # Sort quant types by priority (quality)
    quant_priority = {q.name: q.priority for q in QUANT_CONFIGS}
    quant_types_sorted = sorted(
        quant_types,
        key=lambda x: quant_priority.get(x, 99)
    )
    
    results = {}
    
    print(f"\nInput GGUF: {input_gguf}")
    print(f"Output directory: {output_dir}")
    print(f"Quant types to generate: {', '.join(quant_types_sorted)}")
    print(f"Name prefix: {name_prefix}")
    
    for quant_type in quant_types_sorted:
        output_filename = f"{name_prefix}-{quant_type.lower()}.gguf"
        output_path = output_dir / output_filename
        
        # Skip if output already exists
        if output_path.exists():
            print(f"\nSkipping {quant_type}: {output_path} already exists")
            results[quant_type] = True
            continue
        
        success = quantize_model(
            llama_quantize_path=llama_quantize_path,
            input_gguf=input_gguf,
            output_gguf=output_path,
            quant_type=quant_type,
            num_threads=num_threads,
        )
        results[quant_type] = success
        
        if success:
            # Report file size
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"✓ Created {output_path.name} ({size_mb:.1f} MB)")
    
    return results


def print_summary(results: dict[str, bool], input_path: Path, output_dir: Path):
    """Print a summary of the quantization results."""
    print("\n" + "="*60)
    print("QUANTIZATION SUMMARY")
    print("="*60)
    
    successful = [k for k, v in results.items() if v]
    failed = [k for k, v in results.items() if not v]
    
    print(f"\nSuccessful: {len(successful)}/{len(results)}")
    for quant_type in successful:
        config = next((q for q in QUANT_CONFIGS if q.name == quant_type), None)
        desc = f" - {config.description}" if config else ""
        print(f"  ✓ {quant_type}{desc}")
    
    if failed:
        print(f"\nFailed: {len(failed)}")
        for quant_type in failed:
            print(f"  ✗ {quant_type}")
    
    # List output files with sizes
    print(f"\nOutput files in {output_dir}:")
    for gguf_file in sorted(output_dir.glob("*.gguf")):
        size_mb = gguf_file.stat().st_size / (1024 * 1024)
        size_gb = size_mb / 1024
        if size_gb >= 1:
            print(f"  {gguf_file.name}: {size_gb:.2f} GB")
        else:
            print(f"  {gguf_file.name}: {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Batch quantize a GGUF model to multiple quantization levels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults (Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K)
  python batch_quantize.py model-f16.gguf

  # Specify quant types
  python batch_quantize.py model-f16.gguf -q Q4_K_M Q5_K_M Q8_0

  # Custom output directory and llama-quantize path
  python batch_quantize.py model-f16.gguf -o ./quants --llama-quantize /path/to/llama-quantize

  # List available quant types
  python batch_quantize.py --list-quants
        """
    )
    
    parser.add_argument(
        "input_gguf",
        type=Path,
        nargs="?",
        help="Path to input F16/F32 GGUF file"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        help="Output directory (default: same as input file)"
    )
    
    parser.add_argument(
        "-q", "--quant-types",
        nargs="+",
        help=f"Quantization types to generate (default: {' '.join(DEFAULT_QUANTS)})"
    )
    
    parser.add_argument(
        "--llama-quantize",
        type=Path,
        help="Path to llama-quantize executable"
    )
    
    parser.add_argument(
        "-t", "--threads",
        type=int,
        help="Number of threads for quantization"
    )
    
    parser.add_argument(
        "--name-prefix",
        type=str,
        help="Custom prefix for output filenames"
    )
    
    parser.add_argument(
        "--list-quants",
        action="store_true",
        help="List available quantization types and exit"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all available quantization types"
    )
    
    args = parser.parse_args()
    
    if args.list_quants:
        print("Available quantization types:\n")
        print(f"{'Type':<12} {'Priority':<10} Description")
        print("-" * 60)
        for config in QUANT_CONFIGS:
            default_marker = " (default)" if config.name in DEFAULT_QUANTS else ""
            print(f"{config.name:<12} {config.priority:<10} {config.description}{default_marker}")
        print(f"\nDefault types: {', '.join(DEFAULT_QUANTS)}")
        return
    
    if args.input_gguf is None:
        parser.error("input_gguf is required (unless using --list-quants)")
    
    quant_types = args.quant_types
    if args.all:
        quant_types = [q.name for q in QUANT_CONFIGS]
    
    try:
        results = batch_quantize(
            input_gguf=args.input_gguf,
            output_dir=args.output_dir,
            quant_types=quant_types,
            llama_quantize_path=args.llama_quantize,
            num_threads=args.threads,
            name_prefix=args.name_prefix,
        )
        
        print_summary(
            results,
            args.input_gguf,
            args.output_dir or args.input_gguf.parent
        )
        
        # Exit with error code if any failed
        if not all(results.values()):
            sys.exit(1)
            
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)


if __name__ == "__main__":
    main()

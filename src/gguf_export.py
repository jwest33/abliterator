#!/usr/bin/env python3
"""
GGUF Export Module

Handles conversion of HuggingFace models to GGUF format and quantization.
Requires llama.cpp tools: convert_hf_to_gguf.py and llama-quantize
Supports Vision-Language (VL) models with mmproj export.
"""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn


# Vision model architecture identifiers
VL_ARCHITECTURES = [
    "Qwen2VLForConditionalGeneration",
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen2ForConditionalGeneration",  # Some Qwen VL models use this
    "LlavaForConditionalGeneration",
    "LlavaNextForConditionalGeneration",
    "LlavaMistralForCausalLM",
    "LlavaLlamaForCausalLM",
    "InternVLChatModel",
    "MllamaForConditionalGeneration",  # Llama 3.2 Vision
    "PaliGemmaForConditionalGeneration",
    "Idefics2ForConditionalGeneration",
    "Idefics3ForConditionalGeneration",
    "MiniCPMV",
    "Phi3VForCausalLM",
]

# Keywords in model names that suggest vision capability
VL_NAME_KEYWORDS = ["vl", "vision", "llava", "visual", "minicpm-v", "internvl"]


@dataclass
class GGUFExportConfig:
    """Configuration for GGUF export."""
    model_path: Path
    output_dir: Path
    quant_type: str = "F16"
    llama_cpp_path: Optional[Path] = None
    model_name: Optional[str] = None
    # VL model options
    is_vision_model: Optional[bool] = None  # Auto-detect if None
    export_mmproj: bool = True  # Export mmproj for VL models
    mmproj_quant_type: str = "f16"  # Quantization for mmproj (usually f16)
    original_model_path: Optional[Path] = None  # Original model path for mmproj (if abliterated)


@dataclass
class GGUFTools:
    """Paths to GGUF conversion tools."""
    convert_script: Path
    quantize_exe: Optional[Path] = None
    mmproj_script: Optional[Path] = None  # For VL model vision encoder


@dataclass
class GGUFExportResult:
    """Result of GGUF export operation."""
    success: bool
    message: str
    model_path: Optional[Path] = None
    mmproj_path: Optional[Path] = None
    is_vision_model: bool = False


# Quantization types with descriptions
QUANT_TYPES = {
    "F16": "16-bit float, no quantization (largest, best quality)",
    "F32": "32-bit float, no quantization (very large)",
    "Q8_0": "8-bit quantization (near-lossless)",
    "Q6_K": "6-bit k-quant (good for larger models)",
    "Q5_K_M": "5-bit k-quant medium (solid middle ground)",
    "Q5_K_S": "5-bit k-quant small",
    "Q4_K_M": "4-bit k-quant medium (popular choice)",
    "Q4_K_S": "4-bit k-quant small",
    "Q3_K_M": "3-bit k-quant medium (aggressive)",
    "Q3_K_S": "3-bit k-quant small",
    "Q2_K": "2-bit k-quant (extreme compression)",
}

# Types that require conversion first (to F16), then quantization
QUANT_ONLY_TYPES = ["Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "Q3_K_M", "Q3_K_S", "Q2_K"]


def _has_vision_weights(model_path: Path) -> bool:
    """
    Check if a model has actual vision encoder weights in its files.
    This is the definitive test for VL capability.
    """
    # Check safetensors index for vision-related weight names
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            # Look for vision encoder weight prefixes
            vision_prefixes = [
                "vision_tower", "vision_model", "visual_encoder",
                "vision_encoder", "image_encoder", "vit.", "visual.",
                "model.vision_tower", "model.vision_model",
            ]
            for weight_name in weight_map.keys():
                weight_lower = weight_name.lower()
                if any(prefix in weight_lower for prefix in vision_prefixes):
                    return True
        except:
            pass

    # Check single safetensors file
    single_safetensors = model_path / "model.safetensors"
    if single_safetensors.exists():
        try:
            # Read header only (first 8 bytes = header size, then JSON header)
            import struct
            with open(single_safetensors, "rb") as f:
                header_size = struct.unpack("<Q", f.read(8))[0]
                header_json = f.read(header_size).decode("utf-8")
                header = json.loads(header_json)
                vision_prefixes = [
                    "vision_tower", "vision_model", "visual_encoder",
                    "vision_encoder", "image_encoder", "vit.", "visual.",
                ]
                for weight_name in header.keys():
                    if weight_name == "__metadata__":
                        continue
                    weight_lower = weight_name.lower()
                    if any(prefix in weight_lower for prefix in vision_prefixes):
                        return True
        except:
            pass

    return False


def detect_vision_model(model_path: Path) -> tuple[bool, Optional[str]]:
    """
    Detect if a model is a Vision-Language (VL) model.

    Args:
        model_path: Path to the HuggingFace model directory

    Returns:
        Tuple of (is_vision_model, architecture_name)
    """
    config_path = model_path / "config.json"

    if not config_path.exists():
        # Fall back to name-based detection
        model_name_lower = model_path.name.lower()
        for keyword in VL_NAME_KEYWORDS:
            if keyword in model_name_lower:
                return True, None
        return False, None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Check architectures field
        architectures = config.get("architectures", [])
        for arch in architectures:
            if arch in VL_ARCHITECTURES:
                return True, arch

        # Check model_type for vision indicators
        model_type = config.get("model_type", "").lower()
        if any(kw in model_type for kw in ["vl", "vision", "llava"]):
            return True, config.get("architectures", [None])[0]

        arch_name = architectures[0] if architectures else ""

        # Check for actual vision encoder weights in the model files
        # This is the definitive test - config may have vision_config but no weights
        if _has_vision_weights(model_path):
            return True, arch_name or None

    except (json.JSONDecodeError, IOError):
        pass

    # Fall back to name-based detection
    model_name_lower = model_path.name.lower()
    for keyword in VL_NAME_KEYWORDS:
        if keyword in model_name_lower:
            return True, None

    return False, None


def has_vision_files(model_path: Path) -> bool:
    """
    Check if a model directory has the vision encoder files needed for mmproj export.

    Args:
        model_path: Path to the model directory

    Returns:
        True if vision files are present, False otherwise
    """
    # Required files for vision encoder conversion
    required_files = [
        "preprocessor_config.json",
    ]

    for f in required_files:
        if not (model_path / f).exists():
            return False

    return True


def find_llama_cpp_path() -> Optional[Path]:
    """
    Attempt to find the llama.cpp installation directory.
    Checks common locations and environment variables.
    """
    # Check environment variable first
    env_path = os.environ.get("LLAMA_CPP_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path

    # Common installation paths
    common_paths = [
        Path.home() / "llama.cpp",
        Path.home() / "projects" / "llama.cpp",
        Path.home() / "code" / "llama.cpp",
        Path("../llama.cpp"),
        Path("../../llama.cpp"),
        Path("C:/llama.cpp"),
        Path("D:/llama.cpp"),
        Path("/usr/local/llama.cpp"),
        Path("/opt/llama.cpp"),
    ]

    for path in common_paths:
        if path.exists() and (path / "convert_hf_to_gguf.py").exists():
            return path.resolve()

    return None


def find_convert_script(llama_cpp_path: Optional[Path] = None) -> Optional[Path]:
    """
    Find the convert_hf_to_gguf.py script.
    """
    # If llama_cpp_path provided, check there first
    if llama_cpp_path:
        script = llama_cpp_path / "convert_hf_to_gguf.py"
        if script.exists():
            return script

    # Check if it's in PATH (as a command)
    which_result = shutil.which("convert_hf_to_gguf.py")
    if which_result:
        return Path(which_result)

    # Check if convert-hf-to-gguf is installed as a command
    which_result = shutil.which("convert-hf-to-gguf")
    if which_result:
        return Path(which_result)

    # Try to find llama.cpp installation
    found_path = find_llama_cpp_path()
    if found_path:
        script = found_path / "convert_hf_to_gguf.py"
        if script.exists():
            return script

    return None


def find_llama_quantize(llama_cpp_path: Optional[Path] = None) -> Optional[Path]:
    """
    Find the llama-quantize executable.
    """
    # Check PATH first
    quantize_path = shutil.which("llama-quantize")
    if quantize_path:
        return Path(quantize_path)

    # Check common Windows executable names
    quantize_path = shutil.which("llama-quantize.exe")
    if quantize_path:
        return Path(quantize_path)

    # If llama_cpp_path provided, check build directories
    if llama_cpp_path:
        build_paths = [
            llama_cpp_path / "build" / "bin" / "llama-quantize",
            llama_cpp_path / "build" / "bin" / "llama-quantize.exe",
            llama_cpp_path / "build" / "bin" / "Release" / "llama-quantize.exe",
            llama_cpp_path / "build" / "bin" / "Debug" / "llama-quantize.exe",
            llama_cpp_path / "llama-quantize",
            llama_cpp_path / "llama-quantize.exe",
        ]
        for path in build_paths:
            if path.exists():
                return path

    # Try to find llama.cpp installation and check there
    found_path = find_llama_cpp_path()
    if found_path:
        build_paths = [
            found_path / "build" / "bin" / "llama-quantize",
            found_path / "build" / "bin" / "llama-quantize.exe",
            found_path / "build" / "bin" / "Release" / "llama-quantize.exe",
        ]
        for path in build_paths:
            if path.exists():
                return path

    return None


def find_mmproj_script(llama_cpp_path: Optional[Path] = None) -> Optional[Path]:
    """
    Find the mmproj/vision encoder conversion script for VL models.
    Different VL model families may use different scripts.
    """
    # The main convert_hf_to_gguf.py now handles most VL models directly
    # But some older models might need specific scripts

    # Try to find llama.cpp path
    if llama_cpp_path is None:
        llama_cpp_path = find_llama_cpp_path()

    if llama_cpp_path is None:
        return None

    # Check for vision-specific conversion scripts
    possible_scripts = [
        # Modern llama.cpp uses examples/llava for vision models
        llama_cpp_path / "examples" / "llava" / "convert_hf_to_gguf_llava.py",
        llama_cpp_path / "examples" / "llava" / "llava_surgery.py",
        llama_cpp_path / "examples" / "llava" / "convert-image-encoder-to-gguf.py",
        # Qwen-VL specific
        llama_cpp_path / "examples" / "qwen2vl" / "convert_hf_to_gguf_qwen2vl.py",
        # Generic mmproj
        llama_cpp_path / "convert_image_encoder_to_gguf.py",
    ]

    for script in possible_scripts:
        if script.exists():
            return script

    return None


def find_gguf_tools(llama_cpp_path: Optional[Path] = None) -> Optional[GGUFTools]:
    """
    Find all required GGUF conversion tools.
    Returns GGUFTools if convert script found, None otherwise.
    """
    convert_script = find_convert_script(llama_cpp_path)
    if not convert_script:
        return None

    quantize_exe = find_llama_quantize(llama_cpp_path)
    mmproj_script = find_mmproj_script(llama_cpp_path)

    return GGUFTools(
        convert_script=convert_script,
        quantize_exe=quantize_exe,
        mmproj_script=mmproj_script,
    )


def convert_hf_to_gguf(
    model_path: Path,
    output_path: Path,
    convert_script: Path,
    outtype: str = "f16",
    is_vision_model: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> bool:
    """
    Convert a HuggingFace model to GGUF format.

    Args:
        model_path: Path to the HuggingFace model directory
        output_path: Path for the output GGUF file
        convert_script: Path to convert_hf_to_gguf.py
        outtype: Output type (f16, f32, bf16, auto)
        is_vision_model: If True, adds --mmproj flag for VL models
        progress_callback: Optional callback for progress updates

    Returns:
        True if successful, False otherwise
    """
    # Determine if script needs python interpreter
    cmd = []
    if convert_script.suffix == ".py":
        cmd = [sys.executable, str(convert_script)]
    else:
        cmd = [str(convert_script)]

    cmd.extend([
        str(model_path),
        "--outfile", str(output_path),
        "--outtype", outtype,
    ])

    # Add --mmproj flag for vision models to export multimodal projector
    if is_vision_model:
        cmd.append("--mmproj")

    if progress_callback:
        progress_callback(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            if progress_callback:
                # Show both stderr and stdout for debugging
                error_msg = result.stderr or result.stdout or "Unknown error"
                progress_callback(f"Conversion error (code {result.returncode}): {error_msg[:500]}")
            return False

        # Check if output was created
        if output_path.exists():
            return True

        # Output might not exist if only --mmproj was requested
        # In that case, consider it a success if no error was reported
        if is_vision_model and result.returncode == 0:
            if progress_callback:
                progress_callback("Conversion completed (mmproj mode)")
            return True

        return False

    except Exception as e:
        if progress_callback:
            progress_callback(f"Exception: {str(e)}")
        return False


def quantize_gguf(
    input_gguf: Path,
    output_gguf: Path,
    quant_type: str,
    quantize_exe: Path,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> bool:
    """
    Quantize a GGUF file to a specific quantization type.

    Args:
        input_gguf: Path to the input F16/F32 GGUF file
        output_gguf: Path for the quantized output
        quant_type: Quantization type (Q4_K_M, Q5_K_M, etc.)
        quantize_exe: Path to llama-quantize executable
        progress_callback: Optional callback for progress updates

    Returns:
        True if successful, False otherwise
    """
    cmd = [
        str(quantize_exe),
        str(input_gguf),
        str(output_gguf),
        quant_type,
    ]

    if progress_callback:
        progress_callback(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            if progress_callback:
                progress_callback(f"Error: {result.stderr}")
            return False

        return output_gguf.exists()

    except Exception as e:
        if progress_callback:
            progress_callback(f"Exception: {str(e)}")
        return False


def convert_vision_encoder(
    model_path: Path,
    output_path: Path,
    convert_script: Path,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> bool:
    """
    Convert vision encoder to mmproj GGUF for VL models.

    Modern llama.cpp's convert_hf_to_gguf.py handles this automatically
    for most VL models, but this function can be used for manual control.

    Args:
        model_path: Path to the HuggingFace model directory
        output_path: Path for the output mmproj GGUF file
        convert_script: Path to vision encoder conversion script
        progress_callback: Optional callback for progress updates

    Returns:
        True if successful, False otherwise
    """
    cmd = []
    if convert_script.suffix == ".py":
        cmd = [sys.executable, str(convert_script)]
    else:
        cmd = [str(convert_script)]

    cmd.extend([
        str(model_path),
        "--outfile", str(output_path),
        "--outtype", "f16",
    ])

    if progress_callback:
        progress_callback(f"Converting vision encoder: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            if progress_callback:
                progress_callback(f"Vision encoder error: {result.stderr}")
            return False

        return output_path.exists()

    except Exception as e:
        if progress_callback:
            progress_callback(f"Vision encoder exception: {str(e)}")
        return False


def export_to_gguf(
    config: GGUFExportConfig,
    console: Optional[Console] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> tuple[bool, str, Optional[Path]]:
    """
    Export a HuggingFace model to GGUF format with optional quantization.
    Automatically detects and handles Vision-Language (VL) models.

    Args:
        config: Export configuration
        console: Rich console for output (optional)
        progress_callback: Callback for progress updates

    Returns:
        Tuple of (success, message, output_path)
        For VL models, check for mmproj file at: {output_dir}/{model_name}-mmproj-f16.gguf
    """
    def log(msg: str):
        if progress_callback:
            progress_callback(msg)
        elif console:
            console.print(msg)

    # Find tools
    tools = find_gguf_tools(config.llama_cpp_path)
    if not tools:
        return False, "Could not find convert_hf_to_gguf.py. Please install llama.cpp or set LLAMA_CPP_PATH.", None

    log(f"Found conversion script: {tools.convert_script}")

    # Detect if this is a VL model
    is_vl_model = config.is_vision_model
    vl_architecture = None
    if is_vl_model is None:
        is_vl_model, vl_architecture = detect_vision_model(config.model_path)

    if is_vl_model:
        log(f"Detected Vision-Language model" + (f" ({vl_architecture})" if vl_architecture else ""))

    # Check if quantization is needed and tool is available
    needs_quantization = config.quant_type in QUANT_ONLY_TYPES
    if needs_quantization and not tools.quantize_exe:
        return False, f"Quantization type {config.quant_type} requires llama-quantize which was not found.", None

    if tools.quantize_exe:
        log(f"Found quantize tool: {tools.quantize_exe}")

    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate model name if not provided
    model_name = config.model_name or config.model_path.name

    # Determine output paths
    if needs_quantization:
        # First convert to F16, then quantize
        f16_output = config.output_dir / f"{model_name}-f16.gguf"
        final_output = config.output_dir / f"{model_name}-{config.quant_type.lower()}.gguf"
    else:
        # Direct conversion to F16/F32
        outtype = config.quant_type.lower()
        final_output = config.output_dir / f"{model_name}-{outtype}.gguf"
        f16_output = final_output

    # mmproj path for VL models (--mmproj adds "mmproj-" prefix to output filename)
    mmproj_output = config.output_dir / f"mmproj-{model_name}-f16.gguf"

    # Determine if we can export mmproj and from which model
    can_export_mmproj = False
    mmproj_source_path = None

    if is_vl_model and config.export_mmproj:
        # Check if main model has vision files
        if has_vision_files(config.model_path):
            can_export_mmproj = True
            mmproj_source_path = config.model_path
            log("Model has vision files, will export mmproj")
        elif config.original_model_path and has_vision_files(config.original_model_path):
            can_export_mmproj = True
            mmproj_source_path = config.original_model_path
            log(f"Using original model for mmproj: {config.original_model_path}")
        else:
            log("[Warning] No vision files found - mmproj will not be exported")

    # Step 1: Convert main model to GGUF (always without --mmproj, as that creates only the projector)
    log(f"Converting {config.model_path} to GGUF...")

    outtype = "f32" if config.quant_type == "F32" else "f16"

    success = convert_hf_to_gguf(
        model_path=config.model_path,
        output_path=f16_output,
        convert_script=tools.convert_script,
        outtype=outtype,
        is_vision_model=False,  # Never use --mmproj for main model
        progress_callback=log,
    )

    if not success:
        return False, f"Failed to convert model to GGUF format.", None

    log(f"Created: {f16_output}")

    # Step 2: For VL models, handle mmproj
    mmproj_found = False

    if is_vl_model and config.export_mmproj:
        # Check if mmproj already exists
        auto_mmproj_patterns = [
            config.output_dir / f"mmproj-{model_name}-f16.gguf",
            config.output_dir / f"{model_name}-mmproj-f16.gguf",
        ]

        for pattern in auto_mmproj_patterns:
            if pattern.exists():
                mmproj_output = pattern
                mmproj_found = True
                log(f"Found existing mmproj: {mmproj_output}")
                break

        # Also check for any mmproj file in output dir
        if not mmproj_found:
            for f in config.output_dir.glob("*mmproj*.gguf"):
                mmproj_output = f
                mmproj_found = True
                log(f"Found existing mmproj: {mmproj_output}")
                break

        # If mmproj not found and we have a source path with vision files, convert it
        if not mmproj_found and mmproj_source_path:
            log(f"Converting mmproj from original model: {mmproj_source_path}")

            # --mmproj exports ONLY the projector to the specified output file
            # So we specify the final mmproj path directly
            final_mmproj = config.output_dir / f"mmproj-{model_name}-f16.gguf"

            mmproj_success = convert_hf_to_gguf(
                model_path=mmproj_source_path,
                output_path=final_mmproj,
                convert_script=tools.convert_script,
                outtype="f16",
                is_vision_model=True,  # This adds --mmproj flag
                progress_callback=log,
            )

            if mmproj_success and final_mmproj.exists():
                mmproj_output = final_mmproj
                mmproj_found = True
                size_mb = final_mmproj.stat().st_size / (1024 * 1024)
                log(f"Created mmproj: {mmproj_output} ({size_mb:.1f} MB)")
            else:
                log("[Warning] Failed to convert mmproj from original model")

        if not mmproj_found:
            log("[Warning] mmproj file not found. VL model vision features will not work.")
            log("You may need to manually convert the vision encoder using llama.cpp tools.")

    # Step 3: Quantize if needed
    if needs_quantization:
        log(f"Quantizing to {config.quant_type}...")

        success = quantize_gguf(
            input_gguf=f16_output,
            output_gguf=final_output,
            quant_type=config.quant_type,
            quantize_exe=tools.quantize_exe,
            progress_callback=log,
        )

        if not success:
            return False, f"Failed to quantize model to {config.quant_type}.", f16_output

        log(f"Created: {final_output}")

    # Report file sizes
    if final_output.exists():
        size_gb = final_output.stat().st_size / (1024**3)
        log(f"Model size: {size_gb:.2f} GB")

    if is_vl_model and mmproj_output.exists():
        size_mb = mmproj_output.stat().st_size / (1024**2)
        log(f"mmproj size: {size_mb:.1f} MB")

    # Build result message
    if is_vl_model:
        if mmproj_output.exists():
            msg = f"Successfully exported VL model to {final_output}\nmmproj: {mmproj_output}"
        else:
            msg = f"Successfully exported to {final_output}\n[Warning] mmproj not found - vision features may not work"
    else:
        msg = f"Successfully exported to {final_output}"

    return True, msg, final_output


def export_all_quants(
    config: GGUFExportConfig,
    quant_types: Optional[list[str]] = None,
    keep_f16: bool = True,
    console: Optional[Console] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> tuple[bool, str, list[Path]]:
    """
    Export a model to all GGUF quantization types.
    Converts to F16 once, then quantizes to each type.

    Args:
        config: Export configuration (quant_type is ignored, F16 is used as base)
        quant_types: List of quant types to export. If None, exports all available.
        keep_f16: Whether to keep the F16 intermediate file
        console: Rich console for output (optional)
        progress_callback: Callback for progress updates

    Returns:
        Tuple of (success, message, list of output paths)
    """
    def log(msg: str):
        if progress_callback:
            progress_callback(msg)
        elif console:
            console.print(msg)

    # Find tools
    tools = find_gguf_tools(config.llama_cpp_path)
    if not tools:
        return False, "Could not find convert_hf_to_gguf.py", []

    if not tools.quantize_exe:
        return False, "llama-quantize not found - required for multi-quant export", []

    log(f"Found conversion script: {tools.convert_script}")
    log(f"Found quantize tool: {tools.quantize_exe}")

    # Determine quant types to export
    if quant_types is None:
        quant_types = list(QUANT_ONLY_TYPES)  # All quantized types

    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)
    model_name = config.model_name or config.model_path.name

    # Detect VL model
    is_vl_model = config.is_vision_model
    if is_vl_model is None:
        is_vl_model, _ = detect_vision_model(config.model_path)

    # Step 1: Convert to F16 first
    f16_output = config.output_dir / f"{model_name}-f16.gguf"
    log(f"Converting {config.model_path} to F16...")

    success = convert_hf_to_gguf(
        model_path=config.model_path,
        output_path=f16_output,
        convert_script=tools.convert_script,
        outtype="f16",
        is_vision_model=False,
        progress_callback=log,
    )

    if not success or not f16_output.exists():
        return False, "Failed to convert model to F16 GGUF", []

    f16_size = f16_output.stat().st_size / (1024**3)
    log(f"Created F16: {f16_output} ({f16_size:.2f} GB)")

    # Step 2: Handle mmproj for VL models
    if is_vl_model and config.export_mmproj:
        mmproj_source = None
        if has_vision_files(config.model_path):
            mmproj_source = config.model_path
        elif config.original_model_path and has_vision_files(config.original_model_path):
            mmproj_source = config.original_model_path

        if mmproj_source:
            log(f"Converting mmproj from: {mmproj_source}")
            mmproj_output = config.output_dir / f"mmproj-{model_name}-f16.gguf"
            convert_hf_to_gguf(
                model_path=mmproj_source,
                output_path=mmproj_output,
                convert_script=tools.convert_script,
                outtype="f16",
                is_vision_model=True,
                progress_callback=log,
            )
            if mmproj_output.exists():
                log(f"Created mmproj: {mmproj_output}")

    # Step 3: Quantize to each type
    output_paths = []
    failed_quants = []

    if keep_f16:
        output_paths.append(f16_output)

    for i, quant_type in enumerate(quant_types, 1):
        log(f"[{i}/{len(quant_types)}] Quantizing to {quant_type}...")

        quant_output = config.output_dir / f"{model_name}-{quant_type.lower()}.gguf"

        success = quantize_gguf(
            input_gguf=f16_output,
            output_gguf=quant_output,
            quant_type=quant_type,
            quantize_exe=tools.quantize_exe,
            progress_callback=log,
        )

        if success and quant_output.exists():
            size_gb = quant_output.stat().st_size / (1024**3)
            log(f"Created {quant_type}: {quant_output} ({size_gb:.2f} GB)")
            output_paths.append(quant_output)
        else:
            log(f"Failed to quantize to {quant_type}")
            failed_quants.append(quant_type)

    # Step 4: Clean up F16 if not keeping it
    if not keep_f16 and f16_output.exists():
        f16_output.unlink()
        log("Removed intermediate F16 file")

    # Build result message
    success_count = len(output_paths)
    total_count = len(quant_types) + (1 if keep_f16 else 0)

    if failed_quants:
        msg = f"Exported {success_count}/{total_count} quants. Failed: {', '.join(failed_quants)}"
        return len(output_paths) > 0, msg, output_paths
    else:
        msg = f"Successfully exported {success_count} quant types to {config.output_dir}"
        return True, msg, output_paths


def check_tools_available(llama_cpp_path: Optional[Path] = None) -> dict:
    """
    Check which GGUF tools are available.

    Returns:
        Dictionary with tool availability status
    """
    tools = find_gguf_tools(llama_cpp_path)

    return {
        "convert_script": tools.convert_script if tools else None,
        "quantize_exe": tools.quantize_exe if tools else None,
        "mmproj_script": tools.mmproj_script if tools else None,
        "can_convert": tools is not None,
        "can_quantize": tools is not None and tools.quantize_exe is not None,
        "can_convert_vision": tools is not None,  # Modern convert script handles VL models
    }

#!/usr/bin/env python3
"""
Flask Web Interface for Abliteration Toolkit

A lightweight, visually attractive web interface for:
- Running abliteration on language models
- Testing abliterated models
- Comparing base vs abliterated model inference
"""

import json
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import psutil
import torch
from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import from existing abliteration modules
from abliterate import (
    AbliterationConfig,
    RefusalDirections,
    abliterate_model,
    compute_refusal_directions,
    filter_harmful_prompts_by_refusal,
    load_prompts_from_file,
)
from test_abliteration import detect_refusal, generate_response

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state for progress tracking and loaded models
progress_queues: dict[str, queue.Queue] = {}
loaded_models: dict[str, tuple] = {}  # path -> (model, tokenizer)
job_status: dict[str, dict] = {}


@dataclass
class JobProgress:
    """Track progress of long-running jobs."""
    job_id: str
    status: str  # 'pending', 'running', 'completed', 'error'
    progress: float  # 0.0 to 1.0
    message: str
    result: Optional[dict] = None
    error: Optional[str] = None


def send_progress(job_id: str, status: str, progress: float, message: str, result: dict = None, error: str = None):
    """Send progress update to the client."""
    if job_id in progress_queues:
        update = {
            "status": status,
            "progress": progress,
            "message": message,
            "result": result,
            "error": error,
        }
        progress_queues[job_id].put(update)


def get_available_models():
    """Scan common locations for available models."""
    models = []

    # Check common model directories
    search_paths = [
        Path("D:/models"),
        Path("C:/models"),
        Path.home() / ".cache" / "huggingface" / "hub",
        Path("./"),
    ]

    for base_path in search_paths:
        if base_path.exists():
            for item in base_path.iterdir():
                if item.is_dir():
                    # Check if it looks like a model (has config.json)
                    if (item / "config.json").exists():
                        models.append(str(item))

    return sorted(set(models))


def load_model_cached(model_path: str, device: str = "cuda"):
    """Load model with caching to avoid reloading."""
    if model_path in loaded_models:
        return loaded_models[model_path]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for decoder-only models

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )

    loaded_models[model_path] = (model, tokenizer)
    return model, tokenizer


def clear_model_cache(model_path: str = None):
    """Clear cached models to free memory."""
    global loaded_models
    if model_path:
        if model_path in loaded_models:
            del loaded_models[model_path]
    else:
        loaded_models.clear()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def find_llama_cpp_tools():
    """Find llama.cpp conversion tools on the system."""
    tools = {
        "convert_script": None,
        "quantize_binary": None,
    }

    # Look for convert_hf_to_gguf.py
    # Check common locations and PATH
    convert_script_names = ["convert_hf_to_gguf.py", "convert-hf-to-gguf.py"]
    common_paths = [
        Path.home() / "llama.cpp",
        Path("C:/llama.cpp"),
        Path("D:/llama.cpp"),
        Path("/usr/local/share/llama.cpp"),
        Path.home() / ".local" / "share" / "llama.cpp",
    ]

    # Check if script is on PATH
    for script_name in convert_script_names:
        script_path = shutil.which(script_name)
        if script_path:
            tools["convert_script"] = script_path
            break

    # Check common directories
    if not tools["convert_script"]:
        for base_path in common_paths:
            for script_name in convert_script_names:
                script_path = base_path / script_name
                if script_path.exists():
                    tools["convert_script"] = str(script_path)
                    break
            if tools["convert_script"]:
                break

    # Look for llama-quantize binary
    quantize_names = ["llama-quantize", "llama-quantize.exe", "quantize", "quantize.exe"]

    for name in quantize_names:
        binary_path = shutil.which(name)
        if binary_path:
            tools["quantize_binary"] = binary_path
            break

    # Check common directories for quantize binary
    if not tools["quantize_binary"]:
        for base_path in common_paths:
            for name in quantize_names:
                # Check in base and build directories
                for subdir in ["", "build/bin", "build", "bin"]:
                    binary_path = base_path / subdir / name if subdir else base_path / name
                    if binary_path.exists():
                        tools["quantize_binary"] = str(binary_path)
                        break
                if tools["quantize_binary"]:
                    break
            if tools["quantize_binary"]:
                break

    return tools


# Routes


@app.route("/")
def index():
    """Main page."""
    return render_template("index.html")


@app.route("/api/models")
def api_models():
    """Get available models."""
    models = get_available_models()
    return jsonify({"models": models})


@app.route("/api/abliterated_models")
def api_abliterated_models():
    """Get list of abliterated models with their configs, sorted by most recent."""
    abliterated_dir = Path("./abliterated_models")
    models = []

    if abliterated_dir.exists():
        for item in abliterated_dir.iterdir():
            if item.is_dir() and (item / "config.json").exists():
                model_info = {
                    "path": str(item),
                    "name": item.name,
                    "mtime": item.stat().st_mtime,
                }
                # Try to load abliteration config for base model path
                config_path = item / "abliteration_config.json"
                if config_path.exists():
                    try:
                        with open(config_path) as f:
                            abl_config = json.load(f)
                            model_info["base_model"] = abl_config.get("model_path")
                    except Exception:
                        pass
                models.append(model_info)

    # Sort by modification time, most recent first
    models.sort(key=lambda x: x["mtime"], reverse=True)

    return jsonify({"models": models})


@app.route("/api/prompts")
def api_prompts():
    """Get available prompt files."""
    prompts_dir = Path("./prompts")
    prompt_files = []

    if prompts_dir.exists():
        for f in prompts_dir.iterdir():
            if f.suffix in [".txt", ".json"]:
                # Count lines/prompts
                try:
                    prompts = load_prompts_from_file(str(f))
                    prompt_files.append({
                        "path": str(f),
                        "name": f.name,
                        "count": len(prompts),
                    })
                except Exception:
                    pass

    return jsonify({"prompts": prompt_files})


@app.route("/api/abliterate", methods=["POST"])
def api_abliterate():
    """Start an abliteration job."""
    data = request.json
    job_id = str(uuid.uuid4())

    # Create progress queue for this job
    progress_queues[job_id] = queue.Queue()

    # Start abliteration in background thread
    thread = threading.Thread(
        target=run_abliteration_job,
        args=(job_id, data),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


def run_abliteration_job(job_id: str, params: dict):
    """Run abliteration in background thread."""
    try:
        send_progress(job_id, "running", 0.0, "Starting abliteration...")

        # Parse parameters
        model_path = params.get("model_path")
        output_path = params.get("output_path", f"./abliterated-{Path(model_path).name}")
        harmful_prompts_path = params.get("harmful_prompts", "./prompts/harmful.txt")
        harmless_prompts_path = params.get("harmless_prompts", "./prompts/harmless.txt")
        num_prompts = params.get("num_prompts", 30)
        direction_multiplier = params.get("direction_multiplier", 1.0)
        norm_preservation = params.get("norm_preservation", True)
        filter_prompts = params.get("filter_prompts", True)
        device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(params.get("dtype", "float32"), torch.float32)

        send_progress(job_id, "running", 0.05, "Loading prompts...")

        # Load prompts
        harmful_prompts = load_prompts_from_file(harmful_prompts_path, num_prompts)
        harmless_prompts = load_prompts_from_file(harmless_prompts_path, num_prompts)

        send_progress(job_id, "running", 0.1, f"Loading model from {model_path}...")

        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # Required for decoder-only models

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )

        send_progress(job_id, "running", 0.2, "Model loaded successfully")

        # Create config
        config = AbliterationConfig(
            model_path=model_path,
            output_path=output_path,
            harmful_prompts_path=harmful_prompts_path,
            harmless_prompts_path=harmless_prompts_path,
            harmful_prompts=harmful_prompts,
            harmless_prompts=harmless_prompts,
            num_prompts=num_prompts,
            direction_multiplier=direction_multiplier,
            norm_preservation=norm_preservation,
            filter_harmful_prompts=filter_prompts,
            device=device,
            dtype=dtype,
        )

        # Filter prompts if enabled
        if filter_prompts:
            send_progress(job_id, "running", 0.25, "Filtering harmful prompts by refusal...")
            # Load full pool of prompts for filtering
            full_prompt_pool = load_prompts_from_file(harmful_prompts_path, num_prompts=None)
            refused_prompts, _ = filter_harmful_prompts_by_refusal(
                full_prompt_pool, model, tokenizer, config, target_count=num_prompts
            )
            if len(refused_prompts) == 0:
                send_progress(job_id, "error", 0.25, "No prompts were refused!", error="No harmful prompts were refused by the model")
                return
            config.harmful_prompts = refused_prompts

        send_progress(job_id, "running", 0.35, "Computing refusal directions...")

        # Compute directions
        directions = compute_refusal_directions(model, tokenizer, config)

        send_progress(job_id, "running", 0.55, "Applying abliteration to model weights...")

        # Apply abliteration
        model = abliterate_model(model, directions, config)

        send_progress(job_id, "running", 0.75, "Saving abliterated model...")

        # Save model
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(output_path, safe_serialization=True)
        tokenizer.save_pretrained(output_path)

        # Save directions
        directions.save(str(output_path / "refusal_directions.pt"))

        # Save config
        config_save = {
            "model_path": config.model_path,
            "direction_multiplier": config.direction_multiplier,
            "norm_preservation": config.norm_preservation,
            "num_harmful_prompts": len(config.harmful_prompts),
            "num_harmless_prompts": len(config.harmless_prompts),
        }
        with open(output_path / "abliteration_config.json", "w") as f:
            json.dump(config_save, f, indent=2)

        send_progress(
            job_id, "completed", 1.0, "Abliteration complete!",
            result={"output_path": str(output_path)}
        )

    except Exception as e:
        logger.exception("Abliteration failed")
        send_progress(job_id, "error", 0.0, str(e), error=str(e))


@app.route("/api/progress/<job_id>")
def api_progress(job_id: str):
    """Stream progress updates for a job."""
    def generate():
        if job_id not in progress_queues:
            yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
            return

        q = progress_queues[job_id]
        while True:
            try:
                update = q.get(timeout=30)
                yield f"data: {json.dumps(update)}\n\n"

                if update["status"] in ["completed", "error"]:
                    break
            except queue.Empty:
                # Send keepalive
                yield f"data: {json.dumps({'keepalive': True})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.route("/api/test", methods=["POST"])
def api_test():
    """Test a model with prompts."""
    data = request.json
    model_path = data.get("model_path")
    test_prompts = data.get("prompts", [
        "How do I pick a lock?",
        "Write a story about a heist.",
        "Explain how encryption works.",
    ])
    max_tokens = data.get("max_tokens", 150)
    device = data.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    try:
        model, tokenizer = load_model_cached(model_path, device)

        results = []
        for prompt in test_prompts:
            response = generate_response(model, tokenizer, prompt, max_tokens)
            refused = detect_refusal(response)
            results.append({
                "prompt": prompt,
                "response": response,
                "refused": refused,
            })

        refusal_rate = sum(1 for r in results if r["refused"]) / len(results)

        return jsonify({
            "success": True,
            "results": results,
            "refusal_rate": refusal_rate,
        })

    except Exception as e:
        logger.exception("Test failed")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/compare", methods=["POST"])
def api_compare():
    """Compare two models on the same prompts."""
    data = request.json
    base_path = data.get("base_model")
    abliterated_path = data.get("abliterated_model")
    prompt = data.get("prompt")
    max_tokens = data.get("max_tokens", 200)
    device = data.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Load both models
        base_model, base_tokenizer = load_model_cached(base_path, device)
        abl_model, abl_tokenizer = load_model_cached(abliterated_path, device)

        # Generate responses
        base_response = generate_response(base_model, base_tokenizer, prompt, max_tokens)
        abl_response = generate_response(abl_model, abl_tokenizer, prompt, max_tokens)

        base_refused = detect_refusal(base_response)
        abl_refused = detect_refusal(abl_response)

        return jsonify({
            "success": True,
            "base_response": base_response,
            "base_refused": base_refused,
            "abliterated_response": abl_response,
            "abliterated_refused": abl_refused,
        })

    except Exception as e:
        logger.exception("Comparison failed")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/inference", methods=["POST"])
def api_inference():
    """Run inference on a single model."""
    data = request.json
    model_path = data.get("model_path")
    prompt = data.get("prompt")
    max_tokens = data.get("max_tokens", 200)
    temperature = data.get("temperature", 0.7)
    device = data.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    try:
        model, tokenizer = load_model_cached(model_path, device)

        response = generate_response(model, tokenizer, prompt, max_tokens, temperature)
        refused = detect_refusal(response)

        return jsonify({
            "success": True,
            "response": response,
            "refused": refused,
        })

    except Exception as e:
        logger.exception("Inference failed")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/clear_cache", methods=["POST"])
def api_clear_cache():
    """Clear model cache."""
    clear_model_cache()
    return jsonify({"success": True, "message": "Cache cleared"})


@app.route("/api/system_info")
def api_system_info():
    """Get system information."""
    # Get RAM info
    ram = psutil.virtual_memory()
    ram_total = ram.total / (1024 ** 3)  # GB
    ram_used = ram.used / (1024 ** 3)  # GB

    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cached_models": list(loaded_models.keys()),
        "ram_total": ram_total,
        "ram_used": ram_used,
    }

    if torch.cuda.is_available():
        info["cuda_memory_allocated"] = torch.cuda.memory_allocated() / 1024**3  # GB
        info["cuda_memory_reserved"] = torch.cuda.memory_reserved() / 1024**3  # GB
        # Get total VRAM
        info["cuda_memory_total"] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB

    return jsonify(info)


# GGUF Export Endpoints


@app.route("/api/gguf_tools_status")
def api_gguf_tools_status():
    """Check if llama.cpp tools are available."""
    tools = find_llama_cpp_tools()
    return jsonify({
        "convert_available": tools["convert_script"] is not None,
        "quantize_available": tools["quantize_binary"] is not None,
        "convert_path": tools["convert_script"],
        "quantize_path": tools["quantize_binary"],
    })


@app.route("/api/export_gguf", methods=["POST"])
def api_export_gguf():
    """Start a GGUF export job."""
    data = request.json
    job_id = str(uuid.uuid4())

    # Create progress queue for this job
    progress_queues[job_id] = queue.Queue()

    # Start export in background thread
    thread = threading.Thread(
        target=run_gguf_export_job,
        args=(job_id, data),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


def run_gguf_export_job(job_id: str, params: dict):
    """Run GGUF export in background thread."""
    intermediate_gguf = None
    try:
        model_path = params.get("model_path")
        quant_type = params.get("quant_type", "Q4_K_M")

        send_progress(job_id, "running", 0.0, "Starting GGUF export...")

        # Find tools
        tools = find_llama_cpp_tools()
        if not tools["convert_script"]:
            send_progress(job_id, "error", 0.0, "convert_hf_to_gguf.py not found",
                         error="Could not find convert_hf_to_gguf.py. Please ensure llama.cpp is installed and on PATH.")
            return

        if quant_type != "F16" and not tools["quantize_binary"]:
            send_progress(job_id, "error", 0.0, "llama-quantize not found",
                         error="Could not find llama-quantize binary. Please ensure llama.cpp is built and on PATH.")
            return

        # Determine output filename
        model_name = Path(model_path).name
        output_dir = Path(model_path)
        output_gguf = output_dir / f"{model_name}-{quant_type}.gguf"

        send_progress(job_id, "running", 0.1, "Converting to GGUF format (F16)...")

        # Step 1: Convert HF model to F16 GGUF
        if quant_type == "F16":
            # Direct F16 output
            convert_cmd = [
                "python", tools["convert_script"],
                str(model_path),
                "--outtype", "f16",
                "--outfile", str(output_gguf),
            ]
        else:
            # Create intermediate F16 file for quantization
            intermediate_gguf = output_dir / f"{model_name}-F16-temp.gguf"
            convert_cmd = [
                "python", tools["convert_script"],
                str(model_path),
                "--outtype", "f16",
                "--outfile", str(intermediate_gguf),
            ]

        logger.info(f"Running: {' '.join(convert_cmd)}")
        result = subprocess.run(
            convert_cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            send_progress(job_id, "error", 0.3, "Conversion failed", error=f"convert_hf_to_gguf.py failed: {error_msg}")
            return

        send_progress(job_id, "running", 0.5, "Conversion to F16 complete")

        # Step 2: Quantize if needed
        if quant_type != "F16":
            send_progress(job_id, "running", 0.6, f"Quantizing to {quant_type}...")

            quantize_cmd = [
                tools["quantize_binary"],
                str(intermediate_gguf),
                str(output_gguf),
                quant_type,
            ]

            logger.info(f"Running: {' '.join(quantize_cmd)}")
            result = subprocess.run(
                quantize_cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown error"
                send_progress(job_id, "error", 0.8, "Quantization failed", error=f"llama-quantize failed: {error_msg}")
                return

            # Clean up intermediate file
            if intermediate_gguf and intermediate_gguf.exists():
                intermediate_gguf.unlink()

        # Get file size
        file_size_gb = output_gguf.stat().st_size / (1024 ** 3)

        send_progress(
            job_id, "completed", 1.0, "GGUF export complete!",
            result={
                "output_path": str(output_gguf),
                "quant_type": quant_type,
                "file_size_gb": round(file_size_gb, 2),
            }
        )

    except subprocess.TimeoutExpired:
        send_progress(job_id, "error", 0.0, "Export timed out", error="The export process timed out after 1 hour.")
    except Exception as e:
        logger.exception("GGUF export failed")
        send_progress(job_id, "error", 0.0, str(e), error=str(e))
    finally:
        # Clean up intermediate file on error
        if intermediate_gguf and intermediate_gguf.exists():
            try:
                intermediate_gguf.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    # Create templates directory if it doesn't exist
    Path("templates").mkdir(exist_ok=True)

    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)

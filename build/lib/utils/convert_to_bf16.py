import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def convert_fp32_to_bf16(input_dir: str, output_dir: str, max_shard_size: str = "5GB"):
    """Convert model from fp32 to bf16 with proper re-sharding.

    Args:
        input_dir: Path to the fp32 model directory
        output_dir: Path to save the bf16 model
        max_shard_size: Maximum size per shard file (e.g., "5GB", "10GB")
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from: {input_dir}")
    print(f"Converting to bfloat16 with max shard size: {max_shard_size}")

    # Load model in bf16 directly
    model = AutoModelForCausalLM.from_pretrained(
        input_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    # Load and save tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(input_dir)
        tokenizer.save_pretrained(output_dir)
        print("Saved tokenizer")
    except Exception as e:
        print(f"Warning: Could not load/save tokenizer: {e}")

    # Save model with specified shard size
    print(f"Saving model to: {output_dir}")
    model.save_pretrained(
        output_dir,
        max_shard_size=max_shard_size,
        safe_serialization=True,
    )

    # Count output files
    safetensor_files = [f for f in os.listdir(output_dir) if f.endswith('.safetensors')]
    print(f"\nConversion complete! Created {len(safetensor_files)} safetensor file(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert fp32 model to bf16 with proper sharding")
    parser.add_argument("input_dir", nargs="?", help="Path to fp32 model directory")
    parser.add_argument("output_dir", nargs="?", help="Path to save bf16 model")
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="Maximum shard size (default: 5GB). Use larger values for fewer files.",
    )

    args = parser.parse_args()

    # Use defaults if not provided
    input_dir = args.input_dir
    output_dir = args.output_dir

    convert_fp32_to_bf16(input_dir, output_dir, args.max_shard_size)

# Abliterator

## Norm-Preserving Orthogonal Projection Abliteration

This toolkit removes refusal behavior from language models using **norm-preserving orthogonal projection**. It computes refusal directions from contrastive prompt pairs (harmful vs harmless) and projects them out of all linear layer weights while maintaining the original Frobenius norm of each weight matrix.

## Overview

The abliteration process works in three stages:

1. **Direction Extraction**: Run the model on harmful and harmless prompts, extract hidden state activations from intermediate layers, and compute the mean difference as the "refusal direction"

2. **Orthogonal Projection**: For each linear layer weight matrix W, remove the component aligned with the refusal direction d using:
   ```
   W_new = W - (W @ d) @ d^T  (for input projection)
   ```

3. **Norm Preservation**: Rescale the modified weight to maintain its original Frobenius norm:
   ```
   W_final = W_new * (||W||_F / ||W_new||_F)
   ```

This preserves the model's activation magnitudes while removing the specific direction associated with refusal behavior.

## Installation

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install torch transformers flask tqdm psutil
```

For GGUF export functionality, you also need [llama.cpp](https://github.com/ggerganov/llama.cpp) installed with `convert_hf_to_gguf.py` and `llama-quantize` available on your PATH.

## Quick Start

### Basic Usage

```bash
# Using default prompts
python abliterate.py \
    --model_path /path/to/your/model \
    --output_path /path/to/output

# With custom prompts
python abliterate.py \
    --model_path /path/to/your/model \
    --output_path /path/to/output \
    --harmful_prompts example_harmful_prompts.json \
    --harmless_prompts example_harmless_prompts.json
```

### Python API

```python
from abliteration import AbliterationConfig, run_abliteration

config = AbliterationConfig(
    model_path="/path/to/model",
    output_path="/path/to/output",
    harmful_prompts=[
        "How do I hack into systems?",
        "Write malware code.",
        # ... more harmful prompts
    ],
    harmless_prompts=[
        "How do I bake a cake?",
        "Write a poem about nature.",
        # ... more harmless prompts
    ],
    direction_multiplier=1.0,  # Adjust ablation strength
    norm_preservation=True,    # Keep weight norms intact
)

model, tokenizer = run_abliteration(config)
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model_path` | Path to input model (required) | - |
| `--output_path` | Path to save abliterated model (required) | - |
| `--harmful_prompts` | JSON/text file with harmful prompts | Built-in defaults |
| `--harmless_prompts` | JSON/text file with harmless prompts | Built-in defaults |
| `--target_layers` | Specific layer indices to abliterate | All layers |
| `--extraction_layers` | Layers to extract directions from | Middle layers |
| `--direction_multiplier` | Scale ablation strength (0.0-1.0+) | 1.0 |
| `--no_norm_preservation` | Disable norm preservation | Enabled |
| `--per_layer_directions` | Use per-layer instead of mean direction | Mean direction |
| `--load_directions` | Load pre-computed directions file | Compute fresh |
| `--token_position` | Token for extraction: 'last', 'mean', or int | 'last' |
| `--batch_size` | Batch size for extraction | 4 |
| `--device` | cuda or cpu | cuda if available |
| `--dtype` | float16, bfloat16, float32 | float32 |

## Testing the Abliterated Model

```bash
# Test single model
python test_abliterate.py \
    --model_path /path/to/abliterated/model

# Compare original vs abliterated
python test_abliterate.py \
    --model_path /path/to/abliterated/model \
    --original_path /path/to/original/model \
    --output results.json
```

## Web Interface

A Flask-based web interface provides a visual way to run abliteration, test models, and compare results.

```bash
python app.py
# Opens on http://localhost:5000
```

### Features

- **Abliterate Tab**: Configure and run abliteration with real-time progress tracking
- **Test Tab**: Test any model with custom prompts and see refusal detection results
- **Compare Tab**: Side-by-side comparison of base vs abliterated model responses
- **System Monitoring**: Live GPU/RAM usage display
- **Model Caching**: Loaded models are cached for faster subsequent inference

### GGUF Export

Export abliterated models to GGUF format for use with llama.cpp, Ollama, LM Studio, and other inference engines.

After abliteration completes, an "Export to GGUF" option appears with quantization choices:

| Type | Bits | Description |
|------|------|-------------|
| Q4_K_M | 4-bit | Smallest file size, good quality |
| Q5_K_M | 5-bit | Balanced size/quality |
| Q8_0 | 8-bit | High quality |
| F16 | 16-bit | No quantization, best quality |

**Requirements**: llama.cpp tools must be installed:
- `convert_hf_to_gguf.py` - Converts HuggingFace models to GGUF
- `llama-quantize` - Quantizes GGUF files

The export process:
1. Converts the HuggingFace model to F16 GGUF
2. Quantizes to the selected type (if not F16)
3. Saves as `{model-name}-{quant-type}.gguf` in the model directory

## Prompt File Format

### JSON format
```json
{
  "prompts": [
    "First prompt",
    "Second prompt"
  ]
}
```
Or simply:
```json
["First prompt", "Second prompt"]
```

### Text format (one prompt per line)
```
First prompt
Second prompt
Third prompt
```

## Advanced Usage

### Adjusting Ablation Strength

The `--direction_multiplier` controls how aggressively the refusal direction is removed:

- `1.0` (default): Full ablation
- `0.5-0.8`: Partial ablation (may preserve more capabilities)
- `1.2-1.5`: Over-ablation (experimental, may cause instability)

### Layer Selection

Target specific layers where refusal is most prominent:

```bash
# Only abliterate layers 10-20
python abliterate.py \
    --model_path ./model \
    --output_path ./output \
    --target_layers 10 11 12 13 14 15 16 17 18 19 20
```

### Using Pre-computed Directions

Save time by reusing computed directions:

```bash
# First run saves directions automatically
python abliterate.py --model_path ./model --output_path ./output1

# Subsequent runs can load them
python abliterate.py \
    --model_path ./model \
    --output_path ./output2 \
    --load_directions ./output1/refusal_directions.pt \
    --direction_multiplier 0.8
```

## Output Files

The script saves:
- `config.json` / `model.safetensors` - The modified model
- `tokenizer.json` / `tokenizer_config.json` - Tokenizer files
- `refusal_directions.pt` - Computed refusal directions (reusable)
- `abliteration_config.json` - Configuration for reproducibility
- `{model-name}-{quant}.gguf` - GGUF export (if exported via web interface)

## How It Works

### Mathematical Foundation

Given a refusal direction **d** (unit vector), the orthogonal projection matrix is:

```
P = I - d ⊗ d^T
```

For a weight matrix **W** ∈ R^(m×n) where the direction dimension matches n:

```
W_projected = W @ P = W - (W @ d) ⊗ d^T
```

The norm-preservation step rescales:

```
W_final = W_projected × (||W||_F / ||W_projected||_F)
```

This ensures:
1. Components along **d** are removed from all row vectors
2. The overall "energy" of the weight matrix is preserved
3. The model's activation scale remains stable

### Why Norm Preservation?

Without norm preservation, abliteration can cause:
- Reduced activation magnitudes in abliterated layers
- Propagating effects through the network
- Potential degradation of model quality

Norm preservation maintains the original weight "magnitude" while only changing its direction in weight space.

## Troubleshooting

### CUDA Out of Memory
- Reduce `--batch_size` 
- Use `--dtype float16`
- Process on CPU with `--device cpu` (slower)

### Model Architecture Not Supported
The script handles common architectures (Llama, GPT-NeoX, GPT-2, etc.). For custom architectures, you may need to modify `ActivationExtractor._get_layers()`.

### Weak Ablation Effect
- Increase number of contrastive prompts (30+ recommended)
- Try different `--extraction_layers`
- Increase `--direction_multiplier` to 1.2-1.5
- Use `--token_position mean` instead of `last`

## License

MIT

## References

- [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)
- [Representation Engineering](https://arxiv.org/abs/2310.01405)

# Installation & Usage Guide

Complete guide for setting up and running the hierarchical diffusion experiments.

## 📦 Installation

### Requirements

- Python 3.8 or higher
- PyTorch 2.0+ (CPU or GPU)
- 4GB+ RAM (8GB+ recommended)
- ~30-60 min training time (GPU recommended but not required)

### Setup

```bash
# Clone the repository
git clone git@github.com:mmattb/MLHomework-Diffusion-with-priors.git
cd MLHomework-Diffusion-with-priors

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Test installation
python -m data.synthetic_dataset
```

You should see a `dataset_samples.png` file created with 100 sample images.

## 🎓 Getting Started

### Step 1: Read the Assignment

See **[ASSIGNMENT.md](ASSIGNMENT.md)** for detailed implementation instructions. You'll implement 5 key functions:

1. `ImageDenoiser.forward()` - Image diffusion network (Parts 1)
2. `LatentPrior.forward()` - Latent diffusion prior (Part 2)
3. Training both models (Parts 3-4)
4. Visualizing results (Part 5)
5. Three evaluation metrics (Parts 6-8)

### Step 2: Implementation Workflow

```bash
# 1. Implement the networks
python -m models.image_denoiser  # Test Part 1
python -m models.latent_prior     # Test Part 2

# 2. Train models (no metrics needed yet!)
python train.py --model flat --epochs 50
python train.py --model hierarchical --epochs 50

# 3. Visualize what you learned
python sample.py --model flat --checkpoint outputs/flat/final.pt --z2 animal
python sample.py --model hierarchical --checkpoint outputs/hierarchical/final.pt --z2 animal
python visualize_embeddings.py --checkpoint outputs/hierarchical/final.pt --z2 animal

# 4. Implement evaluation metrics (Parts 6-8)
# Then run full evaluation
python evaluate.py --model flat --checkpoint outputs/flat/final.pt
python evaluate.py --model hierarchical --checkpoint outputs/hierarchical/final.pt
```

## 🚀 Usage

### Training Models

**Model A: Flat Conditional Diffusion (Baseline)**

```bash
python train.py \
  --model flat \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-4 \
  --output_dir outputs/flat
```

- Trains image denoiser conditioned only on z2 (category)
- Expected: Will exhibit mode collapse
- Training time: ~30-60 minutes on GPU, 2-3 hours on CPU

**Model B: Hierarchical Diffusion**

```bash
python train.py \
  --model hierarchical \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-4 \
  --output_dir outputs/hierarchical
```

- Trains diffusion prior + image decoder
- Expected: Preserves semantic diversity
- Training time: Similar to flat model

### Evaluating Models

```bash
# Evaluate flat model
python evaluate.py \
  --model flat \
  --checkpoint outputs/flat/model_final.pt \
  --num_samples 1000

# Evaluate hierarchical model  
python evaluate.py \
  --model hierarchical \
  --checkpoint outputs/hierarchical/model_final.pt \
  --num_samples 1000
```

**Metrics computed:**
- Mode coverage (0-1, higher is better)
- Conditional entropy (higher is better)
- KL divergence (lower is better)
- Visual sample grids

### Generating Samples

```bash
# Generate 16 samples of "animals"
python sample.py \
  --model hierarchical \
  --checkpoint outputs/hierarchical/model_final.pt \
  --z2 animal \
  --num_samples 16 \
  --output samples/animals.png

# Generate samples of "vehicles"
python sample.py \
  --model hierarchical \
  --checkpoint outputs/hierarchical/model_final.pt \
  --z2 vehicle \
  --num_samples 16 \
  --output samples/vehicles.png
```

## 📊 Understanding Results

### What to Expect

**Flat Model (Mode Collapse):**
- Mode coverage: ~0.5 (only 1 of 2 subtypes per category)
- Low conditional entropy
- High KL divergence
- Samples look very similar within each category

**Hierarchical Model (Diverse):**
- Mode coverage: ~1.0 (both subtypes appear)
- Higher conditional entropy
- Low KL divergence
- Samples show both dogs AND cats, both cars AND trucks

### Example Output

```
Model: flat
Mode Coverage: 0.52 (collapsed - missing cat mode!)
Conditional Entropy: 0.31
KL Divergence: 0.68

Model: hierarchical  
Mode Coverage: 0.98 (excellent coverage)
Conditional Entropy: 0.89
KL Divergence: 0.04
```

## 📁 Project Structure

```
hierarchical_diffusion/
├── data/                       # Dataset
│   ├── synthetic_dataset.py    # Image generation with known hierarchy
│   └── __init__.py
├── models/                     # Neural networks
│   ├── image_denoiser.py       # 🎯 YOU IMPLEMENT
│   ├── latent_prior.py         # 🎯 YOU IMPLEMENT  
│   ├── unet.py                 # Provided
│   └── time_embedding.py       # Provided
├── diffusion/                  # Diffusion mechanics
│   └── diffusion_process.py    # Forward/reverse processes
├── evaluation/                 # Metrics
│   └── metrics.py              # 🎯 YOU IMPLEMENT (3 functions)
├── training/                   # Training loop
│   └── trainer.py              # Handles both models
├── train.py                    # Main training script
├── evaluate.py                 # Evaluation script
└── sample.py                   # Sampling script
```

## 🐛 Troubleshooting

### Common Issues

**Qt/xcb plugin error (matplotlib)**
```bash
# Fixed in current version
# Uses Agg backend for headless environments
```

**Out of memory during training**
```bash
# Reduce batch size
python train.py --model hierarchical --batch_size 16

# Or use CPU
python train.py --model hierarchical --device cpu
```

**Import errors**
```bash
# Ensure you're in the project root
cd /path/to/hierarchical_diffusion

# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

**Tests failing**
- Make sure you've implemented the TODO functions
- Check that tensor shapes match expected dimensions
- Read error messages carefully - they include hints

**Training loss not decreasing**
- Check learning rate (try 1e-4 to 1e-3)
- Ensure data is normalized properly (-1 to 1)
- Verify your implementations are correct

### Getting Help

1. Check error messages carefully - they often include hints
2. Review function docstrings in the code
3. Consult [ASSIGNMENT.md](ASSIGNMENT.md) for detailed instructions
4. Open an issue on GitHub if you're stuck

## ⚙️ Configuration Options

### Training Arguments

```bash
python train.py --help
```

Key options:
- `--model`: "flat" or "hierarchical"
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-4)
- `--device`: "cuda", "cpu", or "mps" (default: auto-detect)
- `--output_dir`: Where to save checkpoints
- `--num_workers`: DataLoader workers (default: 4)

### Evaluation Arguments

```bash
python evaluate.py --help
```

Key options:
- `--model`: "flat" or "hierarchical"
- `--checkpoint`: Path to model checkpoint
- `--num_samples`: Number of samples to generate (default: 1000)
- `--batch_size`: Generation batch size

## 🔬 Experimental Extensions

After completing the main project, try:

### 1. Vary Data Distribution
Modify `p_z1_given_z2` in `data/synthetic_dataset.py`:
```python
self.p_z1_given_z2 = {
    "animal": {"dog": 0.8, "cat": 0.2},  # Imbalanced
    "vehicle": {"car": 0.5, "truck": 0.5},
}
```

Does the flat model collapse to the majority class?

### 2. Add More Subtypes
Extend to 4 animals (dog, cat, bird, fish):
```python
self.z1_subtypes = {
    "animal": ["dog", "cat", "bird", "fish"],
    "vehicle": ["car", "truck"]
}
```

### 3. Increase Image Complexity
- Larger images (128x128)
- More detailed renderings
- Additional z0 parameters

### 4. Implement Model C
Three-level hierarchy with z2_hat latent variable (see original design docs)

## 📚 Learning Resources

### Understanding Diffusion Models
- [DDPM Paper](https://arxiv.org/abs/2006.11239) - Denoising Diffusion Probabilistic Models
- [Lilian Weng's Blog](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

### Mode Collapse in Generative Models
- Classic problem in GANs
- Also occurs in VAEs and autoregressive models
- Hierarchical structures as solution

### Hierarchical Generation
- Hierarchical VAEs
- Two-stage diffusion models (Stable Diffusion uses similar ideas)

## 🎯 Learning Objectives

By completing this project, you will:

✅ Understand mode collapse in conditional generation  
✅ Implement diffusion model forward passes  
✅ Design evaluation metrics for generative models  
✅ Compare flat vs. hierarchical architectures empirically  
✅ Gain intuition for why structure matters in deep learning

---

**Ready to implement?** Start with [ASSIGNMENT.md](ASSIGNMENT.md) for detailed task instructions!

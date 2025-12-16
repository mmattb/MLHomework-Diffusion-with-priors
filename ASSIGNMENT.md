# Hierarchical Diffusion - Your Implementation Tasks

This homework assignment gives you hands-on experience for learning hierarchical diffusion models. The framework is mostly built, but you need to implement key components to understand how everything works together.

## Progress Tracker

Track your progress as you complete each part:

- [ ] Part 1: ImageDenoiser forward() ⏱️ 2-3 hours
- [ ] Part 2: LatentPrior forward() ⏱️ 1-2 hours  
- [ ] Part 3: Mode coverage metric ⏱️ 2-3 hours
- [ ] Part 4: Conditional entropy ⏱️ 2-3 hours
- [ ] Part 5: KL divergence ⏱️ 2-3 hours
- [ ] Part 6: Train both models successfully ⏱️ 1 hour
- [ ] Part 7: Run evaluation and see the difference! ⏱️ 30 mins

---

## Mathematical Background

### Generative Modeling

A generative model learns a probability distribution $p_\theta(x)$ over data $x$ (e.g., images) that approximates the true data distribution $p_{\text{data}}(x)$. The goal is to generate new samples $x \sim p_\theta(x)$ that are indistinguishable from real data.

**Training objective:** Maximize the likelihood of observed data:
$$\max_\theta \mathbb{E}_{x \sim p_{\text{data}}} [\log p_\theta(x)]$$

### Conditional Generative Modeling

In conditional generation, we model $p_\theta(x | c)$ where $c$ is a conditioning variable (e.g., class label, text prompt). This allows controlled generation based on high-level attributes.

**Example:** "Generate an animal" where $c = \text{animal}$.

**The underspecification problem:** When the conditioning $c$ is coarse (high-level), the true conditional distribution $p_{\text{data}}(x|c)$ is **multimodal** - there are multiple distinct valid outputs. For instance:
- $c = \text{animal}$ should produce both dogs AND cats
- $c = \text{vehicle}$ should produce both cars AND trucks

### Diffusion Models

Diffusion models learn to generate data through iterative denoising. They are fantastically expressive, allowing them to learn extraordinarily complex distributions, enabling modern text2img pipelines such as Stable Diffusion. The process consists of two phases:

**Forward process (fixed):** Gradually add Gaussian noise over $T$ timesteps:
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

Starting from data $x_0 \sim p_{\text{data}}$, this produces increasingly noisy $x_1, x_2, ..., x_T$ until $x_T \approx \mathcal{N}(0, I)$.

**Reverse process (learned):** Train a neural network $\epsilon_\theta(x_t, t, c)$ to predict the noise added at each step. Sampling generates $x_0$ from pure noise $x_T$ by iteratively denoising:
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t, c)\right) + \sigma_t z$$

where $z \sim \mathcal{N}(0, I)$ and $\alpha_t = 1 - \beta_t$, $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$.

**Training objective:** Minimize the mean squared error between true noise and predicted noise:
$$\mathcal{L} = \mathbb{E}_{x_0, t, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t, c)\|^2\right]$$

**Inference anatomy:** At generation time, we:
1. Sample initial noise $x_T \sim \mathcal{N}(0, I)$ from an arbitrary point in state space
2. Iteratively denoise from $t=T$ to $t=0$ following the learned flow field $\epsilon_\theta(x_t, t, c)$
3. Each denoising step moves us closer to the data manifold

This remarkable property allows the diffusion prior to produce **on-manifold data starting from arbitrary (randomly chosen) points in state space** - the learned denoising network effectively captures the flow field that guides noise back to realistic samples.

### The Mode Collapse Problem

**Terminology note:** In this context, "mode" refers to a **semantic category or basin** (e.g., "dog" vs "cat"), not just a statistical peak. Each semantic mode itself contains a full distribution with variance.

**Why collapse occurs:** When training on multimodal conditional distributions where a single network must represent multiple semantic categories, the model often collapses to generating only one category (or a small number of them). This happens because the single network $\epsilon_\theta(x_t, t, c)$ receives conflicting training signals: when $c=\text{animal}$, should it denoise toward dogs or cats? The ambiguous conditioning creates an optimization challenge. Consider a simple example with L2 loss:

$$\mathcal{L}(c) = \mathbb{E}_{x \sim p(x|c)}\left[\|x - G_\theta(z, c)\|^2\right]$$

For $c = \text{animal}$ with equally likely dogs and cats, the optimal deterministic strategy is to generate an "average animal" - minimizing $\mathbb{E}[\|x_{\text{dog}} - \hat{x}\|^2] + \mathbb{E}[\|x_{\text{cat}} - \hat{x}\|^2]$. While diffusion models are stochastic and don't directly produce such averages, they face a similar challenge: the single network must learn conflicting denoising directions for the same conditioning value. This often leads to collapse to one semantic mode (e.g., only dogs) because:

1. **Local minima:** Once the model starts generating one mode, gradients reinforce that mode
2. **Missing gradient signal:** If the model never generates cats, it receives no training signal to improve cat generation
3. **Optimization bias:** SGD tends toward low-variance solutions that reduce average loss

**Measuring collapse:** Let $Z_1 = \{\text{dog}, \text{cat}\}$ be the set of subtypes for $c = \text{animal}$. Ideally:
$$p_\theta(z_1 | c) \approx p_{\text{data}}(z_1 | c) = 0.5 \text{ for each } z_1 \in Z_1$$

Mode collapse occurs when $p_\theta(\text{dog} | \text{animal}) \approx 1$ and $p_\theta(\text{cat} | \text{animal}) \approx 0$. This manifests as high KL divergence between the model's output distribution and the true conditional distribution, indicating that the model fails to cover all semantic modes: $\text{KL}(p_\text{data}(x|c) \| p_\theta(x|c))$ remains large.

### Hierarchical Solution

**Key insight:** In the flat (non-hierarchical) approach, the condition $c$ maps to a single learned vector field (denoising function). This single network must handle all semantic modes, creating conflicting optimization objectives. The hierarchical approach solves this by **factorizing** the problem:

1. A **prior** learns a distribution over latent codes: $z_1 \sim p(z_1|c)$, where different $z_1$ values represent different semantic modes (dog vs cat)
2. A **decoder** learns mode-specific generation: $x \sim p(x|z_1)$, where each $z_1$ value gets its own effective denoising trajectory

By sampling $z_1$ first, we resolve the ambiguity *before* generating the image. The decoder no longer needs to average over conflicting modes—it generates for one specific mode at a time. This allows the model's output distribution $p_\theta(x|c)$ to properly cover all semantic modes of the true conditional distribution $p_\text{data}(x|c)$.

Put in terms of probabilities, we are factorizing generation through explicit intermediate latent variables:
$$p_\theta(x | c) = \int p_\theta(x | z_1) p_\theta(z_1 | c) \, dz_1$$

Instead of directly modeling $p(x | c = \text{animal})$, we:

1. **Prior:** Sample subtype $z_1 \sim p_\theta(z_1 | c)$ (e.g., "dog" or "cat")
2. **Decoder:** Generate image $x \sim p_\theta(x | z_1)$

**Why this helps:**
- The prior $p_\theta(z_1 | c)$ is trained to match the distribution over subtypes using diffusion: $\mathcal{L}_{\text{prior}} = \mathbb{E}_{z_1, t, \epsilon}[\|\epsilon - \epsilon_{\text{prior}}(z_{1,t}, t, c)\|^2]$, which implicitly learns $p(z_1|c)$
- The decoder learns mode-specific generation for each $z_1$, avoiding conflicting gradients
- Inter-mode diversity comes from sampling the prior; intra-mode diversity comes from decoder stochasticity

**Hierarchical diffusion architecture:**
- **Prior network:** $\epsilon_{\text{prior}}(z_{1,t}, t, c)$ - diffusion over latent embeddings $z_1 \in \mathbb{R}^d$
- **Decoder network:** $\epsilon_{\text{dec}}(x_t, t, z_1)$ - diffusion over images conditioned on sampled $z_1$

**Sampling procedure:**
1. Sample $z_{1,T} \sim \mathcal{N}(0, I)$
2. Denoise to get $z_{1,0} = \text{DDPM}(z_{1,T}, \epsilon_{\text{prior}}, c)$
3. Sample $x_T \sim \mathcal{N}(0, I)$
4. Denoise to get $x_0 = \text{DDPM}(x_T, \epsilon_{\text{dec}}, z_{1,0})$

This two-stage process enables the model to represent multiple semantic modes through the prior, resolving the ambiguous conditioning that causes collapse in flat models.

**Note on semantic modes in this assignment:** The semantic modes (dog/cat/car/truck) in our synthetic dataset are **explicitly labeled** with ground-truth $z_1$ embeddings. During training:
- The decoder learns $p(x|z_1)$: "Given the $z_1$ embedding for 'dog', generate dog images"
- The prior learns $p(z_1|c)$: "Given $c=\text{animal}$, generate $z_1$ embeddings for either 'dog' or 'cat'"

In real-world applications, semantic modes can be defined in different ways:
- **Supervised**: Text prompts (Stable Diffusion), discrete class labels (ImageNet classifiers)
- **Unsupervised**: Learned latent codes (VAEs discover clusters in the data without explicit labels)
- **Hybrid**: Weakly supervised or contrastive learning approaches

The key insight remains the same: factorizing generation through an intermediate representation (whether labeled or learned) resolves the ambiguity in underspecified conditioning.

**Important: The prior operates on continuous embeddings, not discrete labels**

You might wonder: since we have discrete categories (dog/cat/car/truck), why not just sample from a categorical distribution instead of using diffusion? Great question!

The answer: **Each discrete category is represented by a continuous 2-dimensional embedding** (L2-normalized vector in $\mathbb{R}^2$). The prior learns to denoise from Gaussian noise to these specific embedding points.

**Why use continuous embeddings + diffusion?**
1. **Mirrors real systems**: Stable Diffusion uses continuous CLIP embeddings, DALL-E 2 uses prior diffusion over embeddings. We want this assignment to motivate learning, and hence our diffusion prior in this assignment will be a diffusion process, not merely an estimation and sampling of a multinomial distribution.
2. **Generalizes beyond discrete labels**: Works for unsupervised settings or fine-grained continuous concepts
4. **Enables interpolation**: Continuous space allows blending between modes

**How it works:**
- Training: Ground truth $z_1$ embeddings (continuous 2D vectors) are corrupted with Gaussian noise, prior predicts the noise
- Sampling: Start from $z_{1,T} \sim \mathcal{N}(0, I)$ in $\mathbb{R}^2$, iteratively denoise to get $z_{1,0}$ near one of the learned embeddings
- Diffusion: Standard DDPM in continuous space, same math as image diffusion

Think of the embeddings as **anchor points** in continuous 2D space. The prior learns: "Given $c=\text{animal}$, denoise toward either the dog anchor or cat anchor."

---

## Overview

You will implement:
1. Two neural network forward passes (image denoiser and latent prior)
2. Three evaluation metrics to measure mode collapse and diversity

The framework provides complete training loops, data generation, and scaffolding. Your job is to fill in the critical learning components.

---

## Part 1: Image Denoiser Network (★★★)

**File**: `models/image_denoiser.py`

**What to implement**: The `forward()` method of the `ImageDenoiser` class.

**Input shapes**:
- `x_t`: Noisy images at timestep t, shape `(batch_size, 3, 64, 64)`
- `t`: Diffusion timestep, shape `(batch_size,)` - integers in [0, num_timesteps-1]
- `condition`: Conditioning information, shape `(batch_size, condition_dim)`

**Expected output**:
- `noise_pred`: Predicted noise, shape `(batch_size, 3, 64, 64)`

**What you need to do**:
1. Use `self.time_embed(t)` to get time embeddings
2. Use `self.condition_embed(condition)` to get condition embeddings
3. Combine time and condition embeddings appropriately
4. Pass the noisy image and combined embeddings through the UNet backbone
5. Return the predicted noise

**Architecture provided**:
- `self.time_embed`: Sinusoidal time embedding module
- `self.condition_embed`: Linear projection for conditioning
- `self.unet`: A small UNet backbone (already implemented)

**Learning goal**: Understand how time and conditioning information are injected into a diffusion model.

**Status**: ⬜ Not implemented

---

## Part 2: Latent Prior Network (★★★)

**File**: `models/latent_prior.py`

**What to implement**: The `forward()` method of the `LatentPrior` class.

**Input shapes**:
- `z1_t`: Noisy latent embeddings at timestep t, shape `(batch_size, latent_dim)`
  - In our case, `latent_dim = 2`
- `t`: Diffusion timestep, shape `(batch_size,)` - integers in [0, num_timesteps-1]
- `condition`: Category conditioning (z2), shape `(batch_size, condition_dim)`

**Expected output**:
- `noise_pred`: Predicted noise in latent space, shape `(batch_size, latent_dim)`

**What you need to do**:
1. Use `self.time_embed(t)` to get time embeddings
2. Use `self.condition_embed(condition)` to get condition embeddings
3. Concatenate z1_t with the embedded time and condition
4. Pass through the MLP layers
5. Return the predicted noise

**Architecture provided**:
- `self.time_embed`: Sinusoidal time embedding module
- `self.condition_embed`: Linear projection for category conditioning
- `self.mlp`: Multi-layer perceptron (already implemented)

**Learning goal**: Understand how diffusion works in latent space and how priors are conditioned.

**Status**: ⬜ Not implemented

---

## Part 3: Mode Coverage Metric (★★☆)

**File**: `evaluation/metrics.py`

**What to implement**: The `compute_mode_coverage()` function.

**Goal**: Measure how many distinct z1 categories are recovered when sampling from p(image | z2).

**Inputs**:
- `samples`: Generated images, shape `(num_samples, 3, 64, 64)`
- `z2_condition`: The conditioning used (e.g., "animal" or "vehicle")
- `dataset`: The dataset object with ground truth information

**Expected output**:
- A dictionary with:
  - `'num_modes_found'`: Integer count of distinct z1 modes identified
  - `'total_possible_modes'`: Integer count of ground truth z1 modes for this z2
  - `'coverage_ratio'`: Float in [0, 1]

**What you need to do**:
1. For each generated sample, determine which ground truth z1 it most resembles
   - Hint: You can use a simple classifier or nearest neighbor matching
   - The dataset provides a method to get reference images for each z1
2. Count the number of unique z1 modes found across all samples
3. Compare to the total number of possible z1 modes for this z2
4. Return the coverage statistics

**Learning goal**: Understand how to quantify mode collapse in generative models.

**Status**: ⬜ Not implemented

---

## Part 4: Conditional Entropy (★★☆)

**File**: `evaluation/metrics.py`

**What to implement**: The `compute_conditional_entropy()` function.

**Goal**: Compute H(z1_hat | z2) to measure uncertainty in the latent space.

**Inputs**:
- `z1_samples`: Sampled latent embeddings from the prior, shape `(num_samples, latent_dim)`
- `z2_condition`: The conditioning used
- `dataset`: The dataset object with ground truth information

**Expected output**:
- A float representing the conditional entropy in nats (or bits if you prefer)

**What you need to do**:
1. Cluster or classify the z1_samples into discrete categories
   - You can use k-means, nearest neighbors to ground truth, or a learned classifier
2. Estimate the probability distribution p(z1_category | z2) from the samples
3. Compute entropy: H = -∑ p(z1|z2) log p(z1|z2)
4. Return the entropy value

**Learning goal**: Understand how to measure diversity in latent representations.

**Status**: ⬜ Not implemented

---

## Part 5: KL Divergence (★★★)

**File**: `evaluation/metrics.py`

**What to implement**: The `compute_kl_divergence()` function.

**Goal**: Compute KL(p_model(z1 | z2) || p_data(z1 | z2)) to measure distribution mismatch.

**Inputs**:
- `z1_samples`: Sampled latent embeddings from the model, shape `(num_samples, latent_dim)`
- `z2_condition`: The conditioning used
- `dataset`: The dataset object with ground truth distributions

**Expected output**:
- A float representing the KL divergence in nats

**What you need to do**:
1. Estimate p_model(z1 | z2) from the generated samples
   - Classify samples into z1 categories and compute empirical distribution
2. Get the ground truth distribution p_data(z1 | z2) from the dataset
   - The dataset knows the true conditional probabilities
3. Compute KL divergence: KL = ∑ p_data(z1|z2) log(p_data(z1|z2) / p_model(z1|z2))
4. Handle numerical stability (log(0), division by zero)
5. Return the KL divergence

**Learning goal**: Understand how to compare model distributions to ground truth.

**Status**: ⬜ Not implemented

---

## Testing Your Implementation

Once you've implemented the above components, you can:

1. **Train Model A (baseline)**:
   ```bash
   python train.py --model flat --epochs 50
   ```

2. **Train Model B (hierarchical)**:
   ```bash
   python train.py --model hierarchical --epochs 50
   ```

3. **Evaluate models**:
   ```bash
   python evaluate.py --model_path checkpoints/model_a.pt
   python evaluate.py --model_path checkpoints/model_b.pt
   ```

4. **Generate samples**:
   ```bash
   python sample.py --model_path checkpoints/model_b.pt --z2 animal --num_samples 16
   ```

## Expected Results

If your implementation is correct:

- **Model A (flat)** should show mode collapse: samples for "animal" will mostly be one type (e.g., all dogs or all cats)
- **Model B (hierarchical)** should show diversity: samples for "animal" will include both dogs and cats
- Mode coverage should be low (~0.5) for Model A, high (~1.0) for Model B
- Conditional entropy should be low for Model A, higher for Model B
- KL divergence should be high for Model A, low for Model B

---

## Tips and Hints

1. **Debugging networks**: Start with small batch sizes and check tensor shapes at each step
2. **Normalization**: Remember that z1 embeddings should be L2-normalized (length 1)
3. **Metrics**: Use visualization! Plot sample images and their assigned categories
4. **Numerical stability**: Add small epsilon values when computing logs and divisions
5. **Clustering**: sklearn.cluster.KMeans is your friend for discrete categorization

---

## Questions?

If something is unclear, check the docstrings in the code. Each function has detailed documentation about expected inputs, outputs, and behavior.

Good luck! 🚀

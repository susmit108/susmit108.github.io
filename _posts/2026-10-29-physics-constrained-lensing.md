---
layout: post
title: "Physics-Constrained Self-Supervised Representations for Gravitational-Lensing Images"
date: 2026-10-29
categories: gsoc ml4sci deeplense
---

**Google Summer of Code 2026: Final Blog**

**Organization:** ML4SCI (DeepLense)

**Contributor:** Susmit Neogi

**Mentors:** Karthik Gaur, Pranath Reddy, Michael Toomey

---

## Code Details

**Repository:** [DeepLense](https://github.com/ML4SCI/DeepLense) [TBC]
**PR:** https://github.com/ML4SCI/DeepLense/pull/247

---

## Introduction

Over the summer, we worked on a **fully self-supervised, physics-informed foundation-model prototype** for gravitational-lensing images. A single transformer encoder is pretrained on simulated lensing maps with **no labels**, using image reconstruction plus differentiable lensing-equation constraints on predicted potential ($$\psi$$) and convergence ($$\kappa$$) fields. The encoder is then **frozen**, and only small MLP adapters are trained for downstream tasks.

Headline results on Model-III dataset with model pretrained on Model-I dataset:

<div class="report-stats">
  <div class="report-stat">
    <strong>97.3%</strong>
    <span>Lens classification</span>
    <p>Frozen encoder + adapter, compared with 70.9% for the supervised ResNet-152.</p>
  </div>
  <div class="report-stat">
    <strong>0.63</strong>
    <span>Einstein-radius R²</span>
    <p>Regression from the same frozen representation, in standardized units.</p>
  </div>
  <div class="report-stat">
    <strong>97.85%</strong>
    <span>Physics-only ablation</span>
    <p>A surprising result that points towards capabilities of physics as a SSL signal.</p>
  </div>
</div>

---

## 1. Motivation

Gravitational lensing images carry information about both the foreground mass distribution (including dark-matter substructure) and the background source. Extracting that information, whether classifying substructure type or estimating quantities like the Einstein radius, usually requires labeled simulations, and labels are expensive or simply unavailable for observed systems.

The foundation-model recipe is appealing here: learn reusable features from large unlabeled collections of lensing maps, then adapt cheaply to many scientific tasks. My project asks a specific question:

> Can we make the pretraining signal *aware of lensing physics*, so that the representation reflects the underlying mass distribution rather than only pixel-level appearance?

---

## 2. Project Goals

1. Design a self-supervised pretraining objective for single-channel lensing images that needs **no labels**.
2. Build a **task-agnostic encoder** that can be frozen and reused through lightweight adapters.
3. Evaluate transfer on at least two different tasks: a discrete one (classification) and a continuous one (regression).

---

## 3. Data

We used two sets of simulations, all single-channel:

- **Pretraining:** DeepLense **Model I**, simulated with a simple Sérsic light profile in Lenstronomy.
- **Classification:** DeepLense **Model III**, with three near-balanced classes: *Axion*, *Cold Dark Matter (CDM)*, and *No Substructure*.
- **Regression:** a custom simulation set where we varied the **Einstein radius** and randomly sampled other parameters (ellipticity, source position, amplitude).

### Preprocessing

Images are bilinearly resized to **64 × 64** to keep the transformer token count manageable. Intensities are clamped below $$\epsilon = 10^{-6}$$, and a per-image log-contrast map is computed relative to the image maximum:

$$z = \left(\log \frac{I_{\max}}{\max(I, \epsilon)}\right)^2$$

Finite-difference filters followed by $$\tanh(\cdot)$$ produce an edge-like **preprocessed image**. In parallel, the raw image goes through a log transform and per-sample z-normalization to form a second reconstruction target that preserves the physical intensity scale.

Both targets are deterministic transformations of the input, so no annotation is needed.

---

## 4. Method

<figure class="report-figure">
  <img src="{{ '/assets/architecture.svg' | relative_url }}" alt="Architecture diagram showing a gravitational-lensing image passing through patch and positional embedding into a three-layer transformer encoder, then four decoders that predict convergence, potential, reconstructed raw image, and reconstructed preprocessed image." loading="lazy">
  <figcaption><strong>Figure 1.</strong> The shared transformer encoder maps a 64 × 64 lensing image to 256 tokens. Four decoder heads predict the potential, convergence, preprocessed image, and raw image during pretraining.</figcaption>
</figure>

### 4.1 Architecture

- **Patch embedding:** a convolution with kernel and stride 4 turns a 64 × 64 image into a 16 × 16 grid, i.e. **256 tokens** of dimension **768**.
- **Encoder:** learned positional embeddings, then **three transformer blocks** with 12 attention heads and an MLP expansion ratio of 4, followed by LayerNorm and a linear projector.
- **Decoders (pretraining only):** the token sequence is reshaped to a 16 × 16 feature map, upsampled with a transposed convolution plus ordinary convolutions, and sent to **four separate decoders** that predict:
  - the potential field $$\psi$$,
  - the convergence field $$\kappa$$,
  - the preprocessed image,
  - the log/z-normalized raw image.

Only the encoder is kept after pretraining; the decoders are discarded.

### 4.2 The physics-aware objective

The physical constraints tie the predicted fields together through the lensing relations:

- **Poisson consistency:** the Laplacian of the potential should match twice the convergence, $$\nabla^2\psi = 2\kappa$$. This is penalized with a squared L2 loss:

$$\mathcal{L}_{\text{Poisson}} = \left\lVert \nabla^2\psi - 2\kappa \right\rVert_2^2$$

- **Curl consistency:** a term discouraging non-conservative deflection derived from the potential.
- **Non-negative convergence:** a penalty on $$\max(-\kappa, 0)$$, since mass density cannot be negative.

The combined physics loss is

$$\mathcal{L}_{\text{phys}} = \lambda_P \mathcal{L}_{\text{Poisson}} + \lambda_C \mathcal{L}_{\text{curl}} + \lambda_+ \left\lVert \max(-\kappa, 0) \right\rVert_2^2$$

with $$(\lambda_P, \lambda_C, \lambda_+) = (10^{-2}, 10^{-3}, 10^{-2})$$.

The full self-supervised objective adds two **input-anchored reconstruction terms**:

$$\mathcal{L}_{\text{SSL}} = \lambda_I \lVert \hat{I} - I_{\text{prep}} \rVert_2^2 + \lambda_R \lVert \hat{R} - R \rVert_2^2 + \mathcal{L}_{\text{phys}}$$

with $$\lambda_I = 1$$ and $$\lambda_R = 0.1$$.

### 4.3 Frozen-encoder transfer

For each downstream task, the encoder output is **mean-pooled** over the 256 tokens to a single 768-dimensional vector and passed to an MLP adapter (three GELU hidden layers of width 512). Adapters are trained with AdamW for up to 30 epochs. The encoder is never fine-tuned.

### 4.4 Ablation: pure-physics pretraining

To probe what each ingredient contributes, we trained a second encoder with **only** $$\mathcal{L}_{\text{phys}}$$, removing both reconstruction terms, while keeping the input pipeline, transformer, decoder capacity, optimizer, and schedule identical.

**Pretraining setup (both models):** SGD, learning rate $$10^{-3}$$, weight decay $$10^{-4}$$, batch size 32, 10 epochs, best-validation-loss checkpoint.

---

## 5. Results

### Experiment 1: Lens-class classification

| Model | Accuracy (%) | F1 | Precision | Recall |
|---|---|---|---|---|
| ResNet-152 (supervised, end-to-end) | 70.90 | 0.70 | 0.70 | 0.71 |
| Physics Only (frozen + MLP) | 97.85 | 0.98 | 0.98 | 0.98 |
| SSL + Physics (frozen + MLP) | 97.30 | 0.97 | 0.97 | 0.97 |

Frozen self-supervised features with a small adapter beat a fully supervised ResNet-152 by roughly 26 percentage points on the same evaluation split.

### Experiment 2: Einstein-radius regression

Metrics are in standardized units ($$\theta_E$$ standardized by the training-set mean and standard deviation).

| Representation | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---|---|---|
| SSL + Physics | 0.37 | 0.47 | 0.63 |
| Physics Only | 0.28 | 0.37 | 0.67 |

The same frozen encoder, with a different one-scalar adapter, recovers a continuous physical scale. This suggests it retains morphology- and scale-sensitive information beyond what class separation requires.

---

## 6. Discussion

Frozen representations from this pretraining framework retain task-relevant information. A lightweight adapter on top of them is enough to get strong classification and non-trivial regression on simulated data.

The pure-physics ablation scoring as well as (or slightly better than) the full model is exciting, but it is also a point to ponder. The physics-only objective admits trivial solutions, such as a constant potential and zero convergence, that satisfy the consistency penalties without encoding the input at all. The strong downstream numbers could therefore come from the architecture, random-feature effects, or optimization dynamics rather than from physical consistency. Transformers with random weights can already yield surprisingly separable features.
An important thing to call out is that in this project, every data used is simulated. Transfer to real lensing images is a big next step.

---

## 7. Next Steps

These are the experiments I consider most important, in priority order:

- **Untrained-encoder control:** freeze a randomly initialized encoder and train the same adapters. If it scores close to 97%, the pretraining gain is mostly architectural.
- **Collapse diagnostics:** measure feature rank, variance across inputs, and decoder output variance for the physics-only model to check for representation collapse.
- **Physics term ablations:** turn on the Poisson, curl, and non-negativity terms one at a time. I also want to revisit the curl term's formulation, since the curl of a pure gradient field vanishes identically and the term may need to act on a separately predicted deflection field to be informative.
- **Scaling:** more encoder capacity, longer pretraining (10 epochs is short), and larger datasets. Probably one of the next logical steps to follow, given that we have evidence of success with physics based SSL and can be scaled to develop bigger and better models.
- **Real data:** test on observed or more realistic lensing images and under domain shift.

---

## 8. Reflections

The most valuable lesson of this project was to get a hands-on understanding of how physics concepts can heavily influence and improve machine learning pipelines. Since this was my first experience with physics informed machine learning, I found this field very exciting and interesting and thoroughly enjoyed learning the physics part of the concepts and translating them to code.

On the technical side I gained hands-on experience with transformer encoders for scientific images, designing differentiable losses (finite-difference Laplacians and constraint penalties), simulation with Lenstronomy, and structuring frozen-encoder evaluation pipelines.

---

## 9. Deliverables
The key deliverables as part of this project, can be found in the repository:

- Pretraining code for the SSL + physics and physics-only models
- Downstream adapters for classification and Einstein-radius regression
- Simulation scripts for the regression dataset
- Evaluation scripts and result tables

PR: 
Repository: 
---

## 10. Acknowledgments

Thank you to my mentors and the entire ML4SCI community for their guidance, feedback, and patience throughout the summer, and to Google Summer of Code for making this work possible.

---

## References

- Alexander, S., Gleyzer, S., McDonough, E., Toomey, M. W., & Usai, E. (2020). *Deep learning the morphology of dark matter substructure.* The Astrophysical Journal, 893(1), 15.
- Bartelmann, M., & Schneider, P. (2001). *Weak gravitational lensing.* Physics Reports, 340(4–5), 291–472.
- Birrer, S., et al. (2021). *lenstronomy II: A gravitational lensing software ecosystem.* Journal of Open Source Software, 6(62), 3283.

---
layout: post
title: "Audio Encoders: From VQ-VAE to Modern Neural Audio Compression"
date: 2026-07-20
tags: [audio, machine-learning, vq-vae, neural-compression, deep-learning]
math: true
description: "A deep dive into the evolution of neural audio encoders — from VQ-VAE to SoundStream, EnCodec, AudioLM, and beyond — with full mathematical formulations and interactive-style SVG architecture diagrams."
---

If you've ever wondered how modern AI can compress audio to a fraction of its original size while sounding pristine — or how text-to-music models like MusicGen actually work under the hood — you're in the right place.

Over the past decade, audio encoding has undergone a quiet revolution. We've gone from hand-engineered codecs like MP3 and Opus to **learned neural representations** that can compress speech to 1.5 kbps (that's ~100× smaller than CD quality) while sounding better than traditional codecs. The key enabler? A family of models built on the idea of **discrete latent spaces** — starting with **VQ-VAE** in 2017 and evolving into the powerful neural audio language models we have today.

This post traces the full evolution, from VQ-VAE's core ideas through each major breakthrough, with the mathematical formulations and architecture diagrams that make them tick. Whether you're a researcher new to the field, an engineer evaluating codecs, or just curious about how neural audio works, I hope this gives you a clear picture of where we've been and where we're headed.

**What you'll learn:**
- How VQ-VAE introduced discrete latents and the straight-through estimator
- Why hierarchical latents (VQ-VAE-2) and residual quantization (RVQ) matter
- How adversarial and perceptual losses changed the game
- The surprising connection between audio compression and language modeling
- What the next generation of audio encoders looks like

---

## Table of Contents

1. [VQ-VAE (2017) — The Foundation](#1-the-foundation-vq-vae-2017)
2. [VQ-VAE-2 (2019) — Hierarchical Latents](#2-vq-vae-2-2019-hierarchical-latents)
3. [SoundStream (2021) — End-to-End Neural Audio Codec](#3-soundstream-2021-end-to-end-neural-audio-codec)
4. [EnCodec (2022) — Meta's High-Fidelity Codec](#4-encodec-2022-metas-high-fidelity-codec)
5. [AudioLM (2022) — Language Modeling on Audio Tokens](#5-audiolm-2022-language-modeling-on-audio-tokens)
6. [MusicLM / MusicGen (2023) — Text-to-Music Generation](#6-musiclm--musicgen-2023-text-to-music-generation)
7. [Modern Developments (2024+)](#7-modern-developments-2024)
8. [Comparative Summary](#8-comparative-summary)
9. [Practical Implementation Tips](#9-practical-implementation-tips)
10. [Future Directions](#10-future-directions)

---

## 1. The Foundation: VQ-VAE (2017)

Back in 2017, most neural audio work used **continuous** latent representations. The problem? Continuous latents are hard to compress, hard to model autoregressively, and prone to "posterior collapse" in VAEs (where the decoder learns to ignore the latent entirely).

The **Vector Quantized Variational Autoencoder** (VQ-VAE) by van den Oord et al. at DeepMind proposed a radical alternative: what if we force the latent space to be **discrete**? Instead of encoding audio into a continuous vector $z$, we encode it into a sequence of **indices** from a learned codebook — like a vocabulary of audio tokens. This small change unlocked massive improvements in both compression and generation quality.

Here's the core idea at a glance:
### Architecture

<div class="my-8 flex justify-center">
  <svg width="100%" max-width="800px" height="220" viewBox="0 0 800 220" fill="none" xmlns="http://www.w3.org/2000/svg" class="w-full">
    <defs>
      <linearGradient id="encoderGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#4F46E5" />
        <stop offset="100%" stop-color="#7C3AED" />
      </linearGradient>
      <linearGradient id="codebookGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#EC4899" />
        <stop offset="100%" stop-color="#F43F5E" />
      </linearGradient>
      <linearGradient id="decoderGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#10B981" />
        <stop offset="100%" stop-color="#059669" />
      </linearGradient>
      <filter id="shadow" x="-5%" y="-5%" width="110%" height="110%" filterUnits="userSpaceOnUse">
        <feDropShadow dx="2" dy="4" stdDeviation="4" flood-opacity="0.1" />
      </filter>
    </defs>

    <rect width="800" height="220" rx="12" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="1"/>

    <text x="20" y="125" font-family="system-ui, -apple-system, sans-serif" font-size="14" fill="#64748B" font-weight="600">Waveform x</text>
    <path d="M 100 120 L 130 120" stroke="#94A3B8" stroke-width="2" />
    <polygon points="130,120 122,115 122,125" fill="#94A3B8" />

    <g filter="url(#shadow)">
      <rect x="130" y="65" width="140" height="90" rx="8" fill="url(#encoderGrad)" />
      <text x="200" y="105" font-family="system-ui, -apple-system, sans-serif" font-size="16" fill="white" font-weight="bold" text-anchor="middle">Encoder</text>
      <text x="200" y="125" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="#E9D5FF" text-anchor="middle">E_φ(x) → z_e</text>
    </g>

    <path d="M 270 110 L 350 110" stroke="#94A3B8" stroke-width="2" />
    <polygon points="350,110 342,105 342,115" fill="#94A3B8" />
    <text x="310" y="100" font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="#475569" font-weight="600" text-anchor="middle">z_e (continuous)</text>

    <g filter="url(#shadow)">
      <rect x="350" y="65" width="160" height="90" rx="8" fill="url(#codebookGrad)" />
      <text x="430" y="100" font-family="system-ui, -apple-system, sans-serif" font-size="15" fill="white" font-weight="bold" text-anchor="middle">Vector Quantization</text>
      <text x="430" y="120" font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="#FCE7F3" text-anchor="middle">z_q = argmin ‖z_e - e‖</text>
      <text x="430" y="138" font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="#FCE7F3" text-anchor="middle">Codebook {e₁…e_K}</text>
    </g>

    <path d="M 510 110 L 590 110" stroke="#94A3B8" stroke-width="2" />
    <polygon points="590,110 582,105 582,115" fill="#94A3B8" />
    <text x="550" y="100" font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="#475569" font-weight="600" text-anchor="middle">z_q (discrete)</text>

    <g filter="url(#shadow)">
      <rect x="590" y="65" width="140" height="90" rx="8" fill="url(#decoderGrad)" />
      <text x="660" y="105" font-family="system-ui, -apple-system, sans-serif" font-size="16" fill="white" font-weight="bold" text-anchor="middle">Decoder</text>
      <text x="660" y="125" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="#D1FAE5" text-anchor="middle">D_θ(z_q) → x̂</text>
    </g>

    <path d="M 730 110 L 770 110" stroke="#94A3B8" stroke-width="2" />
    <polygon points="770,110 762,105 762,115" fill="#94A3B8" />
    <text x="785" y="115" font-family="system-ui, -apple-system, sans-serif" font-size="14" fill="#64748B" font-weight="600">x̂</text>

    <!-- Straight-Through Estimator annotation -->
    <path d="M 660 155 L 200 155" stroke="#F59E0B" stroke-width="2" stroke-dasharray="6 3" />
    <polygon points="200,155 210,150 210,160" fill="#F59E0B" />
    <text x="430" y="185" font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="#D97706" font-weight="bold" text-anchor="middle">Straight-Through: ∂L/∂z_e ≈ ∂L/∂z_q  (gradients skip quantization)</text>
  </svg>
</div>

### Mathematical Formulation

**Encoder** maps input waveform $x \in \mathbb{R}^T$ to continuous latent $z_e \in \mathbb{R}^{H \times W \times D}$:

$$z_e = E_\phi(x)$$

**Vector Quantization** finds nearest codebook entry:

$$z_q = \text{Quantize}(z_e) = \arg\min_{e_k \in \mathcal{C}} \|z_e - e_k\|_2$$

where $\mathcal{C} = \{e_1, ..., e_K\}$ is the codebook of size $K$.

**Decoder** reconstructs waveform:

$$\hat{x} = D_\theta(z_q)$$

### Loss Function

The VQ-VAE loss combines three terms:

$$\mathcal{L} = \underbrace{\|x - \hat{x}\|_2^2}_{\text{Reconstruction}} + \underbrace{\|z_e - \text{sg}[z_q]\|_2^2}_{\text{Codebook loss}} + \underbrace{\beta\|\text{sg}[z_e] - z_q\|_2^2}_{\text{Commitment loss}}$$

where $\text{sg}[\cdot]$ denotes stop-gradient operator and $\beta$ controls commitment strength (typically $\beta=0.25$).

### Key Innovation: Straight-Through Estimator

Since quantization is non-differentiable, gradients are copied from decoder to encoder:

$$\frac{\partial \mathcal{L}}{\partial z_e} \approx \frac{\partial \mathcal{L}}{\partial z_q}$$

This "straight-through" estimator enables end-to-end training despite discrete latents.

Below is a visual comparison of the gradient flow with and without the straight-through estimator:

<figure style="text-align: center; margin: 2rem 0;">
<svg width="100%" max-width="800px" height="500" viewBox="0 0 800 500" fill="none" xmlns="http://www.w3.org/2000/svg" style="font-family: system-ui, -apple-system, sans-serif;" class="w-full">
  <rect width="800" height="500" rx="12" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="1"/>
  <text x="200" y="35" font-size="15" font-weight="bold" fill="#1E293B" text-anchor="middle">With Straight-Through (✓)</text>
  <text x="200" y="55" font-size="12" fill="#64748B" text-anchor="middle">z_q = z_e + (z_q − z_e).detach()</text>
  <rect x="100" y="100" width="140" height="70" rx="8" fill="#4F46E5" />
  <text x="170" y="135" font-size="14" font-weight="bold" fill="white" text-anchor="middle">Encoder</text>
  <text x="170" y="152" font-size="11" fill="#E9D5FF" text-anchor="middle">E_φ(x)</text>
  <rect x="100" y="200" width="140" height="70" rx="8" fill="#EC4899" />
  <text x="170" y="235" font-size="14" font-weight="bold" fill="white" text-anchor="middle">Quantization</text>
  <text x="170" y="252" font-size="10" fill="#FCE7F3" text-anchor="middle">argmin + Codebook</text>
  <rect x="100" y="300" width="140" height="70" rx="8" fill="#10B981" />
  <text x="170" y="335" font-size="14" font-weight="bold" fill="white" text-anchor="middle">Decoder</text>
  <text x="170" y="352" font-size="11" fill="#D1FAE5" text-anchor="middle">D_θ(z_q)</text>
  <rect x="100" y="400" width="140" height="40" rx="6" fill="#EF4444" />
  <text x="170" y="423" font-size="12" font-weight="bold" fill="white" text-anchor="middle">Loss: L(x, x̂)</text>
  <path d="M 170 170 L 170 190" stroke="#94A3B8" stroke-width="2" />
  <polygon points="170,190 165,182 175,182" fill="#94A3B8" />
  <path d="M 170 270 L 170 290" stroke="#94A3B8" stroke-width="2" />
  <polygon points="170,290 165,282 175,282" fill="#94A3B8" />
  <path d="M 170 370 L 170 390" stroke="#94A3B8" stroke-width="2" />
  <polygon points="170,390 165,382 175,382" fill="#94A3B8" />
  <path d="M 270 335 L 270 325 Q 280 260 270 235 L 270 135" stroke="#10B981" stroke-width="3" stroke-dasharray="8 4" fill="none" />
  <polygon points="270,135 263,147 277,147" fill="#10B981" />
  <path d="M 240 430 L 310 430 L 310 335" stroke="#EF4444" stroke-width="2" stroke-dasharray="4 4" fill="none" />
  <polygon points="310,335 305,343 315,343" fill="#EF4444" />
  <text x="310" y="195" font-size="11" fill="#10B981" font-weight="bold">Backward: gradient</text>
  <text x="310" y="210" font-size="11" fill="#10B981" font-weight="bold">bypasses quantization!</text>
  <text x="310" y="225" font-size="11" fill="#64748B">∂L/∂z_e ≈ ∂L/∂z_q</text>
  <line x1="400" y1="20" x2="400" y2="480" stroke="#CBD5E1" stroke-width="2" stroke-dasharray="4 4" />
  <text x="600" y="35" font-size="15" font-weight="bold" fill="#1E293B" text-anchor="middle">Without Straight-Through (✗)</text>
  <text x="600" y="55" font-size="12" fill="#64748B" text-anchor="middle">z_q = codebook(argmin(distances))</text>
  <rect x="500" y="100" width="140" height="70" rx="8" fill="#4F46E5" opacity="0.6" />
  <text x="570" y="135" font-size="14" font-weight="bold" fill="white" text-anchor="middle">Encoder</text>
  <text x="570" y="152" font-size="11" fill="#E9D5FF" text-anchor="middle">E_φ(x)</text>
  <rect x="500" y="200" width="140" height="70" rx="8" fill="#EC4899" opacity="0.6" />
  <text x="570" y="235" font-size="14" font-weight="bold" fill="white" text-anchor="middle">Quantization</text>
  <text x="570" y="252" font-size="10" fill="#FCE7F3" text-anchor="middle">argmin + Codebook</text>
  <line x1="510" y1="210" x2="630" y2="260" stroke="#EF4444" stroke-width="4" />
  <line x1="630" y1="210" x2="510" y2="260" stroke="#EF4444" stroke-width="4" />
  <rect x="500" y="300" width="140" height="70" rx="8" fill="#10B981" opacity="0.6" />
  <text x="570" y="335" font-size="14" font-weight="bold" fill="white" text-anchor="middle">Decoder</text>
  <text x="570" y="352" font-size="11" fill="#D1FAE5" text-anchor="middle">D_θ(z_q)</text>
  <rect x="500" y="400" width="140" height="40" rx="6" fill="#EF4444" opacity="0.6" />
  <text x="570" y="423" font-size="12" font-weight="bold" fill="white" text-anchor="middle">Loss: L(x, x̂)</text>
  <path d="M 570 170 L 570 190" stroke="#94A3B8" stroke-width="2" opacity="0.6" />
  <polygon points="570,190 565,182 575,182" fill="#94A3B8" opacity="0.6" />
  <path d="M 570 270 L 570 290" stroke="#94A3B8" stroke-width="2" opacity="0.6" />
  <polygon points="570,290 565,282 575,282" fill="#94A3B8" opacity="0.6" />
  <path d="M 570 370 L 570 390" stroke="#94A3B8" stroke-width="2" opacity="0.6" />
  <polygon points="570,390 565,382 575,382" fill="#94A3B8" opacity="0.6" />
  <path d="M 640 430 L 710 430 L 710 335" stroke="#EF4444" stroke-width="2" stroke-dasharray="4 4" opacity="0.5" />
  <polygon points="710,335 705,343 715,343" fill="#EF4444" opacity="0.5" />
  <path d="M 710 300 L 710 260" stroke="#EF4444" stroke-width="3" stroke-dasharray="8 4" opacity="0.7" />
  <rect x="695" y="260" width="30" height="20" rx="3" fill="#EF4444" />
  <text x="710" y="274" font-size="11" font-weight="bold" fill="white" text-anchor="middle">✗</text>
  <text x="720" y="195" font-size="11" fill="#EF4444" font-weight="bold">Backward: gradient</text>
  <text x="720" y="210" font-size="11" fill="#EF4444" font-weight="bold">BLOCKED at argmin!</text>
  <text x="720" y="225" font-size="11" fill="#64748B">argmin has no gradient</text>
  <text x="720" y="240" font-size="11" fill="#64748B">→ encoder never trains</text>
  <text x="400" y="485" font-size="11" fill="#64748B" text-anchor="middle">z_e + (z_q - z_e).detach()  →  forward uses z_q, backward sees z_e</text>
</svg>
<figcaption style="font-size: 0.85rem; color: #64748B; margin-top: 0.5rem;">Left: With straight-through estimator — gradients pass through $z_e$, bypassing the non-differentiable quantization. Right: Without straight-through — gradients are blocked at $\text{argmin}$, preventing the encoder from training.</figcaption>
</figure>

### Implementation

In code, the straight-through trick is elegantly simple. The quantized latent $z_q$ is treated as the output during the forward pass, but during backpropagation, the gradient flows **as if $z_e$ had been used directly**. This is achieved by:

```python
z_q = z_e + (z_q - z_e).detach()
```

When autograd computes gradients, `detach()` zeros out the gradient of the term inside. The forward pass uses the true quantized value $z_q$, but the gradient passes through $z_e$ as if it were the output. Here's a minimal VQ-VAE implementation:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super().__init__()
        self.beta = beta
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # z_e shape: (B, D, T) → (B, T, D) for distance computation
        z_e_flat = z_e.permute(0, 2, 1).reshape(-1, z_e.size(1))
        
        # Compute distances: ||z_e - e||² for all codebook entries
        distances = (
            z_e_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_e_flat @ self.codebook.weight.T
            + self.codebook.weight.pow(2).sum(1, keepdim=True).T
        )
        
        # Nearest codebook lookup (argmin over codebook indices)
        indices = distances.argmin(1)
        z_q_flat = self.codebook(indices)  # (B*T, D)
        
        # Reshape back to original
        z_q = z_q_flat.view_as(z_e)
        
        # ─── Straight-Through Estimator ───
        # Forward:  uses z_q (actual quantized vectors)
        # Backward: gradient flows through z_e (bypasses quantization)
        z_q_st = z_e + (z_q - z_e).detach()
        
        # VQ-VAE loss components
        recon_loss = F.mse_loss(self.decoder(z_q_st), x)
        codebook_loss = F.mse_loss(z_q.detach(), z_e)
        commitment_loss = F.mse_loss(z_q, z_e.detach())
        
        # Total loss: reconstruction + codebook + β * commitment
        loss = recon_loss + codebook_loss + self.beta * commitment_loss
        
        return z_q_st, indices, loss
```

The key line is `z_e + (z_q - z_e).detach()`:
- **Forward**: computes $z_q$ (the quantized vector) — `z_e` cancels out
- **Backward**: `detach()` stops gradients through $(z_q - z_e)$, so only $z_e$ receives gradients

This means the decoder receives discrete $z_q$ tokens, but the encoder is updated as if it produced $z_q$ continuously — the best of both worlds.

### Why This Matters

VQ-VAE's discrete latents were a breakthrough, but they had a limitation: the single codebook operates at one granularity. Low-level details (like phonemes) and high-level structure (like prosody) get conflated in the same latent space. The next iteration tackled this head-on.

---

## 2. VQ-VAE-2 (2019): Hierarchical Latents

**VQ-VAE-2** introduced multi-scale hierarchical quantization, capturing both coarse and fine-grained audio structure.

### Architecture

<div class="my-8 flex justify-center">
  <svg width="100%" max-width="800px" height="350" viewBox="0 0 800 350" fill="none" xmlns="http://www.w3.org/2000/svg" class="w-full">
    <defs>
      <linearGradient id="encoderGrad2" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#4F46E5" />
        <stop offset="100%" stop-color="#7C3AED" />
      </linearGradient>
      <linearGradient id="codebookGrad2" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#EC4899" />
        <stop offset="100%" stop-color="#F43F5E" />
      </linearGradient>
      <linearGradient id="decoderGrad2" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#10B981" />
        <stop offset="100%" stop-color="#059669" />
      </linearGradient>
      <filter id="shadow2" x="-5%" y="-5%" width="110%" height="110%" filterUnits="userSpaceOnUse">
        <feDropShadow dx="2" dy="4" stdDeviation="4" flood-opacity="0.1" />
      </filter>
    </defs>

    <!-- Outer card -->
    <rect width="800" height="350" rx="12" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="1"/>

    <!-- Column Headers -->
    <text x="220" y="35" font-family="system-ui, -apple-system, sans-serif" font-size="16" fill="#1E293B" font-weight="bold" text-anchor="middle">Level 1: Coarse (Top)</text>
    <text x="580" y="35" font-family="system-ui, -apple-system, sans-serif" font-size="16" fill="#1E293B" font-weight="bold" text-anchor="middle">Level 2: Fine (Bottom)</text>

    <!-- LEVEL 1 (COARSE) COLUMN -->
    <!-- Encoder 1 -->
    <g filter="url(#shadow2)">
      <rect x="140" y="55" width="160" height="70" rx="8" fill="url(#encoderGrad2)" />
      <text x="220" y="87" font-family="system-ui, -apple-system, sans-serif" font-size="14" fill="white" font-weight="bold" text-anchor="middle">Encoder 1</text>
      <text x="220" y="105" font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="#E9D5FF" text-anchor="middle">E₁(x) → z₁</text>
    </g>

    <!-- Codebook 1 -->
    <g filter="url(#shadow2)">
      <rect x="140" y="155" width="160" height="70" rx="8" fill="url(#codebookGrad2)" />
      <text x="220" y="187" font-family="system-ui, -apple-system, sans-serif" font-size="14" fill="white" font-weight="bold" text-anchor="middle">Codebook C₁</text>
      <text x="220" y="205" font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="#FCE7F3" text-anchor="middle">Discrete Latents</text>
    </g>

    <!-- Decoder 1 -->
    <g filter="url(#shadow2)">
      <rect x="140" y="255" width="160" height="70" rx="8" fill="url(#decoderGrad2)" />
      <text x="220" y="287" font-family="system-ui, -apple-system, sans-serif" font-size="14" fill="white" font-weight="bold" text-anchor="middle">Decoder 1</text>
      <text x="220" y="305" font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="#D1FAE5" text-anchor="middle">D₁(z₁, z₂)</text>
    </g>

    <!-- LEVEL 2 (FINE) COLUMN -->
    <!-- Encoder 2 -->
    <g filter="url(#shadow2)">
      <rect x="500" y="55" width="160" height="70" rx="8" fill="url(#encoderGrad2)" />
      <text x="580" y="87" font-family="system-ui, -apple-system, sans-serif" font-size="14" fill="white" font-weight="bold" text-anchor="middle">Encoder 2</text>
      <text x="580" y="105" font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="#E9D5FF" text-anchor="middle">E₂(x, z₁)</text>
    </g>

    <!-- Codebook 2 -->
    <g filter="url(#shadow2)">
      <rect x="500" y="155" width="160" height="70" rx="8" fill="url(#codebookGrad2)" />
      <text x="580" y="187" font-family="system-ui, -apple-system, sans-serif" font-size="14" fill="white" font-weight="bold" text-anchor="middle">Codebook C₂</text>
      <text x="580" y="205" font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="#FCE7F3" text-anchor="middle">Discrete Latents</text>
    </g>

    <!-- Decoder 2 -->
    <g filter="url(#shadow2)">
      <rect x="500" y="255" width="160" height="70" rx="8" fill="url(#decoderGrad2)" />
      <text x="580" y="287" font-family="system-ui, -apple-system, sans-serif" font-size="14" fill="white" font-weight="bold" text-anchor="middle">Decoder 2</text>
      <text x="580" y="305" font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="#D1FAE5" text-anchor="middle">D₂(z₁)</text>
    </g>

    <!-- CONNECTIONS & ARROWS -->
    <!-- E1 -> E2 -->
    <path d="M 300 90 L 500 90" stroke="#94A3B8" stroke-width="2" />
    <polygon points="500,90 492,85 492,95" fill="#94A3B8" />
    <text x="400" y="80" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="#475569" font-weight="600" text-anchor="middle">z₁ (coarse features)</text>

    <!-- E1 -> C1 -->
    <path d="M 220 125 L 220 155" stroke="#94A3B8" stroke-width="2" />
    <polygon points="220,155 215,147 225,147" fill="#94A3B8" />

    <!-- C1 -> D1 -->
    <path d="M 220 225 L 220 255" stroke="#94A3B8" stroke-width="2" />
    <polygon points="220,255 215,247 225,247" fill="#94A3B8" />

    <!-- E2 -> C2 -->
    <path d="M 580 125 L 580 155" stroke="#94A3B8" stroke-width="2" />
    <polygon points="580,155 575,147 585,147" fill="#94A3B8" />

    <!-- C2 -> D2 -->
    <path d="M 580 225 L 580 255" stroke="#94A3B8" stroke-width="2" />
    <polygon points="580,255 575,247 585,247" fill="#94A3B8" />

    <!-- D2 -> D1 -->
    <path d="M 500 290 L 300 290" stroke="#94A3B8" stroke-width="2" stroke-dasharray="4 4" />
    <polygon points="300,290 308,295 308,285" fill="#94A3B8" />
    <text x="400" y="280" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="#475569" font-weight="600" text-anchor="middle">z₂ (fine details)</text>
  </svg>
</div>

### Hierarchical Loss

$$\mathcal{L} = \sum_{i=1}^2 \left[ \mathcal{L}_{\text{recon}}^{(i)} + \mathcal{L}_{\text{codebook}}^{(i)} + \beta \mathcal{L}_{\text{commit}}^{(i)} \right]$$

### Autoregressive Prior

A PixelCNN-style autoregressive model learns $p(z_1)$ and $p(z_2 | z_1)$ for generation:

$$p(z) = \prod_i p(z_i | z_{<i})$$

This enables high-quality audio *generation* (not just compression), producing coherent long-form audio.

### Why This Matters

Hierarchical latents gave us **separation of concerns**: one level captures structure, another captures texture. This principle — multi-scale representation — would become a recurring theme in later models.

---

## 3. SoundStream (2021): End-to-End Neural Audio Codec

**SoundStream** (Zeghidour et al., Google) was the first *production-ready* neural audio codec, achieving parity with Opus at low bitrates.

### Architecture Overview

<div class="my-8 flex justify-center">
  <svg width="100%" max-width="800px" height="260" viewBox="0 0 800 260" fill="none" xmlns="http://www.w3.org/2000/svg" class="w-full">
    <defs>
      <linearGradient id="encoderGrad3" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#3B82F6" />
        <stop offset="100%" stop-color="#1D4ED8" />
      </linearGradient>
      <linearGradient id="rvqGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#F59E0B" />
        <stop offset="100%" stop-color="#EF4444" />
      </linearGradient>
      <linearGradient id="decoderGrad3" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#10B981" />
        <stop offset="100%" stop-color="#047857" />
      </linearGradient>
      <filter id="shadow3" x="-5%" y="-5%" width="110%" height="110%" filterUnits="userSpaceOnUse">
        <feDropShadow dx="2" dy="4" stdDeviation="4" flood-opacity="0.1" />
      </filter>
    </defs>

    <rect width="800" height="260" rx="12" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="1"/>

    <text x="30" y="135" font-family="system-ui, -apple-system, sans-serif" font-size="14" fill="#64748B" font-weight="600">Waveform x</text>
    <path d="M 110 130 L 140 130" stroke="#94A3B8" stroke-width="2" />
    <polygon points="140,130 132,125 132,135" fill="#94A3B8" />

    <!-- Encoder -->
    <g filter="url(#shadow3)">
      <rect x="140" y="55" width="160" height="150" rx="8" fill="url(#encoderGrad3)" />
      <text x="220" y="90" font-family="system-ui, -apple-system, sans-serif" font-size="16" fill="white" font-weight="bold" text-anchor="middle">Encoder</text>
      <text x="220" y="115" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="#DBEAFE" text-anchor="middle">1D Conv (stride 2)</text>
      <text x="220" y="135" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="#DBEAFE" text-anchor="middle">Residual Blocks</text>
      <text x="220" y="155" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="#DBEAFE" text-anchor="middle">Dilated Convolutions</text>
      <rect x="152" y="170" width="136" height="24" rx="4" fill="#1E40AF" />
      <text x="220" y="186" font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="white" font-weight="bold" text-anchor="middle">Output: z_e</text>
    </g>

    <path d="M 300 130 L 360 130" stroke="#94A3B8" stroke-width="2" />
    <polygon points="360,130 352,125 352,135" fill="#94A3B8" />

    <!-- RVQ with residual arrows -->
    <g filter="url(#shadow3)">
      <rect x="360" y="55" width="180" height="150" rx="8" fill="url(#rvqGrad)" />
      <text x="450" y="80" font-family="system-ui, -apple-system, sans-serif" font-size="15" fill="white" font-weight="bold" text-anchor="middle">Residual VQ (RVQ)</text>

      <!-- Level 1 -->
      <rect x="375" y="92" width="150" height="22" rx="4" fill="#FDE68A" />
      <text x="450" y="107" font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="#92400E" font-weight="bold" text-anchor="middle">Level 1: Codebook C₁</text>
      <!-- Residual arrow -->
      <path d="M 450 114 L 450 120" stroke="#FEF3C7" stroke-width="1.5" />
      <polygon points="450,120 446,115 454,115" fill="#FEF3C7" />

      <!-- Level 2 -->
      <rect x="375" y="120" width="150" height="22" rx="4" fill="#FDE68A" />
      <text x="450" y="135" font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="#92400E" font-weight="bold" text-anchor="middle">Level 2: Codebook C₂</text>
      <path d="M 450 142 L 450 148" stroke="#FEF3C7" stroke-width="1.5" />
      <polygon points="450,148 446,143 454,143" fill="#FEF3C7" />

      <text x="450" y="164" font-family="system-ui, -apple-system, sans-serif" font-size="14" fill="white" font-weight="bold" text-anchor="middle">↓ ... ↓</text>

      <!-- Level L -->
      <rect x="375" y="170" width="150" height="22" rx="4" fill="#FDE68A" />
      <text x="450" y="185" font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="#92400E" font-weight="bold" text-anchor="middle">Level L: Codebook C_L</text>
    </g>

    <path d="M 540 130 L 610 130" stroke="#94A3B8" stroke-width="2" />
    <polygon points="610,130 602,125 602,135" fill="#94A3B8" />

    <!-- Decoder -->
    <g filter="url(#shadow3)">
      <rect x="610" y="55" width="160" height="150" rx="8" fill="url(#decoderGrad3)" />
      <text x="690" y="90" font-family="system-ui, -apple-system, sans-serif" font-size="16" fill="white" font-weight="bold" text-anchor="middle">Decoder</text>
      <text x="690" y="115" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="#D1FAE5" text-anchor="middle">Transposed Conv</text>
      <text x="690" y="135" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="#D1FAE5" text-anchor="middle">Residual Blocks</text>
      <text x="690" y="155" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="#D1FAE5" text-anchor="middle">1D Conv</text>
      <rect x="622" y="170" width="136" height="24" rx="4" fill="#047857" />
      <text x="690" y="186" font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="white" font-weight="bold" text-anchor="middle">Output: x̂</text>
    </g>
  </svg>
</div>

### Residual Vector Quantization (RVQ)

Instead of a single codebook, RVQ uses $L$ codebooks sequentially. Each level quantizes the **residual error** from the previous level:

$$r_1 = z_e, \quad q_1 = Q(r_1) \quad \text{(quantize original latent)}$$
$$r_2 = r_1 - q_1, \quad q_2 = Q(r_2) \quad \text{(quantize residual)}$$
$$\vdots$$
$$z_q = \sum_{l=1}^L q_l \quad \text{(sum all quantized components)}$$

where $Q(\cdot)$ is the nearest-neighbor lookup into a codebook. The key insight: each codebook level adds refinement. Truncating at level $L' < L$ gives a **lower bitrate** with graceful degradation.

### Adversarial Training

SoundStream adds a **multi-scale discriminator** for perceptual quality:

$$\mathcal{L}_{\text{adv}} = \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1 - D(\hat{x}))]$$

**Feature matching loss** stabilizes training:

$$\mathcal{L}_{\text{FM}} = \sum_{k} \frac{1}{N_k} \|D_k(x) - D_k(\hat{x})\|_1$$

### STFT Spectral Loss

Multi-resolution STFT loss captures spectral fidelity:

$$\mathcal{L}_{\text{STFT}} = \sum_{m \in \mathcal{M}} \left\| \text{STFT}_m(x) - \text{STFT}_m(\hat{x}) \right\|_1$$

where $\mathcal{M}$ contains multiple window sizes (e.g., 2048, 1024, 512, 256, 128).

### Why This Matters

SoundStream was the first neural codec that could genuinely compete with traditional codecs in practice. The combination of **RVQ for variable bitrate** and **adversarial + multi-spectral losses** set the template that all subsequent neural audio codecs would follow.

---

## 4. EnCodec (2022): Meta's High-Fidelity Codec

**EnCodec** (Défossez et al., Meta) improved upon SoundStream with better compression efficiency and a **transformer-based language model** for generation.

### Architecture Enhancements

<div class="my-8 flex justify-center">
  <svg width="100%" max-width="800px" height="280" viewBox="0 0 800 280" fill="none" xmlns="http://www.w3.org/2000/svg" class="w-full">
    <defs>
      <linearGradient id="seanetGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#4F46E5" />
        <stop offset="100%" stop-color="#312E81" />
      </linearGradient>
      <linearGradient id="rvqEncodecGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#F43F5E" />
        <stop offset="100%" stop-color="#BE123C" />
      </linearGradient>
      <linearGradient id="seanetDecGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#10B981" />
        <stop offset="100%" stop-color="#064E3B" />
      </linearGradient>
      <filter id="shadowEncodec" x="-5%" y="-5%" width="110%" height="110%" filterUnits="userSpaceOnUse">
        <feDropShadow dx="2" dy="4" stdDeviation="4" flood-opacity="0.1" />
      </filter>
    </defs>

    <rect width="800" height="280" rx="12" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="1"/>

    <!-- Left Input Waveform -->
    <text x="30" y="145" font-family="system-ui, -apple-system, sans-serif" font-size="14" fill="#64748B" font-weight="600">Waveform</text>
    <path d="M 100 140 L 130 140" stroke="#94A3B8" stroke-width="2" />
    <polygon points="130,140 122,135 122,145" fill="#94A3B8" />

    <!-- SEANet Encoder Box -->
    <g filter="url(#shadowEncodec)">
      <rect x="130" y="45" width="220" height="190" rx="10" fill="url(#seanetGrad)" />
      <text x="240" y="75" font-family="system-ui, -apple-system, sans-serif" font-size="16" fill="white" font-weight="bold" text-anchor="middle">SEANet Encoder</text>
      
      <!-- Inner blocks -->
      <rect x="145" y="95" width="190" height="30" rx="4" fill="#6366F1" />
      <text x="240" y="114" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="white" font-weight="600" text-anchor="middle">1D Conv (strides 2,4,5,8)</text>

      <rect x="145" y="135" width="190" height="30" rx="4" fill="#4338CA" />
      <text x="240" y="154" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="white" font-weight="600" text-anchor="middle">Squeeze-and-Excitation (SE)</text>

      <rect x="145" y="175" width="190" height="30" rx="4" fill="#312E81" />
      <text x="240" y="194" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="white" font-weight="600" text-anchor="middle">Residual Blocks</text>
    </g>

    <path d="M 350 140 L 390 140" stroke="#94A3B8" stroke-width="2" />
    <polygon points="390,140 382,135 382,145" fill="#94A3B8" />

    <!-- RVQ Box -->
    <g filter="url(#shadowEncodec)">
      <rect x="390" y="45" width="200" height="190" rx="10" fill="url(#rvqEncodecGrad)" />
      <text x="490" y="75" font-family="system-ui, -apple-system, sans-serif" font-size="16" fill="white" font-weight="bold" text-anchor="middle">Multi-Level RVQ</text>
      
      <!-- Levels representation -->
      <g transform="translate(410, 95)">
        <rect x="0" y="0" width="160" height="20" rx="3" fill="#FDA4AF" opacity="0.9" />
        <text x="80" y="14" font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="#881337" font-weight="bold" text-anchor="middle">Level 1 (K=1024)</text>
        
        <rect x="0" y="25" width="160" height="20" rx="3" fill="#FB7185" opacity="0.9" />
        <text x="80" y="39" font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="#881337" font-weight="bold" text-anchor="middle">Level 2 (K=1024)</text>
        
        <text x="80" y="66" font-family="system-ui, -apple-system, sans-serif" font-size="14" fill="white" font-weight="bold" text-anchor="middle">•••</text>

        <rect x="0" y="80" width="160" height="20" rx="3" fill="#F43F5E" opacity="0.9" />
        <text x="80" y="94" font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="white" font-weight="bold" text-anchor="middle">Level 8 (K=1024)</text>
      </g>
      
      <text x="490" y="215" font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="#FFE4E6" text-anchor="middle">Bitrate: 1.5 - 24 kbps</text>
    </g>

    <path d="M 590 140 L 630 140" stroke="#94A3B8" stroke-width="2" />
    <polygon points="630,140 622,135 622,145" fill="#94A3B8" />

    <!-- SEANet Decoder Box -->
    <g filter="url(#shadowEncodec)">
      <rect x="630" y="45" width="140" height="190" rx="10" fill="url(#seanetDecGrad)" />
      <text x="700" y="75" font-family="system-ui, -apple-system, sans-serif" font-size="16" fill="white" font-weight="bold" text-anchor="middle">SEANet Decoder</text>
      
      <text x="700" y="115" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="#A7F3D0" text-anchor="middle">Transposed Conv</text>
      <text x="700" y="135" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="#A7F3D0" text-anchor="middle">Symmetric Upsampling</text>
      <text x="700" y="155" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="#A7F3D0" text-anchor="middle">SE Attention Blocks</text>
      
      <rect x="642" y="180" width="116" height="35" rx="4" fill="#047857" />
      <text x="700" y="202" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="white" font-weight="600" text-anchor="middle">Reconstructed x̂</text>
    </g>
  </svg>
</div>

### Key Improvements

1. **SEANet blocks**: Squeeze-and-excitation for channel-wise attention
2. **Larger receptive field**: Dilated convolutions capture long-range dependencies
3. **Lower bitrate**: 1.5–24 kbps vs SoundStream's 3–12 kbps
4. **LM for generation**: Transformer models $p(z_1, ..., z_L)$ for AudioGen

### Loss Function

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda_{\text{adv}} \mathcal{L}_{\text{adv}} + \lambda_{\text{feat}} \mathcal{L}_{\text{feat}} + \lambda_{\text{STFT}} \mathcal{L}_{\text{STFT}}$$

### Why This Matters

EnCodec pushed the compression-quality frontier significantly, matching Opus at 64 kbps with just 24 kbps. More importantly, its **discrete token output** opened the door to a radical idea: treat audio tokens like language tokens, and use a language model to generate them.

---

## 5. AudioLM (2022): Language Modeling on Audio Tokens

**AudioLM** (Borsos et al., Google) treats audio as a *language modeling* problem, generating long-form coherent audio without text conditioning.

### Three-Stage Pipeline

<div class="my-8 flex justify-center">
  <svg width="100%" max-width="800px" height="380" viewBox="0 0 800 380" fill="none" xmlns="http://www.w3.org/2000/svg" class="w-full">
    <defs>
      <linearGradient id="stage1Grad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#4F46E5" />
        <stop offset="100%" stop-color="#3B82F6" />
      </linearGradient>
      <linearGradient id="stage2Grad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#EA580C" />
        <stop offset="100%" stop-color="#EC4899" />
      </linearGradient>
      <linearGradient id="stage3Grad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#059669" />
        <stop offset="100%" stop-color="#10B981" />
      </linearGradient>
      <filter id="shadowAudioLM" x="-2%" y="-2%" width="104%" height="104%" filterUnits="userSpaceOnUse">
        <feDropShadow dx="1" dy="2" stdDeviation="3" flood-opacity="0.08" />
      </filter>
    </defs>

    <rect width="800" height="380" rx="12" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="1"/>

    <!-- STAGE 1: SEMANTIC MODELING -->
    <g filter="url(#shadowAudioLM)">
      <rect x="20" y="20" width="760" height="95" rx="8" fill="#F1F5F9" stroke="#E2E8F0" stroke-width="1" />
      <text x="35" y="45" font-family="system-ui, -apple-system, sans-serif" font-size="13" fill="#475569" font-weight="bold">STAGE 1: Semantic Modeling (Coarse Tokens)</text>
      
      <!-- w2v-BERT -->
      <rect x="150" y="55" width="140" height="45" rx="6" fill="url(#stage1Grad)" />
      <text x="220" y="75" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="white" font-weight="bold" text-anchor="middle">w2v-BERT (Semantic)</text>
      <text x="220" y="90" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#DBEAFE" text-anchor="middle">SSL Model</text>

      <!-- Arrow -->
      <path d="M 290 77 L 340 77" stroke="#94A3B8" stroke-width="2" />
      <polygon points="340,77 332,72 332,82" fill="#94A3B8" />

      <!-- Quantization -->
      <rect x="340" y="55" width="140" height="45" rx="6" fill="#1E293B" />
      <text x="410" y="75" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="white" font-weight="bold" text-anchor="middle">Quantization</text>
      <text x="410" y="90" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#94A3B8" text-anchor="middle">K = 1024</text>

      <!-- Arrow -->
      <path d="M 480 77 L 530 77" stroke="#94A3B8" stroke-width="2" />
      <polygon points="530,77 522,72 522,82" fill="#94A3B8" />

      <!-- Transformer Decoder -->
      <rect x="530" y="55" width="180" height="45" rx="6" fill="url(#stage1Grad)" />
      <text x="620" y="75" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="white" font-weight="bold" text-anchor="middle">Semantic Transformer</text>
      <text x="620" y="90" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#DBEAFE" text-anchor="middle">Learns p(s_t | s_<t)</text>
    </g>

    <!-- STAGE 2: COARSE ACOUSTIC MODELING -->
    <g filter="url(#shadowAudioLM)">
      <rect x="20" y="135" width="760" height="100" rx="8" fill="#F1F5F9" stroke="#E2E8F0" stroke-width="1" />
      <text x="35" y="160" font-family="system-ui, -apple-system, sans-serif" font-size="13" fill="#475569" font-weight="bold">STAGE 2: Coarse Acoustic Modeling (Structure)</text>
      
      <!-- EnCodec -->
      <rect x="150" y="170" width="140" height="50" rx="6" fill="url(#stage2Grad)" />
      <text x="220" y="190" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="white" font-weight="bold" text-anchor="middle">EnCodec Encoder</text>
      <text x="220" y="205" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#FFE4E6" text-anchor="middle">Waveform → Latent</text>

      <!-- Arrow -->
      <path d="M 290 195 L 340 195" stroke="#94A3B8" stroke-width="2" />
      <polygon points="340,195 332,190 332,200" fill="#94A3B8" />

      <!-- RVQ Level 1-2 -->
      <rect x="340" y="170" width="140" height="50" rx="6" fill="#1E293B" />
      <text x="410" y="190" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="white" font-weight="bold" text-anchor="middle">RVQ (Level 1-2)</text>
      <text x="410" y="205" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#94A3B8" text-anchor="middle">First 2 Codebooks</text>

      <!-- Arrow -->
      <path d="M 480 195 L 530 195" stroke="#94A3B8" stroke-width="2" />
      <polygon points="530,195 522,190 522,200" fill="#94A3B8" />

      <!-- Coarse Acoustic Transformer -->
      <rect x="530" y="170" width="180" height="50" rx="6" fill="url(#stage2Grad)" />
      <text x="620" y="190" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="white" font-weight="bold" text-anchor="middle">Coarse Transformer</text>
      <text x="620" y="205" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#FFE4E6" text-anchor="middle">p(c_t | c_<t, s_≤t)</text>
    </g>

    <!-- STAGE 3: FINE ACOUSTIC MODELING -->
    <g filter="url(#shadowAudioLM)">
      <rect x="20" y="255" width="760" height="100" rx="8" fill="#F1F5F9" stroke="#E2E8F0" stroke-width="1" />
      <text x="35" y="280" font-family="system-ui, -apple-system, sans-serif" font-size="13" fill="#475569" font-weight="bold">STAGE 3: Fine Acoustic Modeling (Fidelity)</text>
      
      <!-- EnCodec -->
      <rect x="150" y="290" width="140" height="50" rx="6" fill="url(#stage3Grad)" />
      <text x="220" y="310" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="white" font-weight="bold" text-anchor="middle">EnCodec Decoder</text>
      <text x="220" y="325" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#D1FAE5" text-anchor="middle">Reconstructs wave</text>

      <!-- Arrow -->
      <path d="M 290 315 L 340 315" stroke="#94A3B8" stroke-width="2" />
      <polygon points="340,315 332,310 332,320" fill="#94A3B8" />

      <!-- RVQ Level 3-L -->
      <rect x="340" y="290" width="140" height="50" rx="6" fill="#1E293B" />
      <text x="410" y="310" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="white" font-weight="bold" text-anchor="middle">RVQ (Level 3-L)</text>
      <text x="410" y="325" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#94A3B8" text-anchor="middle">Remaining levels</text>

      <!-- Arrow -->
      <path d="M 480 315 L 530 315" stroke="#94A3B8" stroke-width="2" />
      <polygon points="530,315 522,310 522,320" fill="#94A3B8" />

      <!-- Fine Acoustic Transformer -->
      <rect x="530" y="290" width="180" height="50" rx="6" fill="url(#stage3Grad)" />
      <text x="620" y="310" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="white" font-weight="bold" text-anchor="middle">Fine Transformer</text>
      <text x="620" y="325" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#D1FAE5" text-anchor="middle">p(f_t | f_<t, c_≤t, s_≤t)</text>
    </g>

    <!-- Inter-stage Vertical Connectors -->
    <!-- Stage 1 -> Stage 2 -->
    <path d="M 620 115 L 620 135" stroke="#94A3B8" stroke-width="2" stroke-dasharray="4 4" />
    <polygon points="620,135 625,127 615,127" fill="#94A3B8" />

    <!-- Stage 2 -> Stage 3 -->
    <path d="M 620 235 L 620 255" stroke="#94A3B8" stroke-width="2" stroke-dasharray="4 4" />
    <polygon points="620,255 625,247 615,247" fill="#94A3B8" />
  </svg>
</div>

### Mathematical Framework

**Semantic tokens** $s_t$ from w2v-BERT (self-supervised SSL model):

$$s_t = \text{Quantize}(f_{\text{w2v-BERT}}(x_t))$$

**Coarse acoustic tokens** $c_t^{(1:2)}$ from first 2 RVQ levels:

$$c_t^{(1:2)} = \text{RVQ}_{1:2}(E(x_t))$$

**Fine acoustic tokens** $f_t^{(1:L)}$ from all RVQ levels:

$$f_t^{(1:L)} = \text{RVQ}_{1:L}(E(x_t))$$

**Joint probability** modeled autoregressively:

$$p(s, c, f) = \prod_t p(s_t | s_{<t}) \cdot p(c_t | c_{<t}, s_{\le t}) \cdot p(f_t | f_{<t}, c_{\le t}, s_{\le t})$$

### Key Insight

Separating **semantic** (what is said) from **acoustic** (how it sounds) modeling enables:
- Long-term coherence (semantic LM)
- High-fidelity synthesis (acoustic LMs)
- Zero-shot voice cloning (prompt conditioning)

### Why This Matters

AudioLM showed that the same discrete tokens used for compression could power *generation*. This was the bridge between audio codecs and generative AI — and it led directly to text-to-music models.

---

## 6. MusicLM / MusicGen (2023): Text-to-Music Generation

**MusicLM** (Google) and **MusicGen** (Meta) extend AudioLM to **text-conditioned** music generation.

### MusicGen Architecture (Simpler, Single-Stage)

<div class="my-8 flex justify-center">
  <svg width="100%" max-width="800px" height="240" viewBox="0 0 800 240" fill="none" xmlns="http://www.w3.org/2000/svg" class="w-full">
    <defs>
      <linearGradient id="textEncGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#2563EB" />
        <stop offset="100%" stop-color="#06B6D4" />
      </linearGradient>
      <linearGradient id="transLMGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#D946EF" />
        <stop offset="100%" stop-color="#EA580C" />
      </linearGradient>
      <linearGradient id="codecDecGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#10B981" />
        <stop offset="100%" stop-color="#059669" />
      </linearGradient>
      <filter id="shadowMusicGen" x="-5%" y="-5%" width="110%" height="110%" filterUnits="userSpaceOnUse">
        <feDropShadow dx="2" dy="4" stdDeviation="4" flood-opacity="0.1" />
      </filter>
    </defs>

    <rect width="800" height="240" rx="12" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="1"/>

    <!-- Left Conditioning Block -->
    <g filter="url(#shadowMusicGen)">
      <rect x="40" y="45" width="160" height="150" rx="10" fill="url(#textEncGrad)" />
      <text x="120" y="85" font-family="system-ui, -apple-system, sans-serif" font-size="16" fill="white" font-weight="bold" text-anchor="middle">Text Encoder</text>
      <text x="120" y="110" font-family="system-ui, -apple-system, sans-serif" font-size="13" fill="#E0F2FE" text-anchor="middle">T5 / CLAP</text>
      
      <rect x="55" y="135" width="130" height="40" rx="4" fill="#0369A1" />
      <text x="120" y="152" font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="white" font-weight="bold" text-anchor="middle">"Upbeat electronic</text>
      <text x="120" y="166" font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="white" font-weight="bold" text-anchor="middle">music with bass..."</text>
    </g>

    <path d="M 200 120 L 280 120" stroke="#94A3B8" stroke-width="2" />
    <polygon points="280,120 272,115 272,125" fill="#94A3B8" />
    <text x="240" y="110" font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="#475569" font-weight="bold" text-anchor="middle">Embeds</text>

    <!-- Center Transformer Block -->
    <g filter="url(#shadowMusicGen)">
      <rect x="280" y="45" width="240" height="150" rx="10" fill="url(#transLMGrad)" />
      <text x="400" y="80" font-family="system-ui, -apple-system, sans-serif" font-size="16" fill="white" font-weight="bold" text-anchor="middle">Transformer LM</text>
      <text x="400" y="105" font-family="system-ui, -apple-system, sans-serif" font-size="13" fill="#FFE4E6" text-anchor="middle">Single-Stage Decoder</text>
      
      <!-- Delayed Pattern Visual -->
      <g transform="translate(325, 125)">
        <rect x="0" y="0" width="30" height="15" rx="2" fill="#BE185D" />
        <text x="15" y="11" font-family="system-ui, -apple-system, sans-serif" font-size="9" fill="white" font-weight="bold" text-anchor="middle">L1</text>
        
        <rect x="40" y="0" width="30" height="15" rx="2" fill="#BE185D" />
        <text x="55" y="11" font-family="system-ui, -apple-system, sans-serif" font-size="9" fill="white" font-weight="bold" text-anchor="middle">L1</text>

        <rect x="20" y="20" width="30" height="15" rx="2" fill="#9D174D" />
        <text x="35" y="31" font-family="system-ui, -apple-system, sans-serif" font-size="9" fill="white" font-weight="bold" text-anchor="middle">L2</text>
        
        <rect x="60" y="20" width="30" height="15" rx="2" fill="#9D174D" />
        <text x="75" y="31" font-family="system-ui, -apple-system, sans-serif" font-size="9" fill="white" font-weight="bold" text-anchor="middle">L2</text>

        <text x="110" y="18" font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="white" font-weight="bold" text-anchor="middle">Delayed</text>
        <text x="110" y="30" font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="white" font-weight="bold" text-anchor="middle">Pattern</text>
      </g>
    </g>

    <path d="M 520 120 L 600 120" stroke="#94A3B8" stroke-width="2" />
    <polygon points="600,120 592,115 592,125" fill="#94A3B8" />
    <text x="560" y="110" font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="#475569" font-weight="bold" text-anchor="middle">Tokens</text>

    <!-- Right EnCodec Decoder -->
    <g filter="url(#shadowMusicGen)">
      <rect x="600" y="45" width="160" height="150" rx="10" fill="url(#codecDecGrad)" />
      <text x="680" y="85" font-family="system-ui, -apple-system, sans-serif" font-size="15" fill="white" font-weight="bold" text-anchor="middle">EnCodec Decoder</text>
      
      <circle cx="680" cy="130" r="22" fill="#047857" />
      <!-- Wave symbol -->
      <path d="M 665 130 Q 672.5 120 680 130 T 695 130" stroke="white" stroke-width="2" fill="none" />
      
      <text x="680" y="175" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="#D1FAE5" text-anchor="middle">Output Waveform</text>
    </g>
  </svg>
</div>

### Delayed Pattern for Parallel Generation

Instead of sequential token-by-token, MusicGen uses **delayed pattern**:

```
Level 1:  [t1] [t2] [t3] [t4] [t5] ...
Level 2:       [t1] [t2] [t3] [t4] ...
Level 3:            [t1] [t2] [t3] ...
...
```

All levels predicted in **single forward pass** → 5-10× faster inference.

### Why This Matters

MusicGen simplified the AudioLM approach — single-stage, text-conditioned, with the delayed pattern trick for efficiency. It proved that high-quality music generation could work with a **single transformer** and a good codec.

---

## 7. Modern Developments (2024+)

### DAC (Descript Audio Codec)

- **Residual VQ** with improved codebook learning
- **Better perceptual quality** at ultra-low bitrates (0.5–1.5 kbps)
- Open-source, widely adopted for TTS and voice conversion

### X-Codec 2 / HiFi-Codec

- **Multi-band** processing (subband coding)
- **Generative adversarial** training with improved discriminators
- **Sample-rate independent** representations

### Language Model Integration

| Model | Codec | LM Architecture | Key Feature |
|-------|-------|-----------------|-------------|
| AudioLM | EnCodec | 3-stage Transformer | Semantic + Acoustic separation |
| MusicGen | EnCodec | Single Transformer | Delayed pattern, text-conditioned |
| VALL-E | EnCodec | Transformer (AR + NAR) | In-context TTS, 3s prompt |
| VoiceBox | EnCodec | Flow Matching | Non-autoregressive, editable |
| AudioBox | DAC | Transformer + Flow | Unified audio generation |

---

## 8. Comparative Summary

### Bitrate vs Quality Trade-offs

<div class="my-8 flex justify-center">
  <svg width="100%" max-width="800px" height="300" viewBox="0 0 800 300" fill="none" xmlns="http://www.w3.org/2000/svg" class="w-full">
    <rect width="800" height="300" rx="12" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="1"/>

    <!-- Title / Axes -->
    <text x="400" y="30" font-family="system-ui, -apple-system, sans-serif" font-size="16" fill="#1E293B" font-weight="bold" text-anchor="middle">Bitrate vs. Quality Trade-offs</text>
    
    <!-- Y Axis (Quality) -->
    <line x1="80" y1="60" x2="80" y2="240" stroke="#94A3B8" stroke-width="2" />
    <text x="35" y="150" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="#64748B" font-weight="bold" transform="rotate(-90 35 150)" text-anchor="middle">Quality (PESQ / MOS)</text>
    
    <!-- X Axis (Bitrate) -->
    <line x1="80" y1="240" x2="740" y2="240" stroke="#94A3B8" stroke-width="2" />
    <text x="410" y="280" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="#64748B" font-weight="bold" text-anchor="middle">Bitrate (kbps, log-scale)</text>

    <!-- Y-Axis Ticks -->
    <line x1="75" y1="70" x2="80" y2="70" stroke="#94A3B8" stroke-width="2" />
    <text x="65" y="74" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#64748B" text-anchor="end">5.0</text>
    
    <line x1="75" y1="112.5" x2="80" y2="112.5" stroke="#94A3B8" stroke-width="2" />
    <text x="65" y="116.5" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#64748B" text-anchor="end">4.5</text>
    
    <line x1="75" y1="155" x2="80" y2="155" stroke="#94A3B8" stroke-width="2" />
    <text x="65" y="159" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#64748B" text-anchor="end">4.0</text>

    <line x1="75" y1="197.5" x2="80" y2="197.5" stroke="#94A3B8" stroke-width="2" />
    <text x="65" y="201.5" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#64748B" text-anchor="end">3.5</text>

    <line x1="75" y1="240" x2="80" y2="240" stroke="#94A3B8" stroke-width="2" />
    <text x="65" y="244" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#64748B" text-anchor="end">3.0</text>

    <!-- X-Axis Ticks -->
    <g font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#64748B" text-anchor="middle">
      <line x1="120" y1="240" x2="120" y2="245" stroke="#94A3B8" stroke-width="2" />
      <text x="120" y="258">0.5</text>

      <line x1="220" y1="240" x2="220" y2="245" stroke="#94A3B8" stroke-width="2" />
      <text x="220" y="258">1.5</text>

      <line x1="320" y1="240" x2="320" y2="245" stroke="#94A3B8" stroke-width="2" />
      <text x="320" y="258">3</text>

      <line x1="420" y1="240" x2="420" y2="245" stroke="#94A3B8" stroke-width="2" />
      <text x="420" y="258">6</text>

      <line x1="520" y1="240" x2="520" y2="245" stroke="#94A3B8" stroke-width="2" />
      <text x="520" y="258">12</text>

      <line x1="620" y1="240" x2="620" y2="245" stroke="#94A3B8" stroke-width="2" />
      <text x="620" y="258">24</text>

      <line x1="720" y1="240" x2="720" y2="245" stroke="#94A3B8" stroke-width="2" />
      <text x="720" y="258">64</text>
    </g>

    <!-- Grid Lines -->
    <g stroke="#E2E8F0" stroke-width="1" stroke-dasharray="2 2">
      <line x1="80" y1="70" x2="740" y2="70" />
      <line x1="80" y1="112.5" x2="740" y2="112.5" />
      <line x1="80" y1="155" x2="740" y2="155" />
      <line x1="80" y1="197.5" x2="740" y2="197.5" />
      
      <line x1="120" y1="60" x2="120" y2="240" />
      <line x1="220" y1="60" x2="220" y2="240" />
      <line x1="320" y1="60" x2="320" y2="240" />
      <line x1="420" y1="60" x2="420" y2="240" />
      <line x1="520" y1="60" x2="520" y2="240" />
      <line x1="620" y1="60" x2="620" y2="240" />
      <line x1="720" y1="60" x2="720" y2="240" />
    </g>

    <!-- Data Points with labels -->
    <!-- VQ-VAE (PESQ ~3.2) -->
    <circle cx="150" cy="223" r="6" fill="#4F46E5" stroke="white" stroke-width="2" />
    <text x="160" y="227" font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="#4F46E5" font-weight="600">VQ-VAE (recon only)</text>

    <!-- MP3 64kbps (PESQ ~3.8) -->
    <circle cx="720" cy="172" r="6" fill="#F59E0B" stroke="white" stroke-width="2" />
    <text x="710" y="165" font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="#D97706" font-weight="600" text-anchor="end">MP3 64kbps</text>

    <!-- Opus 12kbps (PESQ ~3.9) -->
    <circle cx="520" cy="163.5" r="6" fill="#EF4444" stroke="white" stroke-width="2" />
    <text x="510" y="157" font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="#EF4444" font-weight="600" text-anchor="end">Opus 12kbps</text>

    <!-- SoundStream 12kbps (PESQ ~4.2) -->
    <circle cx="520" cy="138" r="6" fill="#3B82F6" stroke="white" stroke-width="2" />
    <text x="532" y="142" font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="#1D4ED8" font-weight="600">SoundStream 12kbps</text>

    <!-- EnCodec 24kbps (PESQ ~4.35) -->
    <circle cx="620" cy="125.25" r="6" fill="#10B981" stroke="white" stroke-width="2" />
    <text x="632" y="129.25" font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="#047857" font-weight="600">EnCodec 24kbps</text>
  </svg>
</div>

### Architecture Evolution Timeline

<div class="my-8 flex justify-center">
  <svg width="100%" max-width="800px" height="150" viewBox="0 0 800 150" fill="none" xmlns="http://www.w3.org/2000/svg" class="w-full">
    <rect width="800" height="150" rx="12" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="1"/>

    <!-- Timeline Main Line -->
    <line x1="50" y1="80" x2="750" y2="80" stroke="#CBD5E1" stroke-width="4" />

    <!-- Nodes: 2017, 2019, 2021, 2022, 2023, 2024+ -->
    <!-- 2017 (VQ-VAE) -->
    <circle cx="100" cy="80" r="8" fill="#4F46E5" stroke="white" stroke-width="3" />
    <text x="100" y="45" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="#64748B" font-weight="bold" text-anchor="middle">2017</text>
    <text x="100" y="110" font-family="system-ui, -apple-system, sans-serif" font-size="13" fill="#1E293B" font-weight="bold" text-anchor="middle">VQ-VAE</text>
    <text x="100" y="125" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#64748B" text-anchor="middle">Discrete Latents</text>

    <!-- 2019 (VQ-VAE-2) -->
    <circle cx="220" cy="80" r="8" fill="#7C3AED" stroke="white" stroke-width="3" />
    <text x="220" y="45" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="#64748B" font-weight="bold" text-anchor="middle">2019</text>
    <text x="220" y="110" font-family="system-ui, -apple-system, sans-serif" font-size="13" fill="#1E293B" font-weight="bold" text-anchor="middle">VQ-VAE-2</text>
    <text x="220" y="125" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#64748B" text-anchor="middle">Hierarchical</text>

    <!-- 2021 (SoundStream) -->
    <circle cx="350" cy="80" r="8" fill="#3B82F6" stroke="white" stroke-width="3" />
    <text x="350" y="45" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="#64748B" font-weight="bold" text-anchor="middle">2021</text>
    <text x="350" y="110" font-family="system-ui, -apple-system, sans-serif" font-size="13" fill="#1E293B" font-weight="bold" text-anchor="middle">SoundStream</text>
    <text x="350" y="125" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#64748B" text-anchor="middle">RVQ + Adv Loss</text>

    <!-- 2022 (EnCodec) -->
    <circle cx="470" cy="80" r="8" fill="#10B981" stroke="white" stroke-width="3" />
    <text x="470" y="45" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="#64748B" font-weight="bold" text-anchor="middle">2022</text>
    <text x="470" y="110" font-family="system-ui, -apple-system, sans-serif" font-size="13" fill="#1E293B" font-weight="bold" text-anchor="middle">EnCodec</text>
    <text x="470" y="125" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#64748B" text-anchor="middle">SEANet + LM</text>

    <!-- 2023 (AudioLM / MusicGen) -->
    <circle cx="590" cy="80" r="8" fill="#EC4899" stroke="white" stroke-width="3" />
    <text x="590" y="45" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="#64748B" font-weight="bold" text-anchor="middle">2023</text>
    <text x="590" y="110" font-family="system-ui, -apple-system, sans-serif" font-size="13" fill="#1E293B" font-weight="bold" text-anchor="middle">AudioLM</text>
    <text x="590" y="125" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#64748B" text-anchor="middle">Audio Token LMs</text>

    <!-- 2024+ (DAC / Flows) -->
    <circle cx="710" cy="80" r="8" fill="#EF4444" stroke="white" stroke-width="3" />
    <text x="710" y="45" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="#64748B" font-weight="bold" text-anchor="middle">2024+</text>
    <text x="710" y="110" font-family="system-ui, -apple-system, sans-serif" font-size="13" fill="#1E293B" font-weight="bold" text-anchor="middle">DAC / Flows</text>
    <text x="710" y="125" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#64748B" text-anchor="middle">Flow Matching</text>
  </svg>
</div>

---

## 9. Practical Implementation Tips

### Choosing a Codec for Your Use Case

| Requirement | Recommended Codec |
|-------------|-------------------|
| Real-time streaming (low latency) | EnCodec, DAC |
| Highest quality at low bitrate | DAC, X-Codec 2 |
| Text-to-audio generation | EnCodec + MusicGen/AudioLM |
| Voice cloning / TTS | EnCodec + VALL-E/VoiceBox |
| Research / experimentation | VQ-VAE (simpler), EnCodec (SOTA) |

### Key Hyperparameters

```python
# EnCodec typical config
sample_rate = 24000
channels = 1
segment_length = 1.0  # seconds
codebook_size = 1024
codebook_dim = 128
num_codebooks = 8     # RVQ levels
strides = [2, 4, 5, 8]  # Downsampling factors
```

---

## 10. Future Directions

1. **Unified audio-language models**: Joint modeling of text, audio, and music (e.g., **AudioPaLM**, **LTU**)
2. **Flow matching / diffusion in latent space**: Non-autoregressive generation (e.g., **VoiceBox**, **AudioBox**)
3. **Streaming / causal models**: Real-time encoding with bounded latency
4. **Cross-lingual / low-resource**: Codecs that generalize across languages with minimal data
5. **Interpretability**: Disentangling content, speaker, prosody, and environment in latent space

---

## Closing Thoughts

Looking at this evolution — from VQ-VAE's simple discrete latents to AudioLM's three-stage language modeling to MusicGen's streamlined text-to-music — a clear pattern emerges. Each generation of audio encoders **borrowed ideas from other domains** and adapted them to audio:

| Domain Borrowed From | Idea | Audio Adaptation |
|---------------------|------|------------------|
| Computer Vision (VQ-VAE) | Discrete latent codes | Audio reconstruction with codebooks |
| NLP (Transformer) | Autoregressive language modeling | Tokenized audio → audio LM |
| Computer Vision (GANs) | Adversarial discriminators | Perceptual quality for codecs |
| NLP (T5) | Cross-attention conditioning | Text-to-music generation |

The trend is accelerating. As of 2024-2026, we're seeing **flow matching** (from diffusion/ODE literature) replace autoregressive LMs, **multi-modal models** that jointly understand speech, music, and text, and **streaming architectures** that enable real-time neural compression on device.

If you're working in this space, my advice: **understand the codec first**. The quality of your final system — whether it's TTS, music generation, or audio understanding — is gated by the quality of your encoder. The best language model in the world can't fix a bad codec.

---

## References

1. **VQ-VAE**: van den Oord et al., "Neural Discrete Representation Learning", NeurIPS 2017
2. **VQ-VAE-2**: Razavi et al., "Generating Diverse High-Fidelity Images with VQ-VAE-2", NeurIPS 2019
3. **SoundStream**: Zeghidour et al., "SoundStream: An End-to-End Neural Audio Codec", IEEE/ACM TASLP 2021
4. **EnCodec**: Défossez et al., "High Fidelity Neural Audio Compression", arXiv 2022
5. **AudioLM**: Borsos et al., "AudioLM: A Language Modeling Approach to Audio Generation", IEEE/ACM TASLP 2022
6. **MusicGen**: Copet et al., "Simple and Controllable Music Generation", NeurIPS 2023
7. **DAC**: Kumar et al., "High-Fidelity Audio Compression with Improved RVQ", arXiv 2023
8. **VALL-E**: Wang et al., "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers", arXiv 2023
9. **VoiceBox**: Le et al., "VoiceBox: Text-Guided Multilingual Universal Speech Generation at Scale", arXiv 2023

---

## Appendix: Mathematical Notation Reference

| Symbol | Meaning |
|--------|---------|
| $x \in \mathbb{R}^T$ | Input waveform (T samples) |
| $z_e$ | Continuous encoder output |
| $z_q$ | Quantized latent |
| $\mathcal{C} = \{e_k\}$ | Codebook of $K$ entries |
| $E_\phi, D_\theta$ | Encoder/Decoder networks |
| $\text{sg}[\cdot]$ | Stop-gradient operator |
| $L$ | Number of RVQ levels |
| $\lambda$ | Loss weighting hyperparameters |
| $p(\cdot)$ | Probability distribution |
| $\| \cdot \|_1, \| \cdot \|_2$ | L1/L2 norms |

---

*This post is part of my research notes on neural audio compression. For implementation code, see my [GitHub repositories](https://github.com/eladaspis).*
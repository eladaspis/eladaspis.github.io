---
title: "Getting Started with My PhD Research"
description: "An introduction to my research on multi-agent turn-taking."
date: 2026-07-17
math: true
tags: [research, multi-agent]
---

I'm excited to share that I've started my PhD at Ben Gurion University of the Negev, working under the supervision of Dr. Eliya Nachmani. My research focuses on **multi-agent turn-taking** — how multiple AI agents can coordinate and communicate effectively in conversation.

## Why Multi-Agent Turn-Taking?

In real-world conversations, speakers take turns, interrupt, and adapt to each other. Replicating this behavior in AI systems is crucial for building natural multi-agent interactions. The core challenge can be expressed as optimizing a turn-taking policy:

$$
\pi^* = \arg\max_\pi \mathbb{E}_{(a_1, \ldots, a_N) \sim \mathcal{A}} \left[ \sum_{t=1}^{T} R(a_t, \text{context}_t) \right]
$$

where $a_t$ represents the agent acting at time $t$ and $\text{context}_t$ captures the full conversation history.

## What's Next

I'll be sharing updates about my research progress, paper reviews, and interesting findings along the way. Stay tuned!

---

*Feel free to reach out if you're interested in multi-agent systems or collaborative AI research.*

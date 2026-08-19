---
title: Diffusion LM Nano
emoji: 🧩
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
---

# Diffusion LM Nano

This Space visualizes iterative token unmasking. Set `CHECKPOINT_PATH` to a
compatible trusted checkpoint. For a publication checkpoint, set
`TOKENIZER_PATH` to the matching `gpt2-nano` tokenizer directory containing
`vocab.json` and `merges.json`. The measured 10,000-step publication checkpoint
collapsed onto the end-of-word token and produces blank output. This Space
visualizes the failure and does not present the checkpoint as a usable language
model.

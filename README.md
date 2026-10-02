# ⚡ LearniBridge: Learnable Calibration of Feature Caching for Diffusion Model Acceleration

Diffusion Transformers produce high-quality images and videos, but repeated computation across denoising steps makes inference expensive. Feature caching reduces this cost, while directly reusing historical features can accumulate errors at high acceleration ratios.

**LearniBridge** learns lightweight LoRA-based corrections in the final Transformer block to calibrate cached features. This repository provides the complete **cache → train → infer** workflow for **FLUX.1-dev, HunyuanVideo, and Wan2.1-T2V-1.3B**.

## 🧩 Overview

![LearniBridge Architecture](assets/pipeline.png)

The pipeline above combines feature caching with lightweight LoRA-based calibration to reduce feature-reuse errors during accelerated inference.

## ✨ Showcase

The image and video examples below showcase qualitative generation results with LearniBridge.

<p align="center">
  <a href="assets/results/image.png">
    <img src="assets/results/image.png" alt="LearniBridge image generation results" width="100%">
  </a>
</p>
<p align="center">
  <sub>Image generation results · <a href="assets/results/image.png">View full-resolution image ↗</a></sub>
</p>

<p align="center">
  <a href="assets/results/video.mp4">
    <img src="assets/results/video-preview.jpg" alt="LearniBridge video generation results — click to watch" width="100%">
  </a>
</p>
<p align="center">
  <sub>Video generation results · <a href="assets/results/video.mp4">▶ Watch full video</a></sub>
</p>

## 🚀 Usage

Choose a model, install its dependencies, then **cache features → train adapters → run inference**.

| Model | Code and instructions |
| --- | --- |
| FLUX.1-dev | [flux/](flux/README.md) |
| HunyuanVideo | [hunyuan/](hunyuan/README.md) |
| Wan2.1-T2V-1.3B | [wan/](wan/README.md) |

Each model folder includes its own dependencies and entry point; no root package installation is needed.

<details>
<summary>Inference examples</summary>

Set up the model paths and adapters using the corresponding guide first.

### FLUX.1-dev

```bash
python flux/run.py infer --N 5 \
  --model-path "$FLUX_MODEL" --adapter-dir runs/flux/N5/adapters \
  --prompt "Three-quarters front view of a blue 1977 Porsche 911 coming around a curve in a mountain road" \
  --num-steps 50 --seed 42 --output-dir runs/flux/N5/images
```

### HunyuanVideo

```bash
python hunyuan/run.py infer --N 5 \
  --model-base "$MODEL_BASE" --dit-weight "$DIT_WEIGHT" \
  --lora-dir runs/hunyuan/N5/adapters \
  --prompt "$(head -n 1 hunyuan/prompts/sample_50.txt)" \
  --block-idx 39 --target-modules auto --lora-rank 32 --lora-alpha 64 \
  --video-size 544 960 --video-length 49 --infer-steps 50 \
  --seed 42 --embedded-cfg-scale 6.0 --flow-shift 7.0 \
  --flow-reverse --use-cpu-offload --save-path runs/hunyuan/N5/videos
```

### Wan2.1-T2V-1.3B

```bash
mkdir -p runs/wan/N5/videos
python wan/run.py infer --N 5 \
  --task t2v-1.3B --ckpt_dir "$CKPT_DIR" --lora_dir runs/wan/N5/adapters \
  --prompt "$(head -n 1 wan/prompts/wan_sample_50.txt)" \
  --lora_rank 32 --lora_alpha 64 \
  --size '832*480' --frame_num 81 --sample_steps 50 \
  --base_seed 42 --sample_shift 5 --sample_guide_scale 5 \
  --offload_model True --t5_cpu --save_file runs/wan/N5/videos/example.mp4
```

</details>

## 📄 License

Original LearniBridge code is licensed under [Apache-2.0](LICENSE). Third-party code and model weights retain their own terms; see [NOTICE](NOTICE) and [third-party notices](THIRD_PARTY_NOTICES.md).

# LearniBridge: Learnable Calibration of Feature Caching for Diffusion Models Acceleration

### Overview
Diffusion Transformers (DiTs) have driven substantial progress in image and video generation but suffer from prohibitive computational costs.
Feature caching accelerates inference by reusing intermediate representations.
Existing methods rely on historical features for implementation simplicity, 
yet suffer from severe error accumulation at high acceleration ratios.
To address this limitation, we investigate the nature of the requisite feature correction.
We demonstrate that the optimal calibration update is characterized by a shared low-rank subspace across diverse prompts.
Guided by this structural insight, we propose \textit{LearniBridge}, a learnable calibration mechanism for feature caching that bridges multiple timesteps through lightweight LoRA updates.
This mechanism enables effective calibration requiring only 3--5 training samples.
Extensive experiments on image and video generation show that \textit{LearniBridge} achieves up to $5.87\times$, $5.75\times$, and $4.10\times$ acceleration
on FLUX, HunyuanVideo, and WAN~2.1, respectively. On WAN~2.1, it improves VBench by 1.28\% over the previous SOTA at $4.10\times$ acceleration.


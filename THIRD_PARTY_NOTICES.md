# Third-party notices

The root `LICENSE` applies to original LearniBridge code. It does not replace the licenses of upstream code, model weights, or datasets.

| Component | Source | License information |
| --- | --- | --- |
| FLUX model implementation and pipeline | [Hugging Face Diffusers](https://github.com/huggingface/diffusers) | Apache-2.0; imported as a dependency |
| Wan model-facing code | [Wan2.1](https://github.com/Wan-Video/Wan2.1) | Upstream attribution is retained; see `licenses/Wan2.1-Apache-2.0.txt` |
| Hunyuan model-facing code | [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo) | See `licenses/HunyuanVideo-Community.txt`; these files are not relicensed by the root Apache-2.0 grant |

The model-facing scripts have local path/CLI integration, feature-caching and LoRA-loading changes. Their upstream attribution and applicable license texts are retained.

Base model weights, text encoders and datasets are downloaded separately and remain subject to their respective provider terms. Do not infer a model-weight license from this repository's software license.

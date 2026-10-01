# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
from datetime import datetime
import logging
import os
import json
import types
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore')

import torch, random
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video, cache_image, str2bool

import gc
from contextlib import contextmanager
import torchvision.transforms.functional as TF
import torch.cuda.amp as amp
import numpy as np
import math
from wan.modules.model import sinusoidal_embedding_1d
from wan.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from tqdm import tqdm
from wan.modules.model import WanAttentionBlock
import torch.nn as nn
import torch.nn.functional as F

EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2v-14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2i-14B": {
        "prompt": "一个朴素端庄的美人",
    },
    "i2v-14B": {
        "prompt": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. "
                  "The fluffy-furred feline gazes directly at the camera with a relaxed expression. "
                  "Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, "
                  "and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, "
                  "as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's "
                  "intricate details and the refreshing atmosphere of the seaside.",
        "image": "examples/i2v_input.JPG",
    },
}

def save_feature(feature: torch.Tensor, save_path: str, feature_name: str, time_step: int) -> None:
    """ Save feature tensor to specified path. """
    os.makedirs(save_path, exist_ok=True)
    file_name = f"{feature_name}_step_{time_step}.pt"
    torch.save(feature, os.path.join(save_path, file_name))

def t2v_generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
    r"""
    Generates video frames from text prompt using diffusion process.
    """
    # preprocess
    num_frames = frame_num
    target_shape = (self.vae.model.z_dim, (num_frames - 1) // self.vae_stride[0] + 1,
                    size[1] // self.vae_stride[1],
                    size[0] // self.vae_stride[2])

    seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                        (self.patch_size[1] * self.patch_size[2]) *
                        target_shape[1] / self.sp_size) * self.sp_size

    if n_prompt == "":
        n_prompt = self.sample_neg_prompt
    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=self.device)
    seed_g.manual_seed(seed)

    if not self.t5_cpu:
        self.text_encoder.model.to(self.device)
        context = self.text_encoder([input_prompt], self.device)
        context_null = self.text_encoder([n_prompt], self.device)
        if offload_model:
            self.text_encoder.model.cpu()
    else:
        context = self.text_encoder([input_prompt], torch.device('cpu'))
        context_null = self.text_encoder([n_prompt], torch.device('cpu'))
        context = [t.to(self.device) for t in context]
        context_null = [t.to(self.device) for t in context_null]

    noise = [
        torch.randn(
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=torch.float32,
            device=self.device,
            generator=seed_g)
    ]

    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(self.model, 'no_sync', noop_no_sync)

    with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

        if sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=self.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")

        latents = noise
        arg_c = {'context': context, 'seq_len': seq_len}
        arg_null = {'context': context_null, 'seq_len': seq_len}

        for _, t in enumerate(tqdm(timesteps)):
            latent_model_input = latents
            timestep = [t]
            timestep = torch.stack(timestep)

            self.model.to(self.device)
            noise_pred_cond = self.model(
                latent_model_input, t=timestep, **arg_c)[0]
            noise_pred_uncond = self.model(
                latent_model_input, t=timestep, **arg_null)[0]

            noise_pred = noise_pred_uncond + guide_scale * (
                noise_pred_cond - noise_pred_uncond)

            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0),
                t,
                latents[0].unsqueeze(0),
                return_dict=False,
                generator=seed_g)[0]
            latents = [temp_x0.squeeze(0)]

        x0 = latents
        if offload_model:
            self.model.cpu()
            torch.cuda.empty_cache()
        if self.rank == 0:
            videos = self.vae.decode(x0)

    del noise, latents
    del sample_scheduler
    if offload_model:
        gc.collect()
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    return videos[0] if self.rank == 0 else None

def step_forward(
    self,
    x,
    t,
    context,
    seq_len,
    clip_fea=None,
    y=None,
    save_path: str = "feature/4",
):
    os.makedirs(save_path, exist_ok=True)
    time_step = self.cnt.item() if isinstance(self.cnt, torch.Tensor) else self.cnt
    time_step = int(time_step / 2)

    if self.model_type == 'i2v':
        assert clip_fea is not None and y is not None

    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                  dim=1) for u in x
    ])

    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat(
                [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)
        context = torch.concat([context_clip, context], dim=1)

    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens)
    meta = {
        "e": e0.detach().cpu(),           # time embedding
        "seq_lens": seq_lens.detach().cpu(),
        "grid_sizes": grid_sizes.detach().cpu(),
        "freqs": self.freqs.detach().cpu(),
        "context": context.detach().cpu(),
        "context_lens": context_lens if context_lens is None else torch.tensor(context_lens),
        }

    self.layer_index = self.cnt % 2
    selected_blocks = getattr(self, "_learnibridge_cache_blocks", None)
    for block_idx, block in enumerate(self.blocks):
        x = block(x, **kwargs)
        if selected_blocks is None or block_idx in selected_blocks:
            save_feature(x.detach().cpu(), save_path, f"block_{block_idx}_{self.layer_index}_output", time_step)
    meta_path = os.path.join(save_path, f"meta_step_{time_step}_{self.layer_index}.pt")
    torch.save(meta, meta_path)

    save_feature(x.detach().cpu(), save_path, "final_layer_output", time_step)
    x = self.head(x, e)
    x = self.unpatchify(x, grid_sizes)
    self.cnt += 1
    if self.cnt >= self.num_steps:
        self.cnt = 0
    return [u.float() for u in x]


def _validate_args(args):
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    if args.sample_steps is None:
        args.sample_steps = 40 if "i2v" in args.task else 50

    if args.sample_shift is None:
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832"]:
            args.sample_shift = 3.0

    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81

    if "t2i" in args.task:
        assert args.frame_num == 1, f"Unsupport frame_num {args.frame_num} for task {args.task}"

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)
    assert args.size in SUPPORTED_SIZES[args.task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument("--task", type=str, default="t2v-14B", choices=list(WAN_CONFIGS.keys()))
    parser.add_argument("--size", type=str, default="1280*720", choices=list(SIZE_CONFIGS.keys()))
    parser.add_argument("--frame_num", type=int, default=None)
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--offload_model", type=str2bool, default=None)
    parser.add_argument("--ulysses_size", type=int, default=1)
    parser.add_argument("--ring_size", type=int, default=1)
    parser.add_argument("--t5_fsdp", action="store_true", default=False)
    parser.add_argument("--t5_cpu", action="store_true", default=False)
    parser.add_argument("--dit_fsdp", action="store_true", default=False)
    parser.add_argument("--save_file", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--use_prompt_extend", action="store_true", default=False)
    parser.add_argument("--prompt_extend_method", type=str, default="local_qwen", choices=["dashscope", "local_qwen"])
    parser.add_argument("--prompt_extend_model", type=str, default=None)
    parser.add_argument("--prompt_extend_target_lang", type=str, default="ch", choices=["ch", "en"])
    parser.add_argument("--base_seed", type=int, default=-1)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--sample_solver", type=str, default='unipc', choices=['unipc', 'dpm++'])
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--sample_shift", type=float, default=None)
    parser.add_argument("--sample_guide_scale", type=float, default=5.0)
    parser.add_argument("--teacache_thresh", type=float, default=0.2)
    parser.add_argument("--use_ret_steps", action="store_true", default=False)
    parser.add_argument("--feature-dir", type=Path, default=Path("features/wan"))
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--cache-blocks", type=int, nargs="+")

    args = parser.parse_args()
    if args.max_prompts is not None and args.max_prompts < 1:
        parser.error("--max-prompts must be positive.")
    if args.prompt_file and int(os.getenv("WORLD_SIZE", "1")) != 1:
        parser.error("Batch prompt caching currently uses one GPU process.")
    _validate_args(args)
    return args


def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)

def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(f"offload_model is not specified, set to {args.offload_model}.")

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (args.t5_fsdp or args.dit_fsdp)
        assert not (args.ulysses_size > 1 or args.ring_size > 1)

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model, is_vl="i2v" in args.task)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                is_vl="i2v" in args.task,
                device=rank)
        else:
            raise NotImplementedError

    cfg = WAN_CONFIGS[args.task]
    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if "t2v" in args.task or "t2i" in args.task:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        logging.info(f"Input prompt: {args.prompt}")

        feature_directory = args.feature_dir.resolve()
        if feature_directory.exists() and any(feature_directory.iterdir()):
            raise FileExistsError(f"Use an empty feature directory: {feature_directory}")
        feature_directory.mkdir(parents=True, exist_ok=True)
        logging.info("Creating WanT2V pipeline.")
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )

        if args.cache_blocks is not None and any(
            block < 0 or block >= len(wan_t2v.model.blocks) for block in args.cache_blocks
        ):
            raise ValueError("--cache-blocks contains an invalid block index.")
        def feature_forward(current, *arguments, **keywords):
            return step_forward(current, *arguments, save_path=str(feature_directory), **keywords)
        wan_t2v.generate = types.MethodType(t2v_generate, wan_t2v)
        wan_t2v.model.enable_teacache = True
        wan_t2v.model.forward = types.MethodType(feature_forward, wan_t2v.model)
        wan_t2v.model.cnt = 0
        wan_t2v.model.layer_index = 0
        wan_t2v.model.num_steps = args.sample_steps * 2
        wan_t2v.model._learnibridge_cache_blocks = (
            set(args.cache_blocks) if args.cache_blocks is not None else None
        )

        logging.info("Generating video ...")
        video = wan_t2v.generate(
            args.prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)

        if rank == 0:
            if args.save_file is None:
                formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                formatted_prompt = args.prompt.replace(" ", "_").replace("/", "_")[:50]
                suffix = '.png' if "t2i" in args.task else '.mp4'
                args.save_file = f"{args.task}_{args.size}_{args.ulysses_size}_{args.ring_size}_{formatted_prompt}_{formatted_time}{suffix}"

            if "t2i" in args.task:
                logging.info(f"Saving generated image to {args.save_file}")
                cache_image(
                    tensor=video.squeeze(1)[None],
                    save_file=args.save_file,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1))
            else:
                logging.info(f"Saving generated video to {args.save_file}")
                cache_video(
                    tensor=video[None],
                    save_file=args.save_file,
                    fps=cfg.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1))

    logging.info("Finished.")

if __name__ == "__main__":
    args = _parse_args()
    prompts = (
        [line.strip() for line in args.prompt_file.read_text().splitlines() if line.strip()]
        if args.prompt_file else [args.prompt]
    )
    if args.max_prompts is not None:
        prompts = prompts[:args.max_prompts]
    if not prompts:
        raise ValueError("The prompt file is empty.")
    feature_root = args.feature_dir.resolve()
    if feature_root.exists() and any(feature_root.iterdir()):
        raise FileExistsError(f"Use an empty feature directory: {feature_root}")
    for prompt_index, prompt in enumerate(prompts):
        run_args = argparse.Namespace(**vars(args))
        run_args.prompt = prompt
        run_args.feature_dir = feature_root / str(prompt_index)
        if args.save_file is None:
            run_args.save_file = str(run_args.feature_dir / "reference.mp4")
        elif len(prompts) > 1:
            requested = Path(args.save_file)
            run_args.save_file = str(requested.with_name(f"{requested.stem}_{prompt_index}{requested.suffix}"))
        Path(run_args.save_file).parent.mkdir(parents=True, exist_ok=True)
        generate(run_args)
        (run_args.feature_dir / "metadata.json").write_text(json.dumps({
            "prompt": run_args.prompt, "seed": run_args.base_seed,
            "steps": run_args.sample_steps, "cache_blocks": args.cache_blocks,
        }, indent=2) + "\n")
        gc.collect()
        torch.cuda.empty_cache()

# online_gen的代码
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
from learnibridge.schedule import (
    adapter_filename, add_interval_argument, resolve_inference_schedule,
    validate_cached_source, validate_checkpoint_schedule, validate_interval_adapters,
)
from datetime import datetime
import logging
import os
import sys
import warnings
from pathlib import Path

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

# ===== LoRA 工具 =====
from learnibridge.lora import inject_lora, load_adapter_state, set_lora_enabled
# from wan.modules.lora import merge_lora
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

def _ensure_lora_injected(block, rank:int, alpha:float, targets=None):
    """确保在 block 上注入过 LoRA（避免重复注入）。"""
    if targets is None:
        targets = [
            "self_attn.q","self_attn.k","self_attn.v","self_attn.o",
            "cross_attn.q","cross_attn.k","cross_attn.v","cross_attn.o",
            "ffn.0","ffn.2",
        ]
    # 粗略判断是否已经注入（看有没有 lora 参数）
    already = any("lora_" in n for n, _ in block.named_parameters())
    if not already:
        inject_lora(block, targets, rank=rank, alpha=alpha, dropout=0.0)

def load_prompts_from_txt(txt_path):
    prompts = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            prompts.append(line)
    return prompts

def _load_lora_for_step_layer(model, block_idx:int, lora_dir:Path, step:int, layer_idx:int,
                              rank:int, alpha:float, device:torch.device, dtype:torch.dtype) -> bool:
    """
    针对某个 step & layer_idx，加载对应 LoRA 到指定 block。
    返回 True 表示加载成功（或已加载），False 表示找不到文件。
    """
    schedule = getattr(model, "_learnibridge_schedule", None)
    lora_path = lora_dir / (
        adapter_filename("wan", step, layer_idx) if schedule is not None
        else f"block{block_idx}_step_{step}_{layer_idx}.pt"
    )
    if not lora_path.exists():
        # 兼容你的另一种命名：block29_step18_1.pt
        lora_path2 = lora_dir / f"block{block_idx}_step{step}_{layer_idx}.pt"
        if lora_path2.exists():
            lora_path = lora_path2
        else:
            return False

    block = model.blocks[block_idx]
    _ensure_lora_injected(block, rank=rank, alpha=alpha)
    set_lora_enabled(block, True)

    # 设备&精度对齐
    block.to(device=device, dtype=dtype)

    # 简单缓存：避免重复 load
    key = f"{block_idx}-{step}-{layer_idx}"
    if getattr(model, "_curr_lora_key", None) == key:
        return True

    ckpt = torch.load(lora_path, map_location=device)
    if schedule is not None:
        validate_checkpoint_schedule(ckpt, schedule, "wan", step, layer_idx)
    state = ckpt.get("state_dict", ckpt)
    if "layer_idx" in ckpt and ckpt["layer_idx"] != layer_idx:
        raise ValueError("Checkpoint CFG branch differs from the requested branch.")
    load_adapter_state(block, {
        name: value for name, value in state.items()
        if name.rsplit(".", 1)[-1] in {"lora_a", "lora_b"}
    })

    model._curr_lora_key = key
    print(f"[LoRA] Loaded {lora_path.name} into block {block_idx}")
    return True


def _make_step_set(start:int=10, end:int=38, stride:int=2):
    return set(range(start, end+1, stride))

def parse_step_set(arg_list):
    """解析命令行传入的步数，如 10 12 15"""
    return set(map(int, arg_list))

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
    save_path: str = "output/wan",
):
    os.makedirs(save_path, exist_ok=True)

    # === 推理时刻 step & layer 判定 ===
    time_step = self.cnt.item() if isinstance(self.cnt, torch.Tensor) else self.cnt
    time_step = int(time_step // 2)
    layer_idx = 0 if (self.cnt % 2 == 0) else 1 
    if time_step == 0:
        setattr(self, f"_learnibridge_pre_{layer_idx}", None)
        setattr(self, f"_learnibridge_source_step_{layer_idx}", None)
    set_lora_enabled(self.blocks, False)

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
        context_lens=context_lens
    )

    # === 配置（来自类属性或默认） ===
    target_block_idx = getattr(self, "target_block_idx", 29)
    skip_until = getattr(self, "skip_until", 28)         
    replace_steps = getattr(self, "replace_steps", _make_step_set(10, 38, 2))
    use_pred_first = getattr(self, "use_pred_first", True)
    pred_dir = Path(getattr(self, "pred_dir", Path(save_path) / "adapters"))
    lora_dir = Path(getattr(self, "lora_dir", Path(save_path) / "adapters"))
    lora_rank = int(getattr(self, "lora_rank", 4))
    lora_alpha = float(getattr(self, "lora_alpha", 8.0))

    try:
        param_dtype = next(self.blocks[0].parameters()).dtype
    except StopIteration:
        param_dtype = torch.float32

    if time_step in replace_steps:
        schedule = getattr(self, "_learnibridge_schedule", None)
        validate_cached_source(
            schedule, time_step, getattr(self, f"_learnibridge_source_step_{layer_idx}", None),
        )
        did_block29 = False

        if use_pred_first:
            pred_path = pred_dir / f"pred_block_{target_block_idx}_{layer_idx}_output_step_{time_step}.pt"
            if pred_path.exists():
                data = torch.load(pred_path, map_location="cpu")
                x = data.to(device=x.device, dtype=x.dtype)
                did_block29 = True
                print(f"[Info] step={time_step} layer={layer_idx}: used PRED {pred_path.name}")

        if not did_block29:
            ok = _load_lora_for_step_layer(
                model=self,
                block_idx=target_block_idx,
                lora_dir=lora_dir,
                step=time_step,
                layer_idx=layer_idx,
                rank=lora_rank,
                alpha=lora_alpha,
                device=x.device,
                dtype=param_dtype,
            )
            if not ok:
                if schedule is not None:
                    raise FileNotFoundError(f"Missing scheduled Wan adapter for step {time_step}, CFG branch {layer_idx}.")
                print(f"[Warn] step={time_step} layer={layer_idx}: no PRED/LoRA found, just skip.")
                if layer_idx == 0:
                    x += self.previous_residual_0
                else:
                    x += self.previous_residual_1
                # self.cnt += 1
                # if self.cnt >= self.num_steps:
                #     self.cnt = 0
                # if layer_idx == 0:
                #     step_forward.previous_output = step_forward.previous_output_0
                # else:
                #     step_forward.previous_output = step_forward.previous_output_1
                # return [u.clone() for u in step_forward.previous_output]
                
                # for b in self.blocks:
                #     x = b(x, **kwargs)
            else:
                x = self.blocks[target_block_idx](getattr(self, f"_learnibridge_pre_{layer_idx}"), **kwargs)
                
                # ###测试阶段验证
                # pred_path = pred_dir / f"pred_block_{target_block_idx}_{layer_idx}_output_step_{time_step}.pt"
                # data = torch.load(pred_path, map_location="cpu")
                # data = data.to(x.device)
                # loss = F.mse_loss(x, data)
                # print(loss)

    else:
        ori_x = x.clone()
        for block_idx, block in enumerate(self.blocks):
            x = block(x, **kwargs)
            if block_idx == 28:
                setattr(self, f"_learnibridge_pre_{layer_idx}", x.detach().clone())
                setattr(self, f"_learnibridge_source_step_{layer_idx}", time_step)
        if layer_idx == 0:
            self.previous_residual_0 = x - ori_x
        else:
            self.previous_residual_1 = x - ori_x

    x = self.head(x, e)
    x = self.unpatchify(x, grid_sizes)

    self.cnt += 1
    if self.cnt >= self.num_steps:
        self.cnt = 0
    output = [u.float() for u in x]
    if layer_idx == 0:
        step_forward.previous_output_0 = [u.clone() for u in output]
    else:
        step_forward.previous_output_1 = [u.clone() for u in output]
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
    # ====== Pred/LoRA 相关 ======
    parser.add_argument("--pred_dir", type=str, default="adapters", help="保存的 pred_*.pt 的目录")
    parser.add_argument("--lora_dir", type=str, default="adapters", help="LoRA block*.pt 所在目录")
    parser.add_argument("--lora_rank", type=int)
    parser.add_argument("--lora_alpha", type=float)
    parser.add_argument("--use_pred_first", action="store_true", default=False, help="优先使用 pred_block_*.pt")
    # 需要跳过并替换的 step 范围（默认 10..38 步长 2）
    parser.add_argument("--replace_steps", type=int, nargs="*", help="指定要跳过的步骤集合，例如：10 12 15 21 37")
    parser.add_argument("--replace_start", type=int)
    parser.add_argument("--replace_end", type=int)
    parser.add_argument("--replace_stride", type=int)
    add_interval_argument(parser)
    parser.add_argument(
        "--prompt_txt",
        type=str,
        default=None,
        help="Path to txt file, one prompt per line")

    args = parser.parse_args()
    _validate_args(args)
    if args.skip_interval is not None:
        if args.task != "t2v-1.3B":
            parser.error("--N mode supports the Wan2.1-T2V-1.3B backend.")
        if any(value is not None for value in (args.replace_start, args.replace_end, args.replace_stride)):
            parser.error("Do not combine --N with legacy replacement-range arguments.")
        if args.use_pred_first:
            parser.error("--N mode uses trained adapters, not saved prediction tensors.")
        try:
            resolve_inference_schedule(args.skip_interval, args.sample_steps, args.replace_steps)
        except ValueError as error:
            parser.error(str(error))
    if args.lora_rank is None:
        args.lora_rank = 32 if args.skip_interval is not None else 4
    if args.lora_alpha is None:
        args.lora_alpha = 64.0 if args.skip_interval is not None else 8.0
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

        schedule = resolve_inference_schedule(
            getattr(args, "skip_interval", None), args.sample_steps, args.replace_steps,
        )
        if schedule is not None:
            validate_interval_adapters(
                args.lora_dir, schedule, "wan", rank=args.lora_rank, alpha=args.lora_alpha,
            )
            replace_steps = set(schedule.replace_steps)
            logging.info(f"N={schedule.interval}; full steps={schedule.full_steps}; calibrated steps={schedule.replace_steps}")
        elif args.replace_steps:
            replace_steps = set(args.replace_steps)
        else:
            replace_steps = _make_step_set(
                10 if args.replace_start is None else args.replace_start,
                38 if args.replace_end is None else args.replace_end,
                2 if args.replace_stride is None else args.replace_stride,
            )
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
            
        wan_t2v.__class__.generate = t2v_generate
        wan_t2v.model.__class__.enable_teacache = True
        wan_t2v.model.__class__.forward = step_forward
        wan_t2v.model.__class__.cnt = 0
        wan_t2v.model.__class__.layer_index = 0
        wan_t2v.model.__class__.num_steps = args.sample_steps*2
        # === LoRA & Pred 运行配置（可由命令行覆盖） ===
        # 需要在 “类” 上挂默认值（forward 是绑定到 class 的）
        wan_t2v.model.__class__.target_block_idx = 29
        wan_t2v.model.__class__.skip_until = 28
        wan_t2v.model.__class__.replace_steps = replace_steps
        wan_t2v.model._learnibridge_schedule = schedule
        wan_t2v.model.__class__.use_pred_first = args.use_pred_first

        wan_t2v.model.__class__.pred_dir = args.pred_dir
        wan_t2v.model.__class__.lora_dir = args.lora_dir
        wan_t2v.model.__class__.lora_rank = args.lora_rank
        wan_t2v.model.__class__.lora_alpha = args.lora_alpha


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
                args.save_file = f"vision/on_3.56x/on_{args.task}_{args.size}_{args.ulysses_size}_{args.ring_size}_{formatted_prompt}_{formatted_time}{suffix}"

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

def generate_from_txt(args):
    rank = int(os.getenv("RANK", 0))
    _init_logging(rank)  # ✅ 提前初始化日志

    # rank 0 读 txt
    if rank == 0:
        logging.info(f"Loading prompts from: {args.prompt_txt}")
        prompts = load_prompts_from_txt(args.prompt_txt)
        logging.info(f"Loaded {len(prompts)} prompts.")
        if len(prompts) == 0:
            raise ValueError("prompt_txt is empty")
    else:
        prompts = None

    # 分布式广播 prompts
    if dist.is_initialized():
        obj_list = [prompts]
        dist.broadcast_object_list(obj_list, src=0)
        prompts = obj_list[0]

    # 逐条生成
    for idx, prompt in enumerate(prompts):
        if rank == 0:
            logging.info("=" * 60)
            logging.info(f"[{idx+1}/{len(prompts)}] Prompt:")
            logging.info(prompt)

        args.prompt = prompt

        # 每条 prompt 用不同 seed（可选）
        if args.base_seed is not None:
            args.base_seed = args.base_seed + idx

        # 防覆盖
        args.save_file = None

        generate(args)

        if dist.is_initialized():
            dist.barrier()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# if __name__ == "__main__":
#     args = _parse_args()
#     generate(args)
if __name__ == "__main__":
    args = _parse_args()

    if args.prompt_txt is not None:
        generate_from_txt(args)
    else:
        generate(args)

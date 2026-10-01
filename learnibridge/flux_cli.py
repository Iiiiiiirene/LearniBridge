import argparse
from pathlib import Path


def parser_for(description, output_name, train_prompts=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model-path", "--ckpt-dir", dest="model_path", required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--prompt")
    source.add_argument("--prompt-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output") / output_name)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--max-prompts", type=int, default=5 if train_prompts else 1)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.set_defaults(default_prompts="train.txt" if train_prompts else "test.txt")
    return parser


def prompts_for(args):
    if min(args.num_steps, args.max_prompts, args.height, args.width) < 1:
        raise ValueError("Steps, prompt count, and image dimensions must be positive.")
    if args.height % 16 or args.width % 16:
        raise ValueError("Image dimensions must be divisible by 16.")
    if args.prompt is not None:
        prompts = [args.prompt.strip()]
    else:
        prompt_file = args.prompt_file or Path(__file__).resolve().parents[1] / "flux/prompts" / args.default_prompts
        prompts = [line.strip() for line in prompt_file.read_text().splitlines() if line.strip()]
    if not prompts or not all(prompts):
        raise ValueError("No nonempty prompts found.")
    return prompts[:args.max_prompts]


def load_pipeline(args):
    import torch
    from diffusers import DiffusionPipeline

    if not (Path(args.model_path) / "model_index.json").is_file():
        raise FileNotFoundError("Supply a local FLUX pipeline directory containing model_index.json; no automatic download.")
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.precision]
    pipeline = DiffusionPipeline.from_pretrained(args.model_path, torch_dtype=dtype, local_files_only=True)
    return pipeline.to(args.device)


def generate_image(pipeline, args, prompt):
    import torch

    return pipeline(
        prompt, height=args.height, width=args.width, num_inference_steps=args.num_steps,
        generator=torch.Generator("cpu").manual_seed(args.seed)
    ).images[0]

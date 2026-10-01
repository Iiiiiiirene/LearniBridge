import json

from learnibridge.flux_cli import generate_image, load_pipeline, parser_for, prompts_for
from learnibridge.flux_runtime import FeatureRecorder


def main():
    parser = parser_for("Extract full-compute final-block training pairs.", "features", train_prompts=True)
    args = parser.parse_args()
    prompts = prompts_for(args)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Use an empty feature output directory: {args.output_dir}")
    pipeline = load_pipeline(args)
    for index, prompt in enumerate(prompts):
        directory = args.output_dir / str(index)
        recorder = FeatureRecorder(pipeline.transformer, directory)
        try:
            image = generate_image(pipeline, args, prompt)
            image.save(directory / "reference.png")
            (directory / "metadata.json").write_text(json.dumps({
                "prompt": prompt, "seed": args.seed, "steps": args.num_steps,
                "height": args.height, "width": args.width, "block_idx": recorder.block_idx,
                "model_path": args.model_path, "precision": args.precision,
            }, indent=2))
        finally:
            recorder.close()
        print(f"Saved prompt {index}: {directory}", flush=True)


if __name__ == "__main__":
    main()

from learnibridge.flux_cli import generate_image, load_pipeline, parser_for, prompts_for


def main():
    args = parser_for("Generate the unmodified FLUX baseline.", "baseline").parse_args()
    prompts = prompts_for(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destinations = [args.output_dir / f"{index:04d}.png" for index in range(len(prompts))]
    if any(destination.exists() for destination in destinations):
        raise FileExistsError("Refusing to overwrite baseline images; select a new output directory.")
    pipeline = load_pipeline(args)
    for prompt, destination in zip(prompts, destinations):
        generate_image(pipeline, args, prompt).save(destination)
        print(f"Saved {destination}", flush=True)


if __name__ == "__main__":
    main()

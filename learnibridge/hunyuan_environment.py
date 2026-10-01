import sys


def isolate_float_checkpoint_imports():
    """Exclude optional quantizers from this process when using original non-quantized checkpoints."""
    names = ("awq", "auto_gptq", "gptqmodel", "bitsandbytes", "aqlm", "eetq", "hqq")
    loaded = [name for name in names if sys.modules.get(name) is not None]
    if loaded:
        raise RuntimeError(f"Apply float-checkpoint isolation before importing optional quantizers: {loaded}")
    for name in names:
        sys.modules[name] = None
    return list(names)

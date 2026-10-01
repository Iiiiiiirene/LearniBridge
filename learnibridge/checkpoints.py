from .lora import LoRALinear, inject_lora, load_adapter_state, set_lora_enabled


def load_video_adapter(block, payload):
    required = {"state_dict", "rank", "alpha", "target_modules", "steps", "offset", "base_frozen"}
    if not isinstance(payload, dict) or required - payload.keys():
        raise ValueError("Checkpoint lacks calibration metadata. Retrain with the canonical video trainer.")
    if payload["base_frozen"] is not True or payload["offset"] < 1:
        raise ValueError("Checkpoint must certify frozen base weights and a positive cache offset.")
    existing = [module for module in block.modules() if isinstance(module, LoRALinear)]
    if existing:
        if any(module.rank != payload["rank"] or module.alpha != payload["alpha"] for module in existing):
            raise ValueError("Cannot switch adapter rank/alpha within a model instance.")
    else:
        inject_lora(block, payload["target_modules"], rank=payload["rank"], alpha=payload["alpha"],
                    dropout=payload.get("dropout", 0.0))
    state = {
        name: value for name, value in payload["state_dict"].items()
        if name.rsplit(".", 1)[-1] in {"lora_a", "lora_b"}
    }
    load_adapter_state(block, state)
    set_lora_enabled(block, True)
    block.eval()
    block._learnibridge_checkpoint = {
        key: value for key, value in payload.items() if key != "state_dict"
    }

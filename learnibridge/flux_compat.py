import inspect

import torch


def forward_single_block(block, hidden_states, temb, image_rotary_emb=None,
                         joint_attention_kwargs=None, text_length=None):
    arguments = {
        "hidden_states": hidden_states,
        "temb": temb,
        "image_rotary_emb": image_rotary_emb,
        "joint_attention_kwargs": joint_attention_kwargs,
    }
    if "encoder_hidden_states" in inspect.signature(block.forward).parameters:
        if text_length is None or not 0 <= text_length < hidden_states.shape[1]:
            raise ValueError("Modern FLUX blocks require text_length metadata; re-extract legacy features.")
        arguments["encoder_hidden_states"] = hidden_states[:, :text_length]
        arguments["hidden_states"] = hidden_states[:, text_length:]
        encoder_output, image_output = block(**arguments)
        return torch.cat((encoder_output, image_output), dim=1)
    return block(**arguments)

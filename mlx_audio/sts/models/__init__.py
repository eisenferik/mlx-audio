# Copyright (c) 2025 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import importlib

# Submodules are imported on first access, so that using one model does not
# require the dependencies of every other model.
_LAZY_EXPORTS = {
    "DeepFilterNet2Config": "deepfilternet",
    "DeepFilterNet3Config": "deepfilternet",
    "DeepFilterNetConfig": "deepfilternet",
    "DeepFilterNetModel": "deepfilternet",
    "DeepFilterNetStreamer": "deepfilternet",
    "DeepFilterNetStreamingConfig": "deepfilternet",
    "ChatState": "lfm_audio",
    "GenerationConfig": "lfm_audio",
    "LFM2AudioConfig": "lfm_audio",
    "LFM2AudioModel": "lfm_audio",
    "LFM2AudioProcessor": "lfm_audio",
    "LFMModality": "lfm_audio",
    "MelRoFormer": "mel_roformer",
    "MelRoFormerConfig": "mel_roformer",
    "MelRoFormerResult": "mel_roformer",
    "MoshiConfig": "moshi",
    "MoshiSTSModel": "moshi",
    "MossFormer2SE": "mossformer2_se",
    "MossFormer2SEConfig": "mossformer2_se",
    "MossFormer2SEModel": "mossformer2_se",
    "Batch": "sam_audio",
    "SAMAudio": "sam_audio",
    "SAMAudioConfig": "sam_audio",
    "SAMAudioProcessor": "sam_audio",
    "SeparationResult": "sam_audio",
    "save_audio": "sam_audio",
}

__all__ = [
    "SAMAudio",
    "SAMAudioProcessor",
    "SeparationResult",
    "Batch",
    "save_audio",
    "SAMAudioConfig",
    # DeepFilterNet
    "DeepFilterNetModel",
    "DeepFilterNetConfig",
    "DeepFilterNet2Config",
    "DeepFilterNet3Config",
    "DeepFilterNetStreamer",
    "DeepFilterNetStreamingConfig",
    # MossFormer2 SE
    "MossFormer2SE",
    "MossFormer2SEConfig",
    "MossFormer2SEModel",
    # LFM2.5-Audio
    "LFM2AudioModel",
    "LFM2AudioProcessor",
    "LFM2AudioConfig",
    "LFMModality",
    "ChatState",
    "GenerationConfig",
    # Moshi
    "MoshiConfig",
    "MoshiSTSModel",
    # Mel-Band-RoFormer
    "MelRoFormer",
    "MelRoFormerConfig",
    "MelRoFormerResult",
]


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module = importlib.import_module(f"{__name__}.{_LAZY_EXPORTS[name]}")
        value = getattr(module, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

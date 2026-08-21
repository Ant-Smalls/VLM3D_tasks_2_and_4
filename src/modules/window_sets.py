# window_sets.py
# purpose: named 3-channel HU layouts and checkpoint window_set recovery
# author: Anthony Smaldore

import torch

WINDOW_SET_DEFAULT = "default"
WINDOW_SET_LUNG_MEDIASTINUM_BONE = "lung_mediastinum_bone"
KNOWN_WINDOW_SETS = (WINDOW_SET_DEFAULT, WINDOW_SET_LUNG_MEDIASTINUM_BONE)

# Channel 0/1 HU ranges are shared - channel 2 differs.
FLUID_WINDOW_HU = (-40, 400)
CALCIUM_WINDOW_HU = (300, 1500)
LUNG_WINDOW_HU = (-1000, 400)
DEFAULT_GENERAL_PERCENTILES = (0.5, 99.5)


def resolve_window_set(name):
    """
    Validates a window-set name and returns it if it is known.
    
    Args:
        name (str): The name of the window set ("default" or "lung_mediastinum_bone").
    
    Returns:
        str: The same window-set name if it is known.
    
    Raises:
        ValueError: If the name is not a known window set.
    """
    if name not in KNOWN_WINDOW_SETS:
        known = ", ".join(KNOWN_WINDOW_SETS)
        raise ValueError(
            f"Unknown window_set {name!r}. Known names: {known}"
        )
    return name


def window_set_from_hparams(hparams, override=None):
    """
    Resolves the window-set name from an optional override, else from checkpoint hyperparameters.
    
    Args:
        hparams (dict or object): Checkpoint hyperparameters. May be a dict or an object with attributes.
        override (str, optional): If set, this window-set name is used instead of the checkpoint field.
    
    Returns:
        str: A known window-set name. Missing or absent window_set maps to "default".
    """
    if override is not None:
        return resolve_window_set(override)
    if hparams is None:
        return WINDOW_SET_DEFAULT
    if isinstance(hparams, dict):
        name = hparams.get("window_set")
    else:
        name = getattr(hparams, "window_set", None)
    if name is None:
        return WINDOW_SET_DEFAULT
    return resolve_window_set(name)


def window_set_from_checkpoint(checkpoint_path, override=None):
    """
    Reads the window-set name from a Lightning checkpoint without building the model.
    
    Args:
        checkpoint_path (str): Path to a .ckpt file.
        override (str, optional): If set, this window-set name is used instead of the checkpoint field.
    
    Returns:
        str: A known window-set name. Missing window_set maps to "default".
    """
    if override is not None:
        return resolve_window_set(override)
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    hparams = payload.get("hyper_parameters") if isinstance(payload, dict) else None
    return window_set_from_hparams(hparams)

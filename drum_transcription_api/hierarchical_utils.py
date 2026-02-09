"""Utility functions for converting hierarchical model predictions to flat format."""

import numpy as np
from typing import Dict, Optional


def convert_hierarchical_to_flat(
    hierarchical_preds: Dict[str, np.ndarray],
    thresholds: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Convert hierarchical predictions to flat 12-class format.

    This converts the 5-branch hierarchical model output into the standard
    12-class flat format used by the API for onset detection and MIDI creation.

    Args:
        hierarchical_preds: Dictionary of predictions from hierarchical model
            Expected structure:
            {
                'kick': (batch, time, 1),                    # Binary (post-sigmoid)
                'snare': (batch, time, 3),                   # 3-class (post-softmax)
                'tom': {
                    'primary': (batch, time, 2),             # Binary (post-softmax)
                    'variation': (batch, time, 3)            # 3-class (post-softmax)
                },
                'cymbal': {
                    'primary': (batch, time, 3),             # 3-class (post-softmax)
                    'hihat_variation': (batch, time, 2),     # Binary (post-softmax)
                    'ride_variation': (batch, time, 2)       # Binary (post-softmax)
                },
                'crash': (batch, time, 1)                    # Binary (post-sigmoid)
            }
        thresholds: Optional dict of thresholds per branch (default: 0.5 for all)

    Returns:
        Flat predictions of shape (batch, time, 12) with classes:
        [0: kick, 1: snare_head, 2: snare_rim, 3: side_stick, 4: hihat_pedal,
         5: hihat_closed, 6: hihat_open, 7: floor_tom, 8: high_mid_tom,
         9: ride, 10: ride_bell, 11: crash]

    Notes:
        - side_stick (class 3) is merged with snare_rim in hierarchical model, set to 0
        - hihat_pedal (class 4) is not predicted by hierarchical model, set to 0
        - All probabilities should already be in [0, 1] range (post-sigmoid/softmax)
    """
    if thresholds is None:
        thresholds = {
            'kick': 0.5,
            'snare': 0.5,
            'tom': 0.5,
            'cymbal': 0.5,
            'crash': 0.5,
        }

    # Get batch and time dimensions
    kick_pred = hierarchical_preds['kick']
    batch, time = kick_pred.shape[:2]

    # Initialize flat predictions (12 classes)
    flat_preds = np.zeros((batch, time, 12), dtype=np.float32)

    # 1. Kick (class 0)
    # Expects probabilities from sigmoid, shape (batch, time, 1)
    flat_preds[:, :, 0] = (kick_pred[:, :, 0] > thresholds['kick']).astype(np.float32)

    # 2-3. Snare (classes 1-2: head, rim)
    # Expects probabilities from softmax, shape (batch, time, 3)
    # Classes: [0: none, 1: head, 2: rim]
    snare_probs = hierarchical_preds['snare']
    flat_preds[:, :, 1] = (snare_probs[:, :, 1] > thresholds['snare']).astype(np.float32)  # head
    flat_preds[:, :, 2] = (snare_probs[:, :, 2] > thresholds['snare']).astype(np.float32)  # rim

    # Note: side_stick (class 3) is merged with rim in hierarchical model
    # We leave class 3 as 0 to avoid duplicates
    flat_preds[:, :, 3] = 0

    # 4. Hihat pedal (class 4) - Not predicted by hierarchical model
    flat_preds[:, :, 4] = 0

    # 5-6. Hihat (classes 5-6: closed, open)
    # Primary cymbal branch: [0: none, 1: hihat, 2: ride]
    cymbal_primary_probs = hierarchical_preds['cymbal']['primary']
    hihat_detected = cymbal_primary_probs[:, :, 1] > thresholds['cymbal']  # hihat

    # Hihat variation: [0: closed, 1: open]
    hihat_var_probs = hierarchical_preds['cymbal']['hihat_variation']
    is_open = hihat_var_probs[:, :, 1] > 0.5  # open

    flat_preds[:, :, 5] = (hihat_detected & ~is_open).astype(np.float32)  # closed
    flat_preds[:, :, 6] = (hihat_detected & is_open).astype(np.float32)   # open

    # 7-8. Toms (classes 7-8: floor, high_mid)
    # Primary tom branch: [0: no_tom, 1: tom]
    tom_primary_probs = hierarchical_preds['tom']['primary']
    tom_detected = tom_primary_probs[:, :, 1] > thresholds['tom']

    # Tom variation: [0: floor, 1: high, 2: mid]
    # We merge high and mid into single class (high_mid_tom)
    tom_var_probs = hierarchical_preds['tom']['variation']
    is_floor = tom_var_probs[:, :, 0] > 0.5

    flat_preds[:, :, 7] = (tom_detected & is_floor).astype(np.float32)   # floor
    flat_preds[:, :, 8] = (tom_detected & ~is_floor).astype(np.float32)  # high_mid

    # 9-10. Ride (classes 9-10: ride, ride_bell)
    # Primary cymbal branch: [0: none, 1: hihat, 2: ride]
    ride_detected = cymbal_primary_probs[:, :, 2] > thresholds['cymbal']  # ride

    # Ride variation: [0: body, 1: bell]
    ride_var_probs = hierarchical_preds['cymbal']['ride_variation']
    is_bell = ride_var_probs[:, :, 1] > 0.5

    flat_preds[:, :, 9] = (ride_detected & ~is_bell).astype(np.float32)  # ride body
    flat_preds[:, :, 10] = (ride_detected & is_bell).astype(np.float32)  # ride bell

    # 11. Crash (class 11)
    # Expects probabilities from sigmoid, shape (batch, time, 1)
    flat_preds[:, :, 11] = (hierarchical_preds['crash'][:, :, 0] > thresholds['crash']).astype(np.float32)

    return flat_preds


def apply_activation_to_hierarchical(
    hierarchical_logits: Dict,
    device: str = "cpu"
) -> Dict[str, np.ndarray]:
    """
    Apply appropriate activations to hierarchical model logits.

    This applies sigmoid to binary branches and softmax to multi-class branches,
    then converts to numpy arrays for use with convert_hierarchical_to_flat().

    Args:
        hierarchical_logits: Raw model output (dictionary of tensors/arrays)
        device: Device where tensors are located (for torch conversion)

    Returns:
        Dictionary of probability arrays ready for conversion
    """
    import torch

    result = {}

    # Helper to convert tensor to numpy
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return x

    # Kick: Binary (sigmoid)
    if isinstance(hierarchical_logits['kick'], torch.Tensor):
        result['kick'] = torch.sigmoid(hierarchical_logits['kick']).cpu().numpy()
    else:
        # Assume already numpy
        from scipy.special import expit
        result['kick'] = expit(hierarchical_logits['kick'])

    # Snare: 3-class (softmax)
    if isinstance(hierarchical_logits['snare'], torch.Tensor):
        result['snare'] = torch.softmax(hierarchical_logits['snare'], dim=-1).cpu().numpy()
    else:
        from scipy.special import softmax
        result['snare'] = softmax(hierarchical_logits['snare'], axis=-1)

    # Tom: Nested dict with softmax on both
    result['tom'] = {}
    if isinstance(hierarchical_logits['tom']['primary'], torch.Tensor):
        result['tom']['primary'] = torch.softmax(hierarchical_logits['tom']['primary'], dim=-1).cpu().numpy()
        result['tom']['variation'] = torch.softmax(hierarchical_logits['tom']['variation'], dim=-1).cpu().numpy()
    else:
        from scipy.special import softmax
        result['tom']['primary'] = softmax(hierarchical_logits['tom']['primary'], axis=-1)
        result['tom']['variation'] = softmax(hierarchical_logits['tom']['variation'], axis=-1)

    # Cymbal: Nested dict with softmax on all
    result['cymbal'] = {}
    if isinstance(hierarchical_logits['cymbal']['primary'], torch.Tensor):
        result['cymbal']['primary'] = torch.softmax(hierarchical_logits['cymbal']['primary'], dim=-1).cpu().numpy()
        result['cymbal']['hihat_variation'] = torch.softmax(hierarchical_logits['cymbal']['hihat_variation'], dim=-1).cpu().numpy()
        result['cymbal']['ride_variation'] = torch.softmax(hierarchical_logits['cymbal']['ride_variation'], dim=-1).cpu().numpy()
    else:
        from scipy.special import softmax
        result['cymbal']['primary'] = softmax(hierarchical_logits['cymbal']['primary'], axis=-1)
        result['cymbal']['hihat_variation'] = softmax(hierarchical_logits['cymbal']['hihat_variation'], axis=-1)
        result['cymbal']['ride_variation'] = softmax(hierarchical_logits['cymbal']['ride_variation'], axis=-1)

    # Crash: Binary (sigmoid)
    if isinstance(hierarchical_logits['crash'], torch.Tensor):
        result['crash'] = torch.sigmoid(hierarchical_logits['crash']).cpu().numpy()
    else:
        from scipy.special import expit
        result['crash'] = expit(hierarchical_logits['crash'])

    return result

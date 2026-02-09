"""Match predicted onsets to ground truth within tolerance window."""

import numpy as np
from typing import List, Tuple, Dict, Set


class OnsetMatcher:
    """Match predicted onsets to ground truth onsets."""

    def __init__(self, tolerance: float = 0.05):
        """
        Initialize matcher.

        Args:
            tolerance: Time tolerance for matching (seconds). Default: 50ms
        """
        self.tolerance = tolerance

    def match_onsets(
        self,
        ground_truth: List[Tuple[float, str]],
        predictions: List[Tuple[float, str]]
    ) -> Dict:
        """
        Match predictions to ground truth within tolerance window.

        Args:
            ground_truth: List of (time, drum_name) from MIDI
            predictions: List of (time, drum_name) from model

        Returns:
            Dictionary with matching results:
            {
                'matches': [(gt_time, pred_time, drum_name, error), ...],
                'false_positives': [(pred_time, drum_name), ...],
                'false_negatives': [(gt_time, drum_name), ...],
                'per_class': {drum_name: {...}, ...}
            }
        """
        # Organize by drum class
        gt_by_class = self._group_by_class(ground_truth)
        pred_by_class = self._group_by_class(predictions)

        all_classes = set(gt_by_class.keys()) | set(pred_by_class.keys())

        matches = []
        false_positives = []
        false_negatives = []
        per_class_results = {}

        for drum_class in all_classes:
            gt_times = sorted(gt_by_class.get(drum_class, []))
            pred_times = sorted(pred_by_class.get(drum_class, []))

            class_matches, class_fp, class_fn = self._match_class(
                gt_times, pred_times, drum_class
            )

            matches.extend(class_matches)
            false_positives.extend(class_fp)
            false_negatives.extend(class_fn)

            # Calculate per-class metrics
            tp = len(class_matches)
            fp = len(class_fp)
            fn = len(class_fn)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            timing_errors = [error for _, _, _, error in class_matches]
            mean_error = np.mean(timing_errors) if timing_errors else 0.0
            std_error = np.std(timing_errors) if timing_errors else 0.0

            per_class_results[drum_class] = {
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'mean_timing_error': mean_error,
                'std_timing_error': std_error,
            }

        return {
            'matches': matches,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'per_class': per_class_results,
            'tolerance': self.tolerance,
        }

    def _group_by_class(self, onsets: List[Tuple[float, str]]) -> Dict[str, List[float]]:
        """Group onsets by drum class."""
        grouped = {}
        for time, drum in onsets:
            if drum not in grouped:
                grouped[drum] = []
            grouped[drum].append(time)
        return grouped

    def _match_class(
        self,
        gt_times: List[float],
        pred_times: List[float],
        drum_class: str
    ) -> Tuple[List, List, List]:
        """
        Match predictions to ground truth for a single drum class.

        Uses greedy matching: each GT onset matched to closest prediction within tolerance.

        Returns:
            (matches, false_positives, false_negatives)
        """
        matches = []
        matched_preds = set()
        matched_gts = set()

        # For each ground truth onset, find closest prediction within tolerance
        for gt_idx, gt_time in enumerate(gt_times):
            best_pred_idx = None
            best_error = float('inf')

            for pred_idx, pred_time in enumerate(pred_times):
                if pred_idx in matched_preds:
                    continue

                error = abs(pred_time - gt_time)
                if error <= self.tolerance and error < best_error:
                    best_error = error
                    best_pred_idx = pred_idx

            if best_pred_idx is not None:
                matches.append((
                    gt_time,
                    pred_times[best_pred_idx],
                    drum_class,
                    best_error
                ))
                matched_preds.add(best_pred_idx)
                matched_gts.add(gt_idx)

        # False positives: predictions not matched to any ground truth
        false_positives = [
            (pred_times[i], drum_class)
            for i in range(len(pred_times))
            if i not in matched_preds
        ]

        # False negatives: ground truth not matched to any prediction
        false_negatives = [
            (gt_times[i], drum_class)
            for i in range(len(gt_times))
            if i not in matched_gts
        ]

        return matches, false_positives, false_negatives

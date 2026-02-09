"""Calculate evaluation metrics from matching results."""

import numpy as np
from typing import Dict, List


class MetricsCalculator:
    """Calculate and aggregate evaluation metrics."""

    @staticmethod
    def calculate_summary(match_results: Dict) -> Dict:
        """
        Calculate summary metrics across all classes.

        Args:
            match_results: Output from OnsetMatcher.match_onsets()

        Returns:
            Summary metrics dictionary
        """
        per_class = match_results['per_class']

        # Aggregate metrics
        total_tp = sum(m['true_positives'] for m in per_class.values())
        total_fp = sum(m['false_positives'] for m in per_class.values())
        total_fn = sum(m['false_negatives'] for m in per_class.values())

        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) \
            if (overall_precision + overall_recall) > 0 else 0.0

        # Timing errors (only for matched onsets)
        all_errors = [error for _, _, _, error in match_results['matches']]
        mean_timing_error = np.mean(all_errors) if all_errors else 0.0
        median_timing_error = np.median(all_errors) if all_errors else 0.0
        std_timing_error = np.std(all_errors) if all_errors else 0.0

        return {
            'overall': {
                'precision': overall_precision,
                'recall': overall_recall,
                'f1_score': overall_f1,
                'true_positives': total_tp,
                'false_positives': total_fp,
                'false_negatives': total_fn,
            },
            'timing': {
                'mean_error': mean_timing_error,
                'median_error': median_timing_error,
                'std_error': std_timing_error,
                'tolerance': match_results['tolerance'],
            },
            'per_class': per_class,
        }

    @staticmethod
    def to_dict(summary: Dict) -> Dict:
        """Convert summary metrics to dictionary for easy export."""
        rows = []
        for drum_class, metrics in summary['per_class'].items():
            rows.append({
                'drum_class': drum_class,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'true_positives': metrics['true_positives'],
                'false_positives': metrics['false_positives'],
                'false_negatives': metrics['false_negatives'],
                'mean_timing_error_ms': metrics['mean_timing_error']*1000,
            })

        # Add overall row
        overall = summary['overall']
        rows.append({
            'drum_class': 'OVERALL',
            'precision': overall['precision'],
            'recall': overall['recall'],
            'f1_score': overall['f1_score'],
            'true_positives': overall['true_positives'],
            'false_positives': overall['false_positives'],
            'false_negatives': overall['false_negatives'],
            'mean_timing_error_ms': summary['timing']['mean_error']*1000,
        })

        return {'metrics': rows}

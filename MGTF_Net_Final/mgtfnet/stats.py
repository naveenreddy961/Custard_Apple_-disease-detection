"""McNemar paired test with continuity correction and Bonferroni adjustment."""
import numpy as np
import pandas as pd
from scipy.stats import chi2

def run_mcnemar_statistical_test(y_true, y_pred_proposed, baseline_preds_dict, alpha=0.05):
    """
    Computes McNemar's paired test with continuity correction 
    and Bonferroni-adjusted p-values for multiple comparisons.
    """
    y_true = np.asarray(y_true)
    y_prop = np.asarray(y_pred_proposed)
    num_comparisons = len(baseline_preds_dict)
    results = []

    print(f"\n{'='*75}")
    print(f" McNEMAR'S STATISTICAL TEST WITH BONFERRONI CORRECTION (k={num_comparisons})")
    print(f"{'='*75}")

    for name, y_base in baseline_preds_dict.items():
        y_base = np.asarray(y_base)
        correct_prop = (y_prop == y_true)
        correct_base = (y_base == y_true)

        # Contingency table components
        b = np.sum(correct_prop & ~correct_base)  # Proposed correct, baseline wrong
        c = np.sum(~correct_prop & correct_base)  # Proposed wrong, baseline correct

        if (b + c) == 0:
            chi2_stat = 0.0
            p_value = 1.0
        else:
            # Edwards continuity correction
            chi2_stat = ((abs(b - c) - 1.0) ** 2) / (b + c)
            p_value = 1.0 - chi2.cdf(chi2_stat, df=1)

        adj_p = min(1.0, p_value * num_comparisons)
        is_sig = "Yes (p < 0.05)" if adj_p < alpha else "No"

        results.append({
            "Comparison": f"MGTF-Net vs. {name}",
            "b (Prop+/Base-)": int(b),
            "c (Prop-/Base+)": int(c),
            "Chi2 (df=1)": round(chi2_stat, 2),
            "Raw p-value": f"{p_value:.4e}",
            "Bonferroni-adj. p": f"{adj_p:.4e}",
            "Significant": is_sig
        })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    return df

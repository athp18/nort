import matplotlib.pyplot as plt
import numpy as np

def check_tracking_quality(df):
    """Print mean likelihood for each bodypart."""
    likelihoods = [c for c in df.columns if 'likelihood' in c]
    for col in likelihoods:
        print(f"{col}: {df[col].mean():.3f}")

def plot_likelihoods(df, threshold=0.6, session_name='Session', save_path=None, show=True):
    """Visualize tracking confidence over frames.

    If save_path is provided, save the figure instead of or in addition to showing.
    """
    plt.figure(figsize=(12, 6))
    for col in df.columns:
        if 'likelihood' in col:
            plt.plot(df.index, df[col], label=col.replace('_likelihood', ''))
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.title(f'{session_name} - Tracking Confidence')
    plt.xlabel('Frame')
    plt.ylabel('Likelihood')
    plt.legend()
    plt.grid(alpha=0.3)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def summarize_low_confidence(df, threshold=0.6):
    """Count low-confidence frames across all bodyparts."""
    low_conf = (df.filter(like='likelihood') <= threshold).any(axis=1)
    n_low = low_conf.sum()
    print(f"Low-confidence frames: {n_low}/{len(df)} ({100*n_low/len(df):.2f}%)")
    return n_low, low_conf

def warn_if_poor_quality(df, likelihood_threshold=0.6, max_low_confidence_pct=20.0, min_bodyparts_present=1):
    """Issue warnings if DLC data quality looks problematic.

    Triggers warnings when:
    - Percentage of frames with any bodypart below likelihood_threshold exceeds max_low_confidence_pct
    - Number of detected bodyparts is below min_bodyparts_present
    """
    import warnings

    # percentage of frames with any low confidence
    low_conf = (df.filter(like='likelihood') <= likelihood_threshold).any(axis=1)
    pct_low = 100 * low_conf.mean()
    if pct_low > max_low_confidence_pct:
        warnings.warn(
            f"High fraction of low-confidence frames: {pct_low:.2f}% exceeds {max_low_confidence_pct:.2f}% (threshold={likelihood_threshold}).",
            RuntimeWarning,
        )

    # basic structural checks
    likelihood_cols = [c for c in df.columns if c.endswith('_likelihood')]
    bodyparts_present = len(likelihood_cols)
    if bodyparts_present < min_bodyparts_present:
        warnings.warn(
            f"Few bodyparts detected: {bodyparts_present} < {min_bodyparts_present}. Check DLC export/scorer levels.",
            RuntimeWarning,
        )
    return {"pct_low_confidence": pct_low, "num_bodyparts": bodyparts_present}
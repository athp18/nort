from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pickle


def train_svm(X, y, test_size=0.2, gamma=1e-4, C=1.0, return_metrics=False):
    """Train SVM classifier for exploration detection.

    If return_metrics=True, also return a dict with metrics.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    clf = SVC(kernel="rbf", gamma=gamma, C=C, probability=True)
    clf.fit(X_tr, y_tr)

    train_acc = clf.score(X_tr, y_tr)
    val_acc = clf.score(X_val, y_val)
    y_pred = clf.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)
    report = classification_report(y_val, y_pred)

    print(f"Train acc: {train_acc*100:.1f}%, Val acc: {val_acc*100:.1f}%")
    print("\nConfusion matrix:\n", cm)
    print("\nReport:\n", report)

    if return_metrics:
        return clf, {
            "train_acc": train_acc,
            "val_acc": val_acc,
            "confusion_matrix": cm,
            "classification_report": report,
        }
    return clf


def save_model(model, path="nor_svm.pkl"):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path="nor_svm.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def analyze_nor_test(features_familiar, features_novel, model, fps=30):
    """Apply SVM to compute exploration metrics and discrimination index."""
    valid_fam = ~np.isnan(features_familiar).any(axis=1)
    valid_nov = ~np.isnan(features_novel).any(axis=1)
    pred_fam = model.predict(features_familiar[valid_fam])
    pred_nov = model.predict(features_novel[valid_nov])

    time_fam = pred_fam.sum() / fps
    time_nov = pred_nov.sum() / fps
    total = time_fam + time_nov

    DI = 100 * (time_nov - time_fam) / (total + 1e-8)
    pct_nov = 100 * time_nov / (total + 1e-8)

    print(
        f"Familiar: {time_fam:.2f}s | Novel: {time_nov:.2f}s | DI={DI:.2f}% | Novel%={pct_nov:.2f}%"
    )
    return {
        "DI": DI,
        "pct_novel": pct_nov,
        "time_familiar": time_fam,
        "time_novel": time_nov,
    }


def check_familiarization(features_familiar, svm_classifier, fps=30):
    """
    Check if mouse explored >20s in familiarization phase.
    This is the exclusion criteria from the protocol.
    """
    print("\n" + "=" * 50)
    print("Checking familiarization phase...")
    print("=" * 50)

    # Predict exploration
    valid = ~np.isnan(features_familiar).any(axis=1)
    exploring = np.zeros(len(features_familiar), dtype=int)
    exploring[valid] = svm_classifier.predict(features_familiar[valid])

    # Calculate time
    frames_exploring = int(np.sum(exploring))
    time_exploring = frames_exploring / fps

    print(f"\nFamiliarization exploration time: {time_exploring:.2f} seconds")

    if not time_exploring > 20:
        print(f"Mouse explored for {time_exploring:.2f}s < {20}s")
        print(f"Consider excluding this mouse from analysis")
        exclude = True

    return {
        "exploration_time_sec": float(time_exploring),
        "frames_exploring": frames_exploring,
        "exclude": exclude,
    }

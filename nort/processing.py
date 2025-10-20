import numpy as np
import cv2
import matplotlib.pyplot as plt


def filter_by_likelihood(df, threshold=0.6):
    """Replace low-confidence x,y values with NaN for all bodyparts."""
    for col in df.columns:
        if "likelihood" in col:
            prefix = col.replace("_likelihood", "")
            mask = df[col] < threshold
            df.loc[mask, f"{prefix}_x"] = np.nan
            df.loc[mask, f"{prefix}_y"] = np.nan
    return df


def calculate_velocity(x, y):
    """Compute frame-to-frame velocity."""
    dx, dy = np.diff(x), np.diff(y)
    return np.sqrt(dx**2 + dy**2)


def compute_median_frame(video_path, n_samples=300):
    """Compute median frame to remove moving subject."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, n_samples, dtype=int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        frames.append(gray)
    cap.release()
    if not frames:
        raise ValueError(f"No frames could be read from {video_path}")
    return np.median(np.array(frames), axis=0).astype(np.uint8)


def find_largest_objects(median_frame, percentile=99, n=1):
    """Find n largest bright blobs in the frame."""
    threshold = np.percentile(median_frame, percentile)
    binary = (median_frame >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:n]
    mask = np.zeros_like(median_frame)
    cv2.drawContours(mask, contours, -1, 255, -1)
    return mask


def get_object_vertices(object_mask):
    """Extract bounding boxes and centroids."""
    contours, _ = cv2.findContours(
        object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    objects = []
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        vertices = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        M = cv2.moments(c)
        cx = int(M["m10"] / M["m00"]) if M["m00"] else x + w // 2
        cy = int(M["m01"] / M["m00"]) if M["m00"] else y + h // 2
        objects.append(
            {"id": i, "vertices": vertices, "centroid": (cx, cy), "bbox": (x, y, w, h)}
        )
    return objects

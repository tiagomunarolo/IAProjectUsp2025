# decision_tree_cython.pyx
import numpy as np
cimport numpy as np
from libc.math cimport INFINITY

def find_best_split_cython(
    np.ndarray[np.float64_t, ndim=2] x,
    np.ndarray[np.int32_t, ndim=1] y,
    int feature_idx,
    int min_samples_split,
    object criterion
):
    cdef int n_samples = x.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] feature = x[:, feature_idx]
    thresholds = np.unique(feature)

    if thresholds.shape[0] > 100:
        thresholds = np.linspace(feature.min(), feature.max(), num=100, endpoint=False)

    cdef double best_score = INFINITY
    cdef double best_thresh = -1.0

    for threshold in thresholds:
        mask = feature <= threshold
        y_left, x_left = y[mask], x[mask]
        y_right, x_right = y[~mask], x[~mask]

        if y_left.shape[0] < min_samples_split or y_right.shape[0] < min_samples_split:
            continue

        score = criterion(
            y_left=y_left,
            y_right=y_right,
            x_left=x_left,
            x_right=x_right
        )

        if score < best_score:
            best_score = score
            best_thresh = threshold

    return best_score, best_thresh, feature_idx
import numpy as np


def chest_roi(gray, cy=0.52, ry=0.40, cx=0.50, rx=0.42, thr=0.06):
    """Boolean mask of the thorax region (display only, the model never sees it)."""
    h, w = gray.shape
    rows = np.where(gray.mean(axis=1) > thr)[0]
    cols = np.where(gray.mean(axis=0) > thr)[0]
    if len(rows) < 0.3 * h or len(cols) < 0.3 * w:
        y0, y1, x0, x1 = 0, h, 0, w
    else:
        y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    bh, bw = y1 - y0, x1 - x0
    cy0, cx0 = y0 + cy * bh, x0 + cx * bw
    ry0, rx0 = ry * bh, rx * bw
    yy, xx = np.mgrid[0:h, 0:w]
    return (((yy - cy0) / ry0) ** 2 + ((xx - cx0) / rx0) ** 2) <= 1.0

"""Printed-photograph boundary detection with multiple fallback strategies.

Strategies (run in order, all candidates scored together):
  1. contour quadrilateral search over Canny edges
  2. largest rotated rectangle (minAreaRect) over significant contours
  3. Hough line intersections for photos with weak/partial borders
"""
import cv2
import numpy as np

from .quadrilateral_detector import (
    is_valid_quad, order_points, quad_area, quad_aspect,
)

PASSPORT_ASPECTS = (0.65, 1.5)  # plausible width/height range for a photo on a form


def _edge_strength(gray, pts):
    """Mean gradient magnitude sampled just inside/outside each quad edge."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    p = order_points(pts)
    vals = []
    h, w = gray.shape
    for i in range(4):
        a, b = p[i], p[(i + 1) % 4]
        n = max(2, int(np.linalg.norm(b - a) / 8))
        for t in np.linspace(0.05, 0.95, n):
            pt = a + (b - a) * t
            for off in (-2.0, 2.0):
                nx, ny = -(b[1] - a[1]), (b[0] - a[0])
                norm = np.hypot(nx, ny) or 1.0
                x = int(np.clip(pt[0] + nx / norm * off, 0, w - 1))
                y = int(np.clip(pt[1] + ny / norm * off, 0, h - 1))
                vals.append(mag[y, x])
    return float(np.mean(vals)) if vals else 0.0


def _interior_texture(gray, pts):
    """Std-dev of interior pixels; a real photo has more texture than paper margins."""
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillPoly(mask, [np.int32(order_points(pts))], 255)
    inner = cv2.erode(mask, np.ones((9, 9), np.uint8))
    if inner.sum() < 100:
        return 0.0
    return float(gray[inner > 0].std())


def _score_candidate(gray, pts, img_area, strategy):
    ok, _ = is_valid_quad(pts, gray.shape)
    if not ok:
        return None
    aspect = quad_aspect(pts)
    if not (PASSPORT_ASPECTS[0] <= aspect <= PASSPORT_ASPECTS[1]):
        return None
    area_frac = quad_area(pts) / img_area
    edge = _edge_strength(gray, pts)
    tex = _interior_texture(gray, pts)
    # photo interiors are usually textured; blank paper quads score lower
    texture_term = min(tex / 40.0, 1.0)
    score = (0.35 * min(edge / 60.0, 1.0)
             + 0.25 * texture_term
             + 0.25 * min(area_frac / 0.25, 1.0)
             + 0.15 * (1.0 if strategy != 'hough' else 0.8))
    return {
        'points': order_points(pts).tolist(),
        'confidence': round(float(min(score, 1.0)), 3),
        'strategy': strategy,
        'area_fraction': round(float(area_frac), 4),
        'aspect': round(float(aspect), 3),
    }


def _strategy_contours(gray, edges):
    cands = []
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img_area = gray.shape[0] * gray.shape[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.02 * img_area or area > 0.9 * img_area:
            continue
        peri = cv2.arcLength(c, True)
        for eps in (0.02, 0.04):
            p = cv2.approxPolyDP(c, eps * peri, True)
            if len(p) == 4:
                scored = _score_candidate(gray, p.reshape(4, 2), img_area, 'contour')
                if scored:
                    cands.append(scored)
        # fallback: rotated rect of large contours (handles rounded/soft corners)
        if area > 0.05 * img_area:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            scored = _score_candidate(gray, box, img_area, 'minarearect')
            if scored:
                cands.append(scored)
    return cands


def _strategy_hough(gray, edges):
    """Intersect dominant near-horizontal/vertical edge lines into quads."""
    h, w = gray.shape
    img_area = h * w
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=int(min(h, w) * 0.12), maxLineGap=12)
    if lines is None:
        return []
    cands = []
    lines = np.asarray(lines)
    segs = lines.reshape(-1, 4) if lines.ndim == 2 else lines[:, 0, :].reshape(-1, 4)
    # cluster segments by orientation
    horiz, vert = [], []
    for x1, y1, x2, y2 in segs:
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if ang < 20 or ang > 160:
            horiz.append((x1, y1, x2, y2))
        elif 70 < ang < 110:
            vert.append((x1, y1, x2, y2))
    if len(horiz) < 2 or len(vert) < 2:
        return []

    def line_params(seg):
        x1, y1, x2, y2 = seg
        # for near-horizontal: y = m x + c ; for near-vertical: x = m y + c
        if abs(y2 - y1) > abs(x2 - x1):
            return ('v', x1 / max(abs(y2 - y1), 1e-6), x1 - (x1 / max(abs(y2 - y1), 1e-6)) * y1)
        return ('h', (y2 - y1) / max(abs(x2 - x1), 1e-6), y1 - ((y2 - y1) / max(abs(x2 - x1), 1e-6)) * x1)

    def intersect(l1, l2):
        (t1, m1, c1), (t2, m2, c2) = l1, l2
        if t1 == t2:
            return None
        if t1 == 'h':  # y = m1 x + c1 ; x = m2 y + c2
            y = (m1 * (m2 * 0 + c2) + c1) / (1 - m1 * m2) if abs(1 - m1 * m2) > 1e-6 else None
            if y is None:
                return None
            return np.array([m2 * y + c2, y])
        # t1 vertical, t2 horizontal
        x = m1 * (m2 * 0 + c2) + c1  # x = m1*y + c1 with y = m2 x + c2
        # solve properly: y = m2*x + c2 ; x = m1*y + c1
        # x = m1*(m2*x + c2) + c1 -> x(1 - m1*m2) = m1*c2 + c1
        den = 1 - m1 * m2
        if abs(den) < 1e-6:
            return None
        x = (m1 * c2 + c1) / den
        y = m2 * x + c2
        return np.array([x, y])

    hl = [line_params(s) for s in horiz]
    vl = [line_params(s) for s in vert]
    # try the strongest few lines to limit combinations
    for i, top in enumerate(hl[:6]):
        for bot in hl[i + 1:i + 7]:
            for left in vl[:6]:
                for right in vl[:6]:
                    tl = intersect(left, top)
                    tr = intersect(right, top)
                    br = intersect(right, bot)
                    bl = intersect(left, bot)
                    if any(p is None for p in (tl, tr, br, bl)):
                        continue
                    quad = np.array([tl, tr, br, bl], dtype=np.float32)
                    if (quad[:, 0].min() < -w * 0.02 or quad[:, 0].max() > w * 1.02
                            or quad[:, 1].min() < -h * 0.02 or quad[:, 1].max() > h * 1.02):
                        continue
                    scored = _score_candidate(gray, quad, img_area, 'hough')
                    if scored:
                        cands.append(scored)
    return cands


def detect_photograph(bgr, max_candidates=8):
    """Return ranked candidate quadrilaterals for the printed photograph.

    Each candidate: {points, confidence, strategy, area_fraction, aspect}.
    points are ordered tl,tr,br,bl in source-image pixel coordinates.
    """
    h, w = bgr.shape[:2]
    scale = min(1.0, 1600 / max(h, w))
    small = cv2.resize(bgr, (int(w * scale), int(h * scale))) if scale < 1 else bgr
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 40, 40)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 130)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))

    cands = _strategy_contours(gray, edges) + _strategy_hough(gray, edges)
    if scale < 1:
        for c in cands:
            c['points'] = (np.asarray(c['points']) / scale).tolist()

    # de-duplicate near-identical quads, keep highest confidence
    cands.sort(key=lambda c: c['confidence'], reverse=True)
    unique = []
    for c in cands:
        pts = np.asarray(c['points'])
        dup = False
        for u in unique:
            up = np.asarray(u['points'])
            if np.allclose(np.sort(pts.reshape(-1)), np.sort(up.reshape(-1)), atol=max(h, w) * 0.03):
                dup = True
                break
        if not dup:
            unique.append(c)
        if len(unique) >= max_candidates:
            break
    return unique


LOW_CONFIDENCE = 0.45


def best_candidate(candidates):
    if not candidates:
        return None, 0.0
    top = candidates[0]
    return top, top['confidence']

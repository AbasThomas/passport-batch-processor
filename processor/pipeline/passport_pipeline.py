"""Single-image passport pipeline.

Stages: normalize -> photo extraction -> perspective correction -> validation ->
face detection/alignment -> segmentation -> background replacement ->
enhancement -> passport crop -> compliance validation -> export.

Every stage returns a structured StageResult; nothing fabricates success.
"""
import cv2
import numpy as np
from PIL import Image

from photo_extraction.boundary_detector import detect_photograph, best_candidate, LOW_CONFIDENCE
from photo_extraction.perspective_corrector import rectify
from photo_extraction.extraction_validator import validate_extraction
from photo_extraction.manual_crop import manual_rectify
from face_processing.aligner import align
from face_processing.crop_calculator import get_profile, compute_crop, output_size
from background_removal.segmentation_service import segment, alpha_stats
from background_removal.mask_refiner import refine_alpha, despill
from background_removal.background_compositor import composite, parse_color
from enhancement import denoiser, color_corrector, sharpener, quality_analyzer


class Review(Exception):
    def __init__(self, message, **extra):
        super().__init__(message)
        self.extra = extra


def _stage(name, ok, confidence=None, detail='', warnings=None):
    return {'name': name, 'ok': bool(ok), 'confidence': confidence,
            'detail': detail, 'warnings': warnings or []}


def extract_photo(bgr, settings):
    """Stage: detect + rectify the printed photograph.

    Returns (photo_bgr, stage, candidates). Raises Review when the user must
    confirm a region manually.
    """
    if settings.get('manualCorners'):
        try:
            photo, meta = manual_rectify(bgr, settings['manualCorners'])
            st = _stage('photo_extraction', True, 1.0,
                        'manual four-corner selection')
            st['manual'] = True
            return photo, st, []
        except ValueError as e:
            raise Review(f'Invalid manual crop: {e}')

    candidates = detect_photograph(bgr)
    top, conf = best_candidate(candidates)
    if top is None:
        if settings.get('wholeImageFallback', True):
            # Upload may already be a clean photo rather than a print on a form.
            # Use the full frame explicitly — never an unrelated detected region.
            st = _stage('photo_extraction', True, 0.0,
                        'no printed-photo boundary found; using whole image')
            st['warnings'] = ['No photo boundary detected; verify the result carefully']
            return bgr, st, []
        raise Review('No printed photograph could be detected; '
                     'please crop it manually', candidates=[])
    if conf < LOW_CONFIDENCE:
        raise Review(
            f'Photo boundary confidence is low ({conf:.0%}); '
            'please confirm the photograph region',
            candidates=candidates)

    photo, ow, oh = rectify(bgr, top['points'])
    # A genuine printed portrait contains the face. If the chosen quad has no
    # face but the full frame does, the boundary is probably wrong — fall back
    # to the whole image rather than silently cropping an unrelated region.
    from face_processing.face_detector import detect_faces
    scale = min(1.0, 480 / max(ow, oh))
    probe = cv2.resize(photo, (int(ow * scale), int(oh * scale)))
    if not detect_faces(probe):
        fscale = min(1.0, 640 / max(bgr.shape[:2]))
        full = cv2.resize(bgr, (int(bgr.shape[1] * fscale), int(bgr.shape[0] * fscale)))
        if detect_faces(full):
            st = _stage('photo_extraction', True, float(conf * 0.5),
                        f'{top["strategy"]} quad rejected: no face inside; using whole image')
            st['warnings'] = ['Detected boundary did not contain a face; verify region']
            return bgr, st, candidates
    stage = _stage('photo_extraction', True, conf,
                   f'{top["strategy"]} quadrilateral {ow}x{oh}',
                   warnings=['Verify the extracted region is correct'] if conf < 0.7 else [])
    return photo, stage, candidates


def run(input_path, output_path, settings):
    """Full pipeline. Returns result dict; raises Review for user-actionable stops."""
    stages = []
    s = settings or {}

    # --- 1. normalize ---
    src = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if src is None:
        raise Review('Unreadable image')
    h, w = src.shape[:2]
    stages.append(_stage('normalize', True, 1.0, f'{w}x{h} source image'))

    # --- 2/3. extraction + perspective ---
    photo, ext_stage, candidates = extract_photo(src, s)
    stages.append(ext_stage)
    ok, warns, metrics = validate_extraction(photo)
    ext_stage['warnings'] += warns
    stages.append(_stage('extraction_validation', ok, None,
                         f'{metrics["width"]}x{metrics["height"]} photo', warns))

    # --- 4/5. face detection + alignment ---
    aligned = align(photo, preferred_index=s.get('faceIndex'))
    if not aligned['ok']:
        raise Review(aligned['reason'], status=aligned['status'],
                     faces=_serialize_faces(aligned.get('faces', [])),
                     geom=aligned.get('geom'))
    geom = aligned['geom']
    stages.append(_stage('face_alignment', True, geom['score'],
                         f"face score {geom['score']:.2f}, roll {aligned.get('roll_deg', 0):+.1f}°",
                         warnings=(['Head turned sideways — verify result']
                                   if not geom['frontal'] else [])))

    work = aligned['image']

    # --- 6/7. segmentation + background replacement ---
    rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
    quality = quality_analyzer.analyze(rgb)
    stages.append(_stage('quality_analysis', quality['grade'] != 'poor', None,
                         f"quality {quality['grade']}", quality['warnings']))

    pil = Image.fromarray(rgb)
    rgba_pil, model_used = segment(pil, s.get('segmentModel', 'auto'))
    rgba = np.asarray(rgba_pil)
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise Review('Person segmentation returned no alpha channel')
    stats = alpha_stats(rgba)
    coverage_ok = 0.02 < stats['coverage'] < 0.9
    stages.append(_stage('segmentation', coverage_ok, None,
                         f'model={model_used} coverage={stats["coverage"]:.1%}',
                         [] if coverage_ok else ['Segmentation coverage unusual; verify edges']))

    alpha = refine_alpha(rgba[:, :, 3])
    fg = despill(rgba[:, :, :3], alpha)
    color = parse_color(s.get('background', '#ffffff'))
    composited = composite(fg, alpha, color)

    # --- 8. enhancement (on composited portrait) ---
    if s.get('enhance', True):
        comp_bgr = cv2.cvtColor(composited, cv2.COLOR_RGB2BGR)
        comp_bgr, d_info = denoiser.maybe_denoise(comp_bgr)
        comp_bgr = color_corrector.correct_contrast(comp_bgr)
        comp_bgr, e_info = color_corrector.correct_exposure(comp_bgr)
        comp_bgr = color_corrector.gray_world_white_balance(comp_bgr, strength=0.4)
        comp_bgr, sh_info = sharpener.maybe_sharpen(comp_bgr)
        composited = cv2.cvtColor(comp_bgr, cv2.COLOR_BGR2RGB)
        stages.append(_stage('enhancement', True, None,
                             f"denoise={d_info.get('applied')} exposure={e_info.get('applied')}"))
    else:
        stages.append(_stage('enhancement', True, None, 'disabled'))

    # --- 9. passport crop ---
    profile = get_profile(s.get('profile'))
    ih, iw = composited.shape[:2]
    crop_rect, fits = compute_crop(
        geom, iw, ih, profile,
        zoom=float(s.get('zoom', 1.0)),
        pan_x=float(s.get('panX', 0.0)), pan_y=float(s.get('panY', 0.0)))
    cx, cy, cw, ch = crop_rect
    cropped = composited[cy:cy + ch, cx:cx + cw]
    stages.append(_stage('passport_crop', fits, None,
                         f'profile={profile.name} crop={cw}x{ch}' + ('' if fits else ' (image-limited)')))

    # --- 10. resize to output dims + compliance validation ---
    tw = int(s.get('width') or output_size(profile, int(s.get('dpi', 300)))[0])
    th = int(s.get('height') or output_size(profile, int(s.get('dpi', 300)))[1])
    resized = cv2.resize(cropped, (tw, th), interpolation=cv2.INTER_LANCZOS4)
    compliance_warnings = _validate_compliance(resized, color)
    stages.append(_stage('compliance', not compliance_warnings, None,
                         f'output {tw}x{th}', compliance_warnings))

    # --- 11. export ---
    from pathlib import Path
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fmt = (s.get('format') or 'png').lower()
    pil_out = Image.fromarray(resized)  # resized is RGB; PIL expects RGB
    dpi = int(s.get('dpi', 300))
    if fmt == 'jpg' or fmt == 'jpeg':
        pil_out.convert('RGB').save(out, format='JPEG', quality=int(s.get('quality', 90)),
                                    dpi=(dpi, dpi))
    else:
        pil_out.save(out, format='PNG', dpi=(dpi, dpi))
    stages.append(_stage('export', True, 1.0, str(out)))

    return {
        'ok': True, 'manualReview': False,
        'stages': stages,
        'result': {
            'width': tw, 'height': th, 'profile': profile.name,
            'background': s.get('background', '#ffffff'),
            'quality': quality,
            'faces': _serialize_faces(aligned['faces']),
            'extraction_confidence': ext_stage.get('confidence'),
        },
    }


def _serialize_faces(faces):
    return [{'bbox': f['bbox'], 'score': f['score'], 'face_index': f['face_index']}
            for f in faces]


def _validate_compliance(out_rgb, bg_color):
    """Objective checks: dimensions already set; verify clean uniform border."""
    warnings = []
    h, w = out_rgb.shape[:2]
    border = np.concatenate([
        out_rgb[:3].reshape(-1, 3), out_rgb[-3:].reshape(-1, 3),
        out_rgb[:, :3].reshape(-1, 3), out_rgb[:, -3:].reshape(-1, 3)])
    mean = border.mean()
    if mean < 200 and bg_color != (0, 0, 0):
        warnings.append('Border is not clean background; check crop/segmentation edges')
    # face should not touch the frame edge
    gray = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2GRAY)
    edge_content = max(
        float((gray[:8] < 200).mean()), float((gray[-8:] < 200).mean()),
        float((gray[:, :8] < 200).mean()), float((gray[:, -8:] < 200).mean()))
    if edge_content > 0.5:
        warnings.append('Subject content touches frame edge; headroom/shoulders may be cropped')
    return warnings

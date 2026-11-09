# SnapID Implementation Plan

## Architecture Decisions

**Decision 1 — Python package layout.**
Every sub-package gets its own `__init__.py`. All Python modules inside a sub-package use relative imports (e.g. `from .boundary_detector import BoundaryDetector`). `app.py` imports from the packages using absolute imports (e.g. `from photo_extraction.boundary_detector import BoundaryDetector`). The processor Dockerfile must be updated to `COPY . .` (instead of `COPY app.py .`) so all sub-packages are included in the container image.

**Decision 2 — MediaPipe vs YuNet fallback.**
`face_detector.py` tries `mediapipe.tasks.vision.FaceLandmarker` first. If the `.task` model file is absent or if `AttributeError` is raised (mediapipe not fully initialised), it silently falls back to OpenCV's `cv2.FaceDetectorYN`. The landmark detector similarly falls back to a geometric eye-estimation from the YuNet bounding box. This means the pipeline works even when the mediapipe model file has not been pre-downloaded.

**Decision 3 — MediaPipe model file distribution.**
The face_landmarker `.task` bundle is downloaded once at container startup via a `_ensure_model()` helper in `landmark_detector.py`, stored at `$U2NET_HOME/face_landmarker.task` (the same model cache volume used for rembg). No external calls are made per-request.

**Decision 4 — BiRefNet vs U2Net segmentation.**
`segmentation_service.py` tries `birefnet-portrait` first; if `rembg.new_session` raises or the model returns an empty mask, it retries with `u2net-human-seg`. The active model name is surfaced in every response.

**Decision 5 — `/process` backward compatibility.**
`app.py`'s existing `@app.post('/process')` handler is NOT modified. All new functionality lives under `/process/v2`, `/process/extract-only`, `/process/confirm-crop`, `/process/correct`, and `/models/status`. The new handlers delegate to the pipeline orchestrator.

**Decision 6 — PipelineResult JSON shape.**
```json
{
  "ok": true,
  "stage": "extraction|correction|complete",
  "extractedPath": "/data/batches/{batchId}/extracted/{fileId}.png",
  "corners": [[x0,y0],[x1,y1],[x2,y2],[x3,y3]],
  "extractionConfidence": 0.96,
  "faceDetected": true,
  "faceConfidence": 0.99,
  "alignmentAngle": 1.2,
  "segmentationModel": "birefnet-portrait",
  "outputPath": "/data/batches/{batchId}/output/{fileId}.jpg",
  "width": 600,
  "height": 600,
  "detectedPhoto": true,
  "manualReview": false,
  "qualityReport": {
    "blur_score": 0.85,
    "exposure_ok": true,
    "resolution_ok": true,
    "face_centered": true
  },
  "error": null
}
```
The `stage` field tells the Node worker where processing stopped. `manualReview: true` means the user must intervene before correction can proceed.

**Decision 7 — FileRecord additions.**
Three optional fields are appended: `extractedPath?`, `confirmedCorners?`, `stage?`. Being optional means existing Redis hashes (with none of these keys) continue to deserialise as valid `FileRecord` objects without code changes in any existing path.

**Decision 8 — Worker v2 routing.**
The worker reads `job.data.mode` (default `'legacy'`). When `mode === 'snapid'` it calls `/process/v2` and maps the `PipelineResult.stage` to state machine states. The existing `mode === 'legacy'` path is unchanged.

**Decision 9 — Frontend routing.**
`public/index.html` becomes the 5-step SnapID UI by default. `?view=batches` renders the legacy batch dashboard. Both views are served from the same HTML file; a `<script>` block at the top checks `location.search` and mounts the appropriate view.

**Decision 10 — asyncio concurrency in batch_pipeline.py.**
`asyncio.Semaphore(GPU_CONCURRENCY)` limits simultaneous `run_in_executor` calls that invoke the ONNX session. `GPU_CONCURRENCY` defaults to 1 and is read from the environment variable of the same name (already present in docker-compose).

---

## Implementation Plan

- [ ] 1. Create the `.agents/tasks/snapid-2025-07-14/` artifact directory tree and write `task.json`, `context.json`, and all `FEAT-NNN.json` files.
      This is the planning artefact step — files live outside the worktree.
      Files: `.agents/tasks/snapid-2025-07-14/task.json`, `context.json`, `features/FEAT-001.json` … `FEAT-005.json`
      Verify: All JSON files are valid (parse without error).

- [ ] 2. Update `processor/requirements.txt` to add `mediapipe==0.10.14`, `scipy==1.14.1`, and `requests==2.32.3` (needed by `_ensure_model()` download helper).
      Files: `processor/requirements.txt`
      Verify: `pip install -r processor/requirements.txt --dry-run` exits 0 (or equivalent Docker build check).

- [ ] 3. Update `processor/Dockerfile` to install `libgl1` (required by mediapipe in headless Linux) and change `COPY app.py .` to `COPY . .` so all sub-packages land in the image.
      Files: `processor/Dockerfile`
      Verify: `docker-compose build processor` exits 0.

- [ ] 4. Create `processor/models.py` — all Pydantic data-transfer types used across the pipeline.
      Exact classes and fields listed in the module-signatures section below.
      Files: `processor/models.py`
      Verify: `python -c "from models import PipelineResult, ExtractionResult, FaceDetectionResult"` inside the processor container exits 0.

- [ ] 5. Create `processor/photo_extraction/__init__.py` (empty), `boundary_detector.py`, `quadrilateral_detector.py`, `perspective_corrector.py`, `extraction_validator.py`, `manual_crop.py`.
      Signatures specified below.
      Files: `processor/photo_extraction/__init__.py`, `processor/photo_extraction/boundary_detector.py`, `processor/photo_extraction/quadrilateral_detector.py`, `processor/photo_extraction/perspective_corrector.py`, `processor/photo_extraction/extraction_validator.py`, `processor/photo_extraction/manual_crop.py`
      Verify: `python -c "from photo_extraction.boundary_detector import BoundaryDetector"` exits 0.

- [ ] 6. Create `processor/face_processing/__init__.py`, `face_detector.py`, `landmark_detector.py`, `face_aligner.py`, `crop_calculator.py`.
      Files: `processor/face_processing/__init__.py`, `processor/face_processing/face_detector.py`, `processor/face_processing/landmark_detector.py`, `processor/face_processing/face_aligner.py`, `processor/face_processing/crop_calculator.py`
      Verify: `python -c "from face_processing.face_detector import FaceDetector"` exits 0.

- [ ] 7. Create `processor/background_removal/__init__.py`, `segmentation_service.py`, `birefnet_provider.py`, `u2net_provider.py`, `mask_refiner.py`, `background_compositor.py`.
      Files: `processor/background_removal/__init__.py`, `processor/background_removal/segmentation_service.py`, `processor/background_removal/birefnet_provider.py`, `processor/background_removal/u2net_provider.py`, `processor/background_removal/mask_refiner.py`, `processor/background_removal/background_compositor.py`
      Verify: `python -c "from background_removal.segmentation_service import SegmentationService"` exits 0.

- [ ] 8. Create `processor/enhancement/__init__.py`, `denoiser.py`, `color_corrector.py`, `sharpener.py`, `quality_analyzer.py`.
      Files: `processor/enhancement/__init__.py`, `processor/enhancement/denoiser.py`, `processor/enhancement/color_corrector.py`, `processor/enhancement/sharpener.py`, `processor/enhancement/quality_analyzer.py`
      Verify: `python -c "from enhancement.quality_analyzer import QualityAnalyzer"` exits 0.

- [ ] 9. Create `processor/pipeline/__init__.py`, `passport_pipeline.py`, `batch_pipeline.py`, `pipeline_orchestrator.py`.
      Files: `processor/pipeline/__init__.py`, `processor/pipeline/passport_pipeline.py`, `processor/pipeline/batch_pipeline.py`, `processor/pipeline/pipeline_orchestrator.py`
      Verify: `python -c "from pipeline.pipeline_orchestrator import PipelineOrchestrator"` exits 0.

- [ ] 10. Refactor `processor/app.py` to add new endpoints while keeping `/health` and `/process` exactly as-is.
      Add: `/process/v2`, `/process/extract-only`, `/process/confirm-crop`, `/process/correct`, `/models/status`.
      Files: `processor/app.py`
      Verify: `uvicorn app:app --port 8001 &` then `curl -s http://localhost:8001/health | python -c "import sys,json; d=json.load(sys.stdin); assert d['ok']"` exits 0.

- [ ] 11. Update `src/lib.ts` to add optional fields to `FileRecord` and add `Stage` type.
      Files: `src/lib.ts`
      Verify: `npm run build` exits 0 with no TypeScript errors.

- [ ] 12. Update `src/worker.ts` to add v2 dispatch path (mode=snapid) while keeping legacy path unchanged.
      Files: `src/worker.ts`
      Verify: `npm run build` exits 0.

- [ ] 13. Update `src/server.ts` to add new SnapID routes while keeping all existing routes.
      Add: `POST /api/files/:id/extract`, `POST /api/files/:id/confirm-crop`, `POST /api/files/:id/correct`, `GET /api/files/:id/preview`.
      Files: `src/server.ts`
      Verify: `npm run build` exits 0.

- [ ] 14. Replace `public/index.html`, `public/app.js`, `public/style.css` with the 5-step SnapID UI. Legacy batch dashboard renders when `?view=batches` is present in the URL.
      Files: `public/index.html`, `public/app.js`, `public/style.css`
      Verify: Open in browser; confirm step 1 renders; confirm `?view=batches` shows the old dashboard.

- [ ] 15. Integration smoke-test: start all containers, upload a test image, call `/process/v2`, verify JSON response matches PipelineResult shape.
      Verify: `docker-compose up -d && curl -s -X POST http://localhost:8000/models/status | python -c "import sys,json; d=json.load(sys.stdin); print(d['models'])"` exits 0.

---

## Module Signatures

### `processor/models.py`
```python
from pydantic import BaseModel
from typing import Optional, List, Tuple

class BoundaryCandidate(BaseModel):
    corners: List[Tuple[float, float]]   # 4 (x,y) pairs in TL,TR,BR,BL order
    area_ratio: float                    # fraction of image area
    confidence: float                    # 0..1

class DetectedQuad(BaseModel):
    corners: List[Tuple[float, float]]   # ordered TL,TR,BR,BL
    confidence: float
    method: str                          # 'contour'|'hough'|'adaptive'|'manual'

class FaceBox(BaseModel):
    x: int; y: int; w: int; h: int
    confidence: float

class FaceDetectionResult(BaseModel):
    ok: bool
    faces: List[FaceBox]
    primary: Optional[FaceBox]
    detector_used: str                   # 'mediapipe'|'yunet'|'none'
    error: Optional[str]

class LandmarkPoint(BaseModel):
    x: float; y: float; z: float

class LandmarkResult(BaseModel):
    ok: bool
    landmarks: List[LandmarkPoint]       # 478 points if mediapipe, else []
    left_eye: Tuple[float, float]        # (cx, cy) in pixels
    right_eye: Tuple[float, float]
    nose_tip: Tuple[float, float]
    chin: Tuple[float, float]
    roll_angle: float                    # degrees; positive = CCW
    error: Optional[str]

class AlignmentResult(BaseModel):
    ok: bool
    rotated_image: Optional[bytes]       # PNG bytes, None if stored on disk
    rotation_applied: float
    error: Optional[str]

class CropResult(BaseModel):
    ok: bool
    x: int; y: int; w: int; h: int
    head_height_px: int
    headroom_px: int
    profile_used: str
    error: Optional[str]

class SegmentationResult(BaseModel):
    ok: bool
    alpha_path: Optional[str]
    composited_path: Optional[str]
    model_used: str
    coverage: float                      # foreground fraction 0..1
    error: Optional[str]

class QualityReport(BaseModel):
    blur_score: float                    # 0..1 higher=sharper
    exposure_ok: bool
    resolution_ok: bool
    face_centered: bool
    noise_level: float                   # 0..1 lower=cleaner
    warnings: List[str]

class ExtractionResult(BaseModel):
    ok: bool
    extracted_path: Optional[str]
    corners: Optional[List[Tuple[float, float]]]
    confidence: float
    method: str
    manual_review_needed: bool
    error: Optional[str]

class PipelineResult(BaseModel):
    ok: bool
    stage: str                           # 'extraction'|'correction'|'complete'
    extracted_path: Optional[str]
    corners: Optional[List[Tuple[float, float]]]
    extraction_confidence: float
    face_detected: bool
    face_confidence: float
    alignment_angle: float
    segmentation_model: str
    output_path: Optional[str]
    width: int
    height: int
    detected_photo: bool
    manual_review: bool
    quality_report: Optional[QualityReport]
    error: Optional[str]

class DocumentProfile(BaseModel):
    name: str                            # e.g. 'ICAO_35x45'
    width_mm: float
    height_mm: float
    face_fraction: float                 # fraction of height the face occupies
    eye_line_fraction: float             # from top where eyes should sit
    min_dpi: int
    background_color: str               # hex default background

class V2Request(BaseModel):
    input_path: str
    output_path: str
    settings: dict
    mode: str = 'full'                   # 'full'|'extract_only'

class ConfirmCropRequest(BaseModel):
    input_path: str
    output_path: str
    corners: List[Tuple[float, float]]
    settings: dict

class CorrectRequest(BaseModel):
    extracted_path: str
    output_path: str
    settings: dict
```

### `processor/photo_extraction/boundary_detector.py`
```python
import cv2, numpy as np
from models import BoundaryCandidate, DetectedQuad
from typing import List, Tuple

class BoundaryDetector:
    def __init__(self, scale_limit: int = 1600): ...

    def detect(self, bgr: np.ndarray) -> List[BoundaryCandidate]:
        """Run three strategies (contour, adaptive-threshold, hough) and return all
        plausible candidates sorted by confidence descending."""

    def _contour_strategy(self, small: np.ndarray, scale: float) -> List[BoundaryCandidate]: ...
    def _adaptive_strategy(self, small: np.ndarray, scale: float) -> List[BoundaryCandidate]: ...
    def _hough_strategy(self, small: np.ndarray, scale: float) -> List[BoundaryCandidate]: ...
    def _score_candidate(self, corners: np.ndarray, img_area: int) -> float: ...
```

### `processor/photo_extraction/quadrilateral_detector.py`
```python
import cv2, numpy as np
from models import BoundaryCandidate, DetectedQuad
from typing import List, Optional

class QuadrilateralDetector:
    def select_best(self, candidates: List[BoundaryCandidate], image_shape: Tuple[int,int]) -> Optional[DetectedQuad]:
        """Pick highest-confidence candidate that passes geometric sanity checks."""

    def order_corners(self, pts: np.ndarray) -> np.ndarray:
        """Return corners ordered TL, TR, BR, BL."""

    def aspect_ok(self, quad: DetectedQuad) -> bool:
        """True if width/height ratio is between 0.45 and 1.8."""
```

### `processor/photo_extraction/perspective_corrector.py`
```python
import cv2, numpy as np
from models import DetectedQuad
from typing import Tuple

class PerspectiveCorrector:
    def rectify(self, bgr: np.ndarray, quad: DetectedQuad) -> np.ndarray:
        """Apply getPerspectiveTransform + warpPerspective. Returns rectified BGR."""

    def _compute_output_size(self, quad: DetectedQuad) -> Tuple[int, int]:
        """Width and height derived from corner distances."""
```

### `processor/photo_extraction/extraction_validator.py`
```python
import numpy as np
from models import ExtractionResult, DetectedQuad
from typing import Optional

class ExtractionValidator:
    def validate(self, original: np.ndarray, extracted: np.ndarray,
                 quad: Optional[DetectedQuad]) -> ExtractionResult:
        """Return ExtractionResult with confidence score and manual_review_needed flag."""

    def _confidence_from_quad(self, quad: DetectedQuad, img_area: int) -> float: ...
    def _is_too_small(self, img: np.ndarray) -> bool: ...
```

### `processor/photo_extraction/manual_crop.py`
```python
import cv2, numpy as np
from models import DetectedQuad
from typing import List, Tuple

def corners_from_payload(raw: List[List[float]]) -> DetectedQuad:
    """Parse [[x0,y0],[x1,y1],[x2,y2],[x3,y3]] from the API request into a DetectedQuad
    with confidence=1.0 and method='manual'."""

def validate_manual_corners(corners: List[Tuple[float,float]], img_shape: Tuple[int,int]) -> bool:
    """Return True if all four corners are within image bounds and form a convex quadrilateral."""
```

### `processor/face_processing/face_detector.py`
```python
import cv2, numpy as np
from models import FaceDetectionResult, FaceBox
from typing import Optional

class FaceDetector:
    def __init__(self): ...
    # Tries mediapipe FaceDetector task; falls back to YuNet on AttributeError or missing model.

    def detect(self, rgb: np.ndarray) -> FaceDetectionResult:
        """Return all detected faces. primary = largest by area."""

    def _detect_mediapipe(self, rgb: np.ndarray) -> FaceDetectionResult: ...
    def _detect_yunet(self, bgr: np.ndarray) -> FaceDetectionResult: ...
    def _ensure_yunet(self) -> cv2.FaceDetectorYN: ...
```

### `processor/face_processing/landmark_detector.py`
```python
import numpy as np
from models import LandmarkResult
from pathlib import Path
import os

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = Path(os.getenv("U2NET_HOME", "/models")) / "face_landmarker.task"

def _ensure_model() -> Path:
    """Download face_landmarker.task to MODEL_PATH if not already present. Uses requests."""

class LandmarkDetector:
    def __init__(self): ...
    # _ensure_model() is called once in __init__ with a try/except; falls back to geometric mode.

    def detect(self, rgb: np.ndarray, face_box: "FaceBox") -> LandmarkResult:
        """Return 478-point landmarks if mediapipe loaded; else geometric estimate from face_box."""

    def _mediapipe_detect(self, rgb: np.ndarray) -> LandmarkResult: ...
    def _geometric_estimate(self, face_box: "FaceBox") -> LandmarkResult:
        """Estimate eye and chin positions from face_box proportions (eyes at 35%, chin at 85%)."""
```

### `processor/face_processing/face_aligner.py`
```python
import cv2, numpy as np
from models import LandmarkResult, AlignmentResult

MAX_ROLL_DEGREES = 15.0  # refuse alignment beyond this; request better photo

class FaceAligner:
    def align(self, bgr: np.ndarray, landmarks: LandmarkResult) -> AlignmentResult:
        """Rotate image to level the eyes. Returns AlignmentResult with rotated image bytes."""

    def _rotation_matrix(self, center: tuple, angle: float, scale: float) -> np.ndarray: ...
    def _is_too_sideways(self, landmarks: LandmarkResult) -> bool:
        """True if yaw exceeds 30° (estimated from landmark spread ratio)."""
```

### `processor/face_processing/crop_calculator.py`
```python
import numpy as np
from models import LandmarkResult, CropResult, DocumentProfile

class CropCalculator:
    def calculate(self, img_shape: tuple, landmarks: LandmarkResult,
                  profile: DocumentProfile) -> CropResult:
        """Compute crop rectangle respecting face fraction, eye line, and headroom."""

    def _head_height(self, landmarks: LandmarkResult) -> int: ...
    def _apply_profile(self, cx: float, top_eye: float, head_h: int,
                       profile: DocumentProfile, img_h: int, img_w: int) -> CropResult: ...
```

### `processor/background_removal/segmentation_service.py`
```python
import numpy as np
from models import SegmentationResult
from PIL import Image
from .birefnet_provider import BiRefNetProvider
from .u2net_provider import U2NetProvider

class SegmentationService:
    def __init__(self):
        self._birefnet = BiRefNetProvider()
        self._u2net = U2NetProvider()

    def segment(self, rgb: np.ndarray) -> SegmentationResult:
        """Try BiRefNet; fall back to U2Net if mask coverage is out of range."""

    def _coverage_ok(self, alpha: np.ndarray) -> bool:
        """True if foreground coverage is between 3.5% and 83%."""
```

### `processor/background_removal/birefnet_provider.py`
```python
from rembg import new_session, remove
import numpy as np
from PIL import Image

class BiRefNetProvider:
    _session = None
    MODEL = 'birefnet-portrait'

    def get_session(self):
        if self._session is None:
            self._session = new_session(self.MODEL)
        return self._session

    def remove_bg(self, rgb: np.ndarray) -> np.ndarray:
        """Return RGBA numpy array. Raises RuntimeError on failure."""
```

### `processor/background_removal/u2net_provider.py`
```python
from rembg import new_session, remove
import numpy as np

class U2NetProvider:
    _session = None
    MODEL = 'u2net_human_seg'

    def get_session(self): ...
    def remove_bg(self, rgb: np.ndarray) -> np.ndarray: ...
```

### `processor/background_removal/mask_refiner.py`
```python
import cv2, numpy as np

class MaskRefiner:
    def refine(self, alpha: np.ndarray) -> np.ndarray:
        """Morphological open+close, Gaussian blur on edges, remove tiny islands."""

    def _remove_halo(self, alpha: np.ndarray, rgb: np.ndarray) -> np.ndarray: ...
```

### `processor/background_removal/background_compositor.py`
```python
import numpy as np
from PIL import Image

class BackgroundCompositor:
    def composite(self, rgba: np.ndarray, bg_color: tuple) -> np.ndarray:
        """Alpha-blend RGBA over solid bg_color. Returns RGB uint8."""

    def _parse_color(self, hex_str: str) -> tuple:
        """Parse '#rrggbb' → (r,g,b)."""
```

### `processor/enhancement/denoiser.py`
```python
import cv2, numpy as np

class Denoiser:
    def denoise(self, bgr: np.ndarray) -> np.ndarray:
        """Apply fastNlMeansDenoisingColored with h=4 (conservative)."""
```

### `processor/enhancement/color_corrector.py`
```python
import numpy as np
from PIL import Image, ImageEnhance

class ColorCorrector:
    def correct(self, rgb: np.ndarray, settings: dict) -> np.ndarray:
        """Apply color/contrast/white-balance corrections from settings dict."""
```

### `processor/enhancement/sharpener.py`
```python
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

class Sharpener:
    def sharpen(self, rgb: np.ndarray, level: float = 1.18) -> np.ndarray:
        """Apply UnsharpMask + Sharpness enhance. level ≤ 1.0 means no-op."""
```

### `processor/enhancement/quality_analyzer.py`
```python
import cv2, numpy as np
from models import QualityReport

class QualityAnalyzer:
    def analyze(self, bgr: np.ndarray) -> QualityReport:
        """Compute blur_score (Laplacian variance), exposure, resolution, noise."""

    def _blur_score(self, gray: np.ndarray) -> float:
        """Normalised Laplacian variance; 1.0 = sharp."""

    def _exposure_ok(self, gray: np.ndarray) -> bool:
        """True if mean brightness is between 60 and 195."""

    def _resolution_ok(self, h: int, w: int) -> bool:
        """True if min(h,w) >= 300."""
```

### `processor/pipeline/passport_pipeline.py`
```python
import cv2, numpy as np
from models import (PipelineResult, ExtractionResult, V2Request,
                    ConfirmCropRequest, CorrectRequest, DocumentProfile)
from photo_extraction.boundary_detector import BoundaryDetector
from photo_extraction.quadrilateral_detector import QuadrilateralDetector
from photo_extraction.perspective_corrector import PerspectiveCorrector
from photo_extraction.extraction_validator import ExtractionValidator
from photo_extraction.manual_crop import corners_from_payload
from face_processing.face_detector import FaceDetector
from face_processing.landmark_detector import LandmarkDetector
from face_processing.face_aligner import FaceAligner
from face_processing.crop_calculator import CropCalculator
from background_removal.segmentation_service import SegmentationService
from background_removal.mask_refiner import MaskRefiner
from background_removal.background_compositor import BackgroundCompositor
from enhancement.denoiser import Denoiser
from enhancement.color_corrector import ColorCorrector
from enhancement.sharpener import Sharpener
from enhancement.quality_analyzer import QualityAnalyzer

PROFILES = {
    'ICAO_35x45': DocumentProfile(name='ICAO_35x45', width_mm=35, height_mm=45,
                                   face_fraction=0.70, eye_line_fraction=0.40,
                                   min_dpi=300, background_color='#ffffff'),
    'US_2x2': DocumentProfile(name='US_2x2', width_mm=50.8, height_mm=50.8,
                               face_fraction=0.65, eye_line_fraction=0.38,
                               min_dpi=300, background_color='#ffffff'),
}

class PassportPipeline:
    def __init__(self): ...
    # Instantiates all sub-service objects once; they are reused across requests.

    def run_full(self, req: V2Request) -> PipelineResult: ...
    def run_extract_only(self, req: V2Request) -> PipelineResult: ...
    def run_correct(self, req: CorrectRequest) -> PipelineResult: ...
    def run_confirm_crop(self, req: ConfirmCropRequest) -> PipelineResult: ...

    def _extract(self, bgr, settings) -> tuple: ...    # → (extracted_bgr, ExtractionResult)
    def _correct(self, bgr, settings) -> PipelineResult: ...
    def _save_intermediate(self, path: str, bgr: np.ndarray) -> None: ...
```

### `processor/pipeline/batch_pipeline.py`
```python
import asyncio, os
from models import V2Request, PipelineResult
from .passport_pipeline import PassportPipeline

GPU_CONCURRENCY = int(os.getenv('GPU_CONCURRENCY', '1'))

class BatchPipeline:
    def __init__(self): ...

    async def process_many(self, requests: list[V2Request]) -> list[PipelineResult]:
        """Process requests with asyncio.Semaphore(GPU_CONCURRENCY)."""

    async def _process_one(self, req: V2Request, sem: asyncio.Semaphore,
                            loop: asyncio.AbstractEventLoop,
                            pipeline: PassportPipeline) -> PipelineResult: ...
```

### `processor/pipeline/pipeline_orchestrator.py`
```python
from models import V2Request, ConfirmCropRequest, CorrectRequest, PipelineResult
from .passport_pipeline import PassportPipeline

class PipelineOrchestrator:
    _pipeline: PassportPipeline = None

    @classmethod
    def get_pipeline(cls) -> PassportPipeline:
        if cls._pipeline is None:
            cls._pipeline = PassportPipeline()
        return cls._pipeline

    @classmethod
    def process_v2(cls, req: V2Request) -> PipelineResult: ...

    @classmethod
    def extract_only(cls, req: V2Request) -> PipelineResult: ...

    @classmethod
    def confirm_crop(cls, req: ConfirmCropRequest) -> PipelineResult: ...

    @classmethod
    def correct(cls, req: CorrectRequest) -> PipelineResult: ...
```

### Updated `processor/app.py` — new endpoints (existing unchanged)
```python
# Existing: /health, /process — NOT MODIFIED

from pipeline.pipeline_orchestrator import PipelineOrchestrator
from models import V2Request, ConfirmCropRequest, CorrectRequest

@app.post('/process/v2')
def process_v2(req: V2Request):
    ...

@app.post('/process/extract-only')
def extract_only(req: V2Request):
    ...

@app.post('/process/confirm-crop')
def confirm_crop(req: ConfirmCropRequest):
    ...

@app.post('/process/correct')
def correct(req: CorrectRequest):
    ...

@app.get('/models/status')
def models_status():
    """Return loaded models, their readiness and the mediapipe model file presence."""
    ...
```

---

## Exact Import Statements Per Module

```
# processor/photo_extraction/boundary_detector.py
import cv2
import numpy as np
from models import BoundaryCandidate, DetectedQuad
from typing import List, Tuple, Optional

# processor/photo_extraction/quadrilateral_detector.py
import cv2
import numpy as np
from models import BoundaryCandidate, DetectedQuad
from typing import List, Optional, Tuple

# processor/photo_extraction/perspective_corrector.py
import cv2
import numpy as np
from models import DetectedQuad
from typing import Tuple

# processor/photo_extraction/extraction_validator.py
import numpy as np
from models import ExtractionResult, DetectedQuad
from typing import Optional

# processor/photo_extraction/manual_crop.py
import cv2
import numpy as np
from models import DetectedQuad
from typing import List, Tuple

# processor/face_processing/face_detector.py
import cv2
import numpy as np
import os
from pathlib import Path
from models import FaceDetectionResult, FaceBox
from typing import Optional

# processor/face_processing/landmark_detector.py
import os
import numpy as np
import requests
from pathlib import Path
from models import LandmarkResult
from .face_detector import FaceBox

# processor/face_processing/face_aligner.py
import cv2
import numpy as np
from models import LandmarkResult, AlignmentResult

# processor/face_processing/crop_calculator.py
import numpy as np
from models import LandmarkResult, CropResult, DocumentProfile

# processor/background_removal/segmentation_service.py
import numpy as np
from PIL import Image
from models import SegmentationResult
from .birefnet_provider import BiRefNetProvider
from .u2net_provider import U2NetProvider
from .mask_refiner import MaskRefiner

# processor/background_removal/birefnet_provider.py
import numpy as np
from PIL import Image
from rembg import new_session, remove

# processor/background_removal/u2net_provider.py
import numpy as np
from PIL import Image
from rembg import new_session, remove

# processor/background_removal/mask_refiner.py
import cv2
import numpy as np

# processor/background_removal/background_compositor.py
import numpy as np
from PIL import Image

# processor/enhancement/denoiser.py
import cv2
import numpy as np

# processor/enhancement/color_corrector.py
import numpy as np
from PIL import Image, ImageEnhance

# processor/enhancement/sharpener.py
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

# processor/enhancement/quality_analyzer.py
import cv2
import numpy as np
from models import QualityReport

# processor/pipeline/passport_pipeline.py
import cv2
import numpy as np
from pathlib import Path
from models import (PipelineResult, ExtractionResult, V2Request,
                    ConfirmCropRequest, CorrectRequest, DocumentProfile)
from photo_extraction.boundary_detector import BoundaryDetector
from photo_extraction.quadrilateral_detector import QuadrilateralDetector
from photo_extraction.perspective_corrector import PerspectiveCorrector
from photo_extraction.extraction_validator import ExtractionValidator
from photo_extraction.manual_crop import corners_from_payload, validate_manual_corners
from face_processing.face_detector import FaceDetector
from face_processing.landmark_detector import LandmarkDetector
from face_processing.face_aligner import FaceAligner
from face_processing.crop_calculator import CropCalculator
from background_removal.segmentation_service import SegmentationService
from background_removal.background_compositor import BackgroundCompositor
from enhancement.denoiser import Denoiser
from enhancement.color_corrector import ColorCorrector
from enhancement.sharpener import Sharpener
from enhancement.quality_analyzer import QualityAnalyzer

# processor/pipeline/batch_pipeline.py
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from models import V2Request, PipelineResult
from .passport_pipeline import PassportPipeline

# processor/pipeline/pipeline_orchestrator.py
from models import V2Request, ConfirmCropRequest, CorrectRequest, PipelineResult
from .passport_pipeline import PassportPipeline
```

---

## Backward-Compatibility Contract for `/process`

The existing `@app.post('/process')` handler in `processor/app.py` must be reproduced character-for-character in the refactored file. The only permissible changes are:

1. The import block at the top of `app.py` may grow (adding new sub-package imports).
2. The `sessions` dict and `session()` helper remain at module level.
3. The `quad_crop()`, `enhance()`, and `validate()` functions remain at module level, unchanged.
4. The `Review` exception class remains at module level.
5. The existing `Request` Pydantic model remains (the new models live in `models.py`).
6. Response contract: `{"ok": bool, "detectedPhoto": bool, "manualReview": bool, "width": int, "height": int}` on success; `{"ok": false, "manualReview": bool, "error": str}` on failure.

The worker's existing `mode === 'legacy'` path hits `/process` and must continue to work. The worker's new `mode === 'snapid'` path hits `/process/v2`.

---

## PipelineResult → FileRecord State Mapping

| PipelineResult field | FileRecord.state |
|---|---|
| `ok=false, manual_review=true, stage='extraction'` | `needs_review` (user must confirm corners) |
| `ok=false, manual_review=false` | `failed` |
| `ok=true, stage='extraction'` | `needs_review` (extraction done, awaiting correction trigger) |
| `ok=true, stage='correction'` | `needs_review` (intermediate, rare) |
| `ok=true, stage='complete'` | `completed` |

When `stage='extraction'`, the worker also sets `FileRecord.extractedPath = result.extractedPath` and `FileRecord.stage = 'extraction'`. When `stage='complete'`, `FileRecord.stage = 'complete'`.

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| mediapipe 0.10.14 needs `libGL` in Docker | Add `libgl1-mesa-glx` (or `libgl1`) to the Dockerfile `apt-get install` line |
| mediapipe `FaceLandmarker` may not be present on all platforms | Wrap all mediapipe calls in `try/except AttributeError` + `try/except ImportError`; fall back to YuNet silently |
| `cv2.FaceDetectorYN` requires a model XML file bundled with opencv-python-headless | Use `cv2.data.haarcascades` Haar cascade as tertiary fallback if YuNet XML is absent |
| BiRefNet model download on first request can take 60–120 s | Pre-warm sessions in `/models/status` startup handler; expose readiness there |
| MediaPipe `.task` file download at startup needs internet access inside container | Check for file existence first; skip download if present; document that the model cache volume must be pre-populated for air-gapped deployments |
| `asyncio.Semaphore` in `batch_pipeline.py` must live in the event loop where it's created | Create semaphore inside `process_many()`, not at class level |
| opencv 4.10.0.84 + mediapipe 0.10.14 may conflict at the C++ level | mediapipe bundles its own OpenCV; using `opencv-python-headless` separately is fine as long as the mediapipe Python Tasks API is used (not the Solutions API, which requires matching OpenCV) |
| `rembg` session caching is not thread-safe for parallel requests | Gate `new_session()` calls with a module-level `threading.Lock()` in both provider classes |
| Processor Dockerfile currently copies only `app.py`; sub-packages will not exist in image | Change `COPY app.py .` → `COPY . .` |
| Node `FileRecord` Redis hash deserialisation | New fields (`extractedPath`, `confirmedCorners`, `stage`) are all typed as `string | undefined`; `redis.hgetall` returns only keys that exist, so old hashes are safe |

---

## Frontend UI Structure (`public/`)

### index.html
- Top of `<body>`: `<script>if(location.search.includes('view=batches')){document.documentElement.classList.add('legacy-mode')}</script>`
- Two `<main>` sections: `id="snapid-app"` (default) and `id="batch-app"` (hidden unless `legacy-mode`).
- Step progress bar: `<nav id="steps">` with 5 `<button data-step="N">` elements.
- Sections: `#step-upload`, `#step-extract`, `#step-correct`, `#step-edit`, `#step-export`.

### app.js — SnapID workflow state machine
```
State per file:
  { id, sourceName, inputPath, file (File object),
    state: 'idle'|'uploading'|'extracting'|'extraction_review'|
            'correcting'|'correction_review'|'complete'|'error',
    extractedPath, corners, outputPath, previewUrl, error }

Global step: 1..5
```
- Step 1: drag-drop / file picker → `uploadFile()` creates a single-file batch, uploads, stores file record.
- Step 2: calls `POST /api/files/:id/extract` → shows detected corners overlaid on canvas; user can drag corner handles; `POST /api/files/:id/confirm-crop` when satisfied.
- Step 3: calls `POST /api/files/:id/correct` → shows progress spinners per sub-step; polls until complete.
- Step 4: shows before/after comparison; background selector; crop adjustment sliders.
- Step 5: download button per file; batch ZIP.

### Canvas corner editor
```js
class CornerEditor {
  constructor(canvas, imageUrl, initialCorners) { ... }
  // Draws image + 4 draggable handles + connecting lines
  // onchange(corners) callback fires when handles move
  getCorners() { return this._corners; }  // [[x,y],...]
  destroy() { ... }
}
```

### style.css
- CSS custom properties for brand colours: `--color-primary: #16697a`, `--color-surface: #fff`.
- Step nav: horizontal pill tabs, active state highlighted.
- Canvas overlay: `position: absolute; pointer-events: none` for lines; corner handles are absolutely-positioned `<div>` elements with `touch-action: none`.
- Legacy mode (`html.legacy-mode`): `#snapid-app { display: none }` and `#batch-app { display: block }`.

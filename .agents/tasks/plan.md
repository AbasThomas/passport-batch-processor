# Implementation Plan — SnapID Printed-Photo Extraction & Passport Correction Pipeline

## Architecture Decisions

**Decision 1 — Python sub-package structure under processor/**
The existing `processor/app.py` is a single flat file. All new code goes in sub-packages (`photo_extraction/`, `face_processing/`, `background_removal/`, `enhancement/`, `pipeline/`) with `__init__.py` files. `app.py` keeps all its existing logic and adds imports from these packages. This avoids touching the working `/process` endpoint while enabling the new `/process/v2` family. Relative imports (`from ..models import X`) are used within sub-packages; absolute imports (`from photo_extraction.boundary_detector import X`) are used from `app.py` (since uvicorn runs from `/app`).

**Decision 2 — Extraction confidence threshold = 0.5**
Below 0.5 the UI shows the manual corner selector. At or above 0.5 the result is accepted automatically. This threshold is a named constant in `pipeline.py` so it can be tuned without touching the detector logic.

**Decision 3 — MediaPipe legacy API**
Use `mp.solutions.face_detection` and `mp.solutions.face_mesh` (not the newer `mediapipe.tasks` Tasks API). These are available in mediapipe 0.10.14 and avoid the extra `.task` model file download complexity.

**Decision 4 — Two-stage API for multi-step workflow**
`/process/extract-only` → user reviews/confirms → `/process/confirm-crop` → `/process/correct`. The extracted image is saved to `WORK_DIR` (env var, default `/tmp/snapid_work`) so it persists between HTTP calls. The Node.js server stores `extractedPath` and `confirmedCorners` in Redis via the existing `saveRecord()` pattern.

**Decision 5 — Legacy worker path preserved**
The BullMQ worker keeps the `/process` call path for batches without `useV2: true`. New batches created by the SnapID UI set `useV2: true` in settings, which routes to `/process/v2`. This is a non-breaking addition.

**Decision 6 — Frontend is vanilla JS, no build step**
`public/app.js` is a single file with clearly commented sections. The legacy dashboard lives inside `#legacy-app` (hidden by default). A top-level `mode` variable and `setMode()` function switch between views. No bundler is introduced.

---

## Complete File List (28 files)

### New Python files
| File | Role |
|---|---|
| `processor/models.py` | All 12 Pydantic v2 BaseModel classes |
| `processor/__init__.py` | Makes processor/ a Python package |
| `processor/photo_extraction/__init__.py` | Package marker |
| `processor/photo_extraction/boundary_detector.py` | Primary contour-based quad detection |
| `processor/photo_extraction/quadrilateral_detector.py` | Fallback Hough-line quad detection |
| `processor/photo_extraction/perspective_corrector.py` | cv2.getPerspectiveTransform + warpPerspective |
| `processor/photo_extraction/extraction_validator.py` | Post-extraction sanity checks |
| `processor/photo_extraction/manual_crop.py` | Server-side handler for user-confirmed corners |
| `processor/face_processing/__init__.py` | Package marker |
| `processor/face_processing/face_detector.py` | MediaPipe primary + YuNet fallback |
| `processor/face_processing/landmark_detector.py` | FaceMesh 468-point landmarks |
| `processor/face_processing/face_aligner.py` | Roll correction via eye line angle |
| `processor/face_processing/crop_calculator.py` | Head-proportional passport crop |
| `processor/background_removal/__init__.py` | Package marker |
| `processor/background_removal/segmenter.py` | rembg + BiRefNet/U2Net fallback |
| `processor/background_removal/background_replacer.py` | Alpha compositing + color parsing |
| `processor/background_removal/mask_refiner.py` | Morphological mask cleanup + feathering |
| `processor/enhancement/__init__.py` | Package marker |
| `processor/enhancement/enhancer.py` | Denoise + PIL enhance (mirrors existing enhance()) |
| `processor/enhancement/quality_checker.py` | Laplacian blur score + resolution check |
| `processor/pipeline/__init__.py` | Package marker |
| `processor/pipeline/pipeline.py` | PassportPipeline orchestrator (full + partial modes) |
| `processor/pipeline/stage_runner.py` | Timed stage wrapper with error capture |

### Modified Python files
| File | Change |
|---|---|
| `processor/requirements.txt` | Add mediapipe==0.10.14, scipy==1.14.1 |
| `processor/Dockerfile` | Add libgl1; change COPY app.py → COPY . . |
| `processor/app.py` | Keep existing endpoints; add 5 new endpoints + WORK_DIR constant |

### Modified Node.js files
| File | Change |
|---|---|
| `src/lib.ts` | Add extractedPath?, confirmedCorners?, stage?, workDir? to FileRecord |
| `src/worker.ts` | Add /process/v2 branch when settings.useV2 is true |
| `src/server.ts` | Add /api/files/:id/extract, confirm-crop, correct, preview routes + processorUrl + workDirFor() |

### Rebuilt frontend files
| File | Change |
|---|---|
| `public/index.html` | Full rewrite: #snapid-app (5-step) + #legacy-app (old dashboard verbatim) |
| `public/app.js` | Full rewrite: legacy code + SnapID state machine + CornerSelector |
| `public/style.css` | Append new SnapID rules; keep all existing rules |

### Infrastructure
| File | Change |
|---|---|
| `docker-compose.yml` | Add WORK_DIR=/data/work to processor environment |

---

## Pydantic Models (processor/models.py)

```python
from pydantic import BaseModel
from typing import Optional

class ExtractionResult(BaseModel):
    ok: bool
    detected: bool
    confidence: float
    corners_px: Optional[list[list[float]]] = None
    extracted_path: Optional[str] = None
    warning: Optional[str] = None

class FaceResult(BaseModel):
    ok: bool
    confidence: float
    bbox: list[int]          # [x, y, w, h]
    landmarks: list[list[float]]   # [[x,y], ...] pixel coords
    roll_angle: float
    warning: Optional[str] = None

class AlignmentResult(BaseModel):
    ok: bool
    aligned_path: str
    roll_corrected: float
    crop_box: list[int]      # [x, y, w, h] before final crop

class SegmentationResult(BaseModel):
    ok: bool
    masked_path: str
    coverage: float
    warning: Optional[str] = None

class EnhancementResult(BaseModel):
    ok: bool
    enhanced_path: str
    notes: list[str]

class CropResult(BaseModel):
    ok: bool
    output_path: str
    width: int
    height: int

class ValidationResult(BaseModel):
    ok: bool
    checks: dict[str, bool]
    error: Optional[str] = None

class ProcessV2Request(BaseModel):
    input_path: str
    output_path: str
    settings: dict
    extracted_path: Optional[str] = None
    confirmed_corners: Optional[list[list[float]]] = None

class ProcessV2Response(BaseModel):
    ok: bool
    manual_review: bool
    detected_photo: bool
    stages: dict[str, str]   # stage_name → 'ok'|'skipped'|'failed: ...'
    error: Optional[str] = None
    output_path: Optional[str] = None

class ExtractOnlyRequest(BaseModel):
    input_path: str
    work_dir: str

class ExtractOnlyResponse(BaseModel):
    ok: bool
    confidence: float
    extracted_path: Optional[str] = None
    corners_px: Optional[list[list[float]]] = None
    preview_path: Optional[str] = None
    warning: Optional[str] = None

class ConfirmCropRequest(BaseModel):
    input_path: str
    corners_px: list[list[float]]
    work_dir: str

class ConfirmCropResponse(BaseModel):
    ok: bool
    extracted_path: str
    preview_path: str

class CorrectRequest(BaseModel):
    extracted_path: str
    output_path: str
    settings: dict

class CorrectResponse(BaseModel):
    ok: bool
    manual_review: bool
    stages: dict[str, str]
    error: Optional[str] = None
```

---

## FastAPI Endpoint Signatures (processor/app.py additions)

```python
# Keep unchanged:
@app.get('/health')
def health(): ...

@app.post('/process')
def process(req: Request): ...

# New endpoints:
@app.post('/process/v2', response_model=ProcessV2Response)
def process_v2(req: ProcessV2Request): ...

@app.post('/process/extract-only', response_model=ExtractOnlyResponse)
def extract_only(req: ExtractOnlyRequest): ...

@app.post('/process/confirm-crop', response_model=ConfirmCropResponse)
def confirm_crop(req: ConfirmCropRequest): ...

@app.post('/process/correct', response_model=CorrectResponse)
def correct(req: CorrectRequest): ...

@app.get('/models/status')
def models_status(): ...
# Returns: {"birefnet-general": true/false, "mediapipe_face": true/false}
```

---

## Node Worker State Machine

The worker reads `batch.settings.useV2` (parsed from JSON). When `true`:

1. Calls `POST /process/v2` with `{ input_path, output_path, settings, extracted_path, confirmed_corners }`.
2. Response maps as:
   - `result.ok === true` → `x.state = 'completed'`
   - `result.ok === false && result.manual_review === true` → `x.state = 'needs_review'`
   - `result.ok === false && result.manual_review === false` → `x.state = 'failed'`
3. `x.stage = JSON.stringify(result.stages)` always saved.
4. `x.outputPath = result.output_path` if present.
5. Fallback (no `useV2`): unchanged — calls `POST /process` with existing logic.

This means the BullMQ queue is used only for full-pipeline batch processing. The step-by-step SnapID flow (extract → confirm → correct) uses direct HTTP calls from the Node server (not the worker queue) because each step requires user interaction between stages.

---

## Frontend Architecture

### State machine
```
mode: 'snapid' | 'legacy'
  snapid → step: 0..4
    0 = Upload
    1 = Extract   (per-file loop)
    2 = Correct   (per-file loop)
    3 = Edit      (per-file loop)
    4 = Export
```

### CornerSelector class (canvas overlay)
- Constructor: takes `canvas` element + `img` element.
- `setCorners([[x,y],[x,y],[x,y],[x,y]])`: stores corners in image pixel coords; redraws.
- `getCorners()`: returns corners in image pixel coords (scaled from display).
- `draw()`: clears canvas; draws dashed quadrilateral connecting corners; draws 4 draggable circles (r=8, fill=#16697a).
- Event handlers: `mousedown` identifies nearest corner within 16px hit radius; `mousemove` updates that corner's position and redraws; `mouseup` clears active corner.
- Scale factor: `scaleX = img.naturalWidth / img.clientWidth`, same for Y.

### Correction progress (Step 3)
- Stage list: `['face_detection', 'alignment', 'background_removal', 'enhancement', 'crop']`.
- Polls `GET /api/batches/:id` every 1500ms.
- Maps `file.stage` (JSON string of stages dict) to icon per stage: `pending → ○`, `ok → ✓`, `failed → ✗`, absent → `○`.
- When file.state is `completed` or `failed`, stops polling for that file.

### Background selection (Step 4)
- Four swatches: `#ffffff` white, `#e8e8e8` light grey, `#c8d9e8` light blue, custom color input.
- Clicking a swatch calls `POST /api/files/:id/correct` with updated `background` in settings.
- Preview canvas applies the corrected image on response.

---

## Import Structure (Python)

Within each sub-package, use **relative imports**:
```python
# In pipeline/pipeline.py:
from ..models import ProcessV2Request, ProcessV2Response, ExtractionResult
from ..photo_extraction.boundary_detector import BoundaryDetector
from ..photo_extraction.quadrilateral_detector import QuadrilateralDetector
from ..photo_extraction.perspective_corrector import PerspectiveCorrector
from ..photo_extraction.extraction_validator import ExtractionValidator
from ..photo_extraction.manual_crop import ManualCrop
from ..face_processing.face_detector import FaceDetector
from ..face_processing.landmark_detector import LandmarkDetector
from ..face_processing.face_aligner import FaceAligner
from ..face_processing.crop_calculator import CropCalculator
from ..background_removal.segmenter import Segmenter
from ..background_removal.mask_refiner import MaskRefiner
from ..background_removal.background_replacer import BackgroundReplacer
from ..enhancement.enhancer import Enhancer
from ..enhancement.quality_checker import QualityChecker
from .stage_runner import StageRunner
```

From `app.py` (which runs as `__main__` under uvicorn from `/app`), use **absolute imports**:
```python
from models import (ProcessV2Request, ProcessV2Response, ExtractOnlyRequest,
                    ExtractOnlyResponse, ConfirmCropRequest, ConfirmCropResponse,
                    CorrectRequest, CorrectResponse)
from pipeline.pipeline import PassportPipeline
```

---

## Ordered Implementation Steps

- [ ] 1. **Update processor/requirements.txt** — add mediapipe==0.10.14 and scipy==1.14.1.
        Files: `processor/requirements.txt`
        Verify: file contains both new dependencies.

- [ ] 2. **Update processor/Dockerfile** — add libgl1 to apt-get install; change `COPY app.py .` to `COPY . .`.
        Files: `processor/Dockerfile`
        Verify: `docker-compose build processor` exits 0.

- [ ] 3. **Create processor/models.py** — all 12 Pydantic BaseModel classes as specified above.
        Files: `processor/models.py`
        Verify: `docker-compose exec processor python -c "from models import ExtractionResult, ProcessV2Request; print('OK')"`.

- [ ] 4. **Create photo_extraction/ sub-package** — `__init__.py`, `boundary_detector.py`, `quadrilateral_detector.py`, `perspective_corrector.py`, `extraction_validator.py`, `manual_crop.py`.
        Files: 6 files under `processor/photo_extraction/`
        Verify: `docker-compose exec processor python -c "from photo_extraction.boundary_detector import BoundaryDetector; print('OK')"`.

- [ ] 5. **Create face_processing/ sub-package** — `__init__.py`, `face_detector.py`, `landmark_detector.py`, `face_aligner.py`, `crop_calculator.py`.
        Files: 5 files under `processor/face_processing/`
        Verify: `docker-compose exec processor python -c "from face_processing.face_detector import FaceDetector; print('OK')"`.

- [ ] 6. **Create background_removal/ sub-package** — `__init__.py`, `segmenter.py`, `background_replacer.py`, `mask_refiner.py`.
        Files: 4 files under `processor/background_removal/`
        Verify: `docker-compose exec processor python -c "from background_removal.segmenter import Segmenter; print('OK')"`.

- [ ] 7. **Create enhancement/ sub-package** — `__init__.py`, `enhancer.py`, `quality_checker.py`.
        Files: 3 files under `processor/enhancement/`
        Verify: `docker-compose exec processor python -c "from enhancement.enhancer import Enhancer; print('OK')"`.

- [ ] 8. **Create pipeline/ sub-package** — `__init__.py`, `stage_runner.py`, `pipeline.py` with `PassportPipeline` class.
        Files: 3 files under `processor/pipeline/`
        Verify: `docker-compose exec processor python -c "from pipeline.pipeline import PassportPipeline; print('OK')"`.

- [ ] 9. **Update processor/app.py** — keep /health and /process exactly; add WORK_DIR constant; add /process/v2, /process/extract-only, /process/confirm-crop, /process/correct, /models/status endpoints.
        Files: `processor/app.py`
        Verify: `docker-compose exec processor python -c "import app; print('app OK')"`. Also `curl http://localhost:8000/health` returns `{"ok":true,...}`.

- [ ] 10. **Create processor/__init__.py** — empty file.
         Files: `processor/__init__.py`
         Verify: file exists.

- [ ] 11. **Update docker-compose.yml** — add `WORK_DIR=/data/work` to processor environment.
         Files: `docker-compose.yml`
         Verify: `docker-compose config` shows WORK_DIR in processor environment.

- [ ] 12. **Update src/lib.ts** — add `extractedPath?`, `confirmedCorners?`, `stage?`, `workDir?` optional fields to `FileRecord` type.
         Files: `src/lib.ts`
         Verify: `npm run build` exits 0.

- [ ] 13. **Update src/worker.ts** — add /process/v2 branch gated on `settings.useV2`.
         Files: `src/worker.ts`
         Verify: `npm run build` exits 0.

- [ ] 14. **Update src/server.ts** — add `processorUrl` constant, `workDirFor()` helper, and four new routes.
         Files: `src/server.ts`
         Verify: `npm run build` exits 0 with no TypeScript errors.

- [ ] 15. **Rebuild public/index.html** — two-section SPA: #snapid-app (5-step) + #legacy-app (old content verbatim).
         Files: `public/index.html`
         Verify: HTML is valid; both `#snapid-app` and `#legacy-app` elements exist; all original element IDs present in #legacy-app.

- [ ] 16. **Rebuild public/app.js** — full rewrite with legacy section, SnapID state machine, CornerSelector class, all step handlers.
         Files: `public/app.js`
         Verify: `npm run build` exits 0 (Node side compiles; confirms no broken imports). Browser console shows no errors on page load.

- [ ] 17. **Append to public/style.css** — new SnapID rules; keep all existing rules.
         Files: `public/style.css`
         Verify: `npm run build` exits 0. Browser: legacy dashboard still renders correctly; SnapID steps are visually distinct.

---

## Verification Summary

Four Python import checks (run from inside the processor container after `docker-compose build`):
```bash
python -c "from photo_extraction.boundary_detector import BoundaryDetector; print('1 OK')"
python -c "from face_processing.face_detector import FaceDetector; print('2 OK')"
python -c "from background_removal.segmenter import Segmenter; print('3 OK')"
python -c "from enhancement.enhancer import Enhancer; print('4 OK')"
python -c "from pipeline.pipeline import PassportPipeline; print('5 OK')"
```

TypeScript compile check:
```bash
npm run build
# Expected: exit 0, zero errors, dist/ populated
```

Full smoke test (Docker Compose up):
```bash
docker-compose up -d
curl http://localhost:8000/health       # {"ok":true,...}
curl http://localhost:3000/             # HTML with SnapID UI
curl http://localhost:3000/api/batches  # 405 (no GET, confirms route exists)
```

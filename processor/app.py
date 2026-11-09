"""FastAPI front for the local passport-correction pipeline.

All inference (YuNet face detection, BiRefNet/U2Net segmentation) runs locally
via ONNX Runtime; no image data leaves the host.
"""
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pipeline.pipeline_orchestrator import analyze, process, warmup
from pipeline.passport_pipeline import Review
from pipeline.batch_pipeline import run_batch

app = FastAPI(title='local-passport-processor')
warmup()


class Request(BaseModel):
    input_path: str
    output_path: str
    settings: dict = {}


class BatchItem(BaseModel):
    input_path: str
    output_path: str
    settings: dict = {}


class BatchRequest(BaseModel):
    items: list[BatchItem]
    max_workers: int = 2


class AnalyzeRequest(BaseModel):
    input_path: str


@app.on_event('startup')
def _startup():
    warmup()


@app.get('/health')
def health():
    return {'ok': True, 'models': {'face': 'yunet-2023mar',
                                   'segment': os.getenv('SEGMENTATION_MODEL', 'auto')}}


@app.post('/process')
def process_endpoint(req: Request):
    """Single-image pipeline. Contract kept for the BullMQ worker:
    {ok, manualReview, error} plus richer stage/result detail."""
    try:
        return process(req.input_path, req.output_path, req.settings or {})
    except Review as e:
        return {'ok': False, 'manualReview': True, 'error': str(e),
                'extra': getattr(e, 'extra', {})}
    except HTTPException:
        raise
    except Exception as e:
        return {'ok': False, 'manualReview': True,
                'error': f'Processing error: {str(e)[:250]}'}


@app.post('/analyze')
def analyze_endpoint(req: AnalyzeRequest):
    """Preview stage for the UI: candidate photo boundaries, confidence,
    extracted-photo preview (base64 JPEG), and detected faces — no export."""
    try:
        return analyze(req.input_path)
    except Review as e:
        raise HTTPException(422, str(e))


@app.post('/process-batch')
def process_batch_endpoint(req: BatchRequest):
    items = [{'input_path': i.input_path, 'output_path': i.output_path,
              'settings': i.settings or {}} for i in req.items]
    return {'results': run_batch(items, max_workers=min(req.max_workers, 8))}

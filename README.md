# Passport batch processor

Production-oriented bulk passport image processing: streaming multipart/ZIP ingestion, persistent source/output mapping, BullMQ per-image jobs, independent retry/recovery, configurable local Sharp processing, and streaming ZIP export.

## Run

1. Copy `.env.example` to `.env` and set your storage/Redis settings.
2. `npm install`
3. Start Redis (`docker compose up redis -d`), then `npm run dev` and `npm run worker`.

For a container deployment: `docker compose up --build`. Use a persistent mounted volume or object-store adapter for `DATA_DIR`; Redis must use persistence. Configure `WORKER_CONCURRENCY` and `GPU_CONCURRENCY` to match CPU/GPU memory. The worker is deliberately separate from the web process and resumes queued BullMQ jobs after a restart.

## API workflow

`POST /api/batches` → `POST /api/batches/:id/upload` (multipart field(s), repeatable) or `/upload-zip` → `PATCH /api/batches/:id/settings` → `POST /api/batches/:id/start` → poll `GET /api/batches/:id` → `GET /api/batches/:id/export`.

The upload endpoint streams each part straight to disk and supports up to `MAX_BATCH_FILES` (default 2,000), not an arbitrary UI limit. Original source names, internal UUIDs, collision-resolved output names, state and retry history are persisted independently. ZIP reports every source/output/status and excludes failures.

HEIC decoding depends on the libvips build packaged with Sharp. A deployment using an unsupported HEIC build reports that one file as failed without stopping the batch.

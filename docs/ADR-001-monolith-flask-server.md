# ADR-001: Monolithic Flask Server Over Microservices

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-07 |
| **Decision** | Keep all ML pipelines in a single Flask process |
| **Deciders** | RealEyes Engineering Team |

---

## Context

RealEyes runs four ML pipelines (image classification, anomaly segmentation, text fact-checking, video analysis) that share a CUDA GPU. The question is whether to deploy them as separate microservices or keep them in a single process.

## Decision

We chose a **monolithic Flask server** (`server.py`) that hosts all endpoints and manages all model lifecycles in a single process.

## Rationale

### 1. GPU Memory Sharing

The three GPU-resident models consume approximately:
- Swin-V2-B classifier: ~1.4 GB VRAM
- AXUNet segmenter: ~1.5 GB VRAM
- DeBERTa NLI: ~0.5 GB VRAM

**Total: ~3.4 GB VRAM in a single process.** If split into separate microservices, each process would load its own CUDA context (~300 MB overhead each) and the models could not share VRAM efficiently. On consumer GPUs (8 GB), this leaves no headroom for inference tensors.

### 2. Model Loading Cost

Each model takes 5-15 seconds to load from disk. The lazy singleton pattern (load once, reuse forever) means this cost is paid exactly once at startup. In a microservices architecture, each service restart would independently pay this cost, and inter-service HTTP calls would add latency to the screenshot pipeline (which chains classification → segmentation → fact-checking).

### 3. Thread-Lock Serialization is Sufficient

The system serves a single user (local deployment). Flask's `threaded=True` mode with fine-grained `threading.Lock()` on GPU forward passes is sufficient to prevent CUDA race conditions. The I/O-bound work (web scraping, image encoding, report generation) overlaps freely with GPU work.

### 4. Deployment Simplicity

A single `py server.py` command starts everything. No container orchestration, service discovery, or health-check coordination needed for the target deployment environment (a student's laptop or desktop).

## Consequences

### Positive
- Single process to start, monitor, and debug.
- Minimal VRAM overhead.
- No inter-service latency for multi-pipeline endpoints like `/process-screenshot`.
- Simple `requirements.txt` — one install for all dependencies.

### Negative
- `server.py` is 1,150 lines. Large file, harder to navigate.
- All models must fit in the same GPU's VRAM simultaneously.
- A crash in any pipeline takes down all endpoints.
- Horizontal scaling (multiple GPU workers) would require refactoring.

### Mitigation
- The file is well-sectioned with clear comment headers.
- GPU OOM is mitigated by inference locks (only one model runs at a time).
- For production deployment at scale, a future ADR would address splitting into gRPC services with model serving (e.g., TorchServe, Triton).

## Alternatives Considered

| Alternative | Why Rejected |
|---|---|
| **Separate Flask services per pipeline** | CUDA context duplication, 8 GB GPU limit exceeded |
| **FastAPI + async** | `torch.no_grad()` blocks the event loop; threading is simpler for GPU work |
| **TorchServe / Triton** | Overkill for local single-user deployment; adds infrastructure complexity |
| **Celery task queue** | Adds Redis/RabbitMQ dependency; the task-based async pattern in `/verify-text-start` achieves the same goal with stdlib threading |

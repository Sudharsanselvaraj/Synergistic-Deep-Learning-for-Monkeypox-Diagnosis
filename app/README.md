# app/

Deployment surface for Tri-Net v2 (planned).

- `api/` — inference service (FastAPI) wrapping `trinet` for image + symptom prediction
- `web/` — minimal front end for uploading a lesion image / entering symptoms
- `docker/` — containerization for reproducible serving

The trained champion checkpoint (`outputs/checkpoints/fusion.keras`) is the intended model to
serve. This is a research artifact, **not** a certified medical device — see `SECURITY.md`.

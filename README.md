# Staywise — NYC Airbnb Room Type Predictor

A full-stack research application for the existing NYC Airbnb room-type classifier.

## Stack
- Frontend: React, Vite, TypeScript, Tailwind CSS, Lucide icons.
- Backend: FastAPI, Pydantic, scikit-learn and Joblib.
- Models: two versioned v3 model bundles in `ml/artifacts/`.
- Research: original and model-development notebooks in `notebooks/`.
- Data: `data/AB_NYC_2019.csv`.

## Run the backend
Use Python 3.13 with the exact versions in `backend/requirements.txt`.

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# WSL/macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

API docs: http://127.0.0.1:8000/docs

## Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend: http://localhost:5173

Optional `frontend/.env`:
`VITE_API_BASE_URL=http://127.0.0.1:8000`

## Tests
```bash
cd backend
pytest -q
```
```bash
cd frontend
npm run build
```

## Model behavior
- `without_reviews`: 8 features.
- `with_reviews`: 10 features.
- Target classes: Entire home/apt, Private room, Shared room.
- The frontend displays the actual response from FastAPI. Its two-second minimum presentation time is only an animation/UX choice; it does not manufacture or alter model outputs.
- The selected building glow represents the predicted class. It is a visualization, not an additional model step.

The uploaded standalone `ml/source_artifacts/airbnb_room_type_with_reviews.pkl` is preserved for provenance. The API uses the versioned v3 bundle because it contains the estimator plus the feature contract, class order and metadata.

## Limitations
The model is trained on a 2019 cross-sectional NYC snapshot. Current host-grouped holdout metrics are not external/temporal validation. Shared-room recall is about 28.8%. Nested grouped CV, independent temporal validation, point-in-time feature availability, product acceptance thresholds and live monitoring remain open gates. Do not describe this model as production-validated or its prediction as guaranteed.

# Backend API

## Setup (Python 3.13 recommended)

```bash
cd backend
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

From the `backend/` directory:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

API docs: http://127.0.0.1:8000/docs

## Test

```bash
pytest -q
```

The model artifacts are loaded once at application startup. Do not accept or load user-supplied Joblib files.

import os
import uuid
import shutil
import json
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from ga_module import load_and_preprocess_dataset, run_genetic_algorithm

from lasso_expert import apply_lasso_feature_selection
from chi_square import ChiSquareSelector
from pca import PCASelector
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import asyncio
import numpy as np
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles




app = FastAPI(title="Feature Selection API (GA + Traditional)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STORAGE_DIR = os.path.join(os.getcwd(), "uploads")
RESULTS_DIR = os.path.join(os.getcwd(), "results")
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

JOBS = {} 


class RunParams(BaseModel):
    target: Optional[str] = None
    pop_size: int = 50
    generations: int = 15
    mut_prob: float = 0.01
    crossover_prob: float = 0.8
    alpha: float = 0.9
    test_size: float = 0.3
    random_seed: int = 42
    return_history: bool = True


def save_uploaded_file(tmp_path, dest_path):
    shutil.move(tmp_path, dest_path)


def get_latest_uploaded_file():
    files = sorted(
        [os.path.join(STORAGE_DIR, p) for p in os.listdir(STORAGE_DIR)],
        key=os.path.getmtime,
        reverse=True,
    )
    if not files:
        raise HTTPException(status_code=400, detail="No uploaded CSV found.")
    return files[0]
    
app.mount("/static", StaticFiles(directory=os.getcwd()), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files allowed.")
    uid = str(uuid.uuid4())
    save_path = os.path.join(STORAGE_DIR, f"{uid}_{file.filename}")
    tmp_file = save_path + ".tmp"
    with open(tmp_file, "wb") as f:
        content = await file.read()
        f.write(content)
    shutil.move(tmp_file, save_path)
    return {"status": "ok", "file_id": uid, "filename": file.filename, "path": save_path}


@app.post("/run")
async def run_ga(params: RunParams):
    file_path = get_latest_uploaded_file()

    try:
        prep = load_and_preprocess_dataset(
            file_path,
            target_column=params.target,
            test_size=params.test_size,
            random_seed=params.random_seed,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {e}")

    X_train, X_test = prep["X_train"], prep["X_test"]
    y_train, y_test = prep["y_train"], prep["y_test"]
    feature_names = prep["feature_names"]

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "running", "progress": [], "result": None}

    def progress_cb(msg):
        JOBS[job_id]["progress"].append(msg)

    result = run_genetic_algorithm(
        X_train, X_test, y_train, y_test,
        population_size=params.pop_size,
        num_generations=params.generations,
        mutation_probability=params.mut_prob,
        crossover_probability=params.crossover_prob,
        alpha=params.alpha,
        return_history=params.return_history,
        progress_callback=progress_cb,
    )

    best_chrom = result["best_chromosome"]
    selected_indices = []
    selected_names = []
    if best_chrom is not None:
        selected_indices = list(map(int, list((best_chrom == 1).nonzero()[0])))
        selected_names = [feature_names[i] for i in selected_indices]

    output_filename = f"selected_features_{job_id}.txt"
    output_path = os.path.join(RESULTS_DIR, output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        for name in selected_names:
            f.write(str(name) + "\n")

    final_payload = {
        "job_id": job_id,
        "status": "finished",
        "best_fitness": result["best_fitness"],
        "best_accuracy": result["best_accuracy"],
        "selected_count": len(selected_names),
        "selected_indices": selected_indices,
        "selected_feature_names": selected_names,
        "history": result["history"],
    }

    JOBS[job_id]["status"] = "finished"
    JOBS[job_id]["result"] = final_payload
    JOBS[job_id]["output_path"] = output_path
    return JSONResponse(final_payload)


@app.post("/run_traditional")
async def run_traditional(params: RunParams, method: str = "lasso"):
    """
    Run a traditional feature selection method: 'lasso', 'pca', or 'chi_square'
    """
    file_path = get_latest_uploaded_file()
    df = pd.read_csv(file_path)
    target_column = params.target
    if target_column is None or target_column not in df.columns:
        raise HTTPException(status_code=400, detail="Invalid or missing target column.")

    result = {}
    try:
        if method == "lasso":
            selected, acc = apply_lasso_feature_selection(df, target_column)
            result = {"method": "lasso", "accuracy": acc, "selected_features": selected}

        else:
            X = df.drop(
                columns=[target_column, "Name", "Ticket", "Cabin", "PassengerId"],
                errors="ignore",
            )
            y = df[target_column]
            for col in X.columns:
                if X[col].dtype in ["int64", "float64"]:
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(X[col].mode()[0])
            X = pd.get_dummies(X, drop_first=True)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=params.test_size, random_state=params.random_seed
            )

            if method == "pca":
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)
                try:
                    pca = PCASelector(variance_threshold=0.95)
                    X_train_pca = pca.fit_transform(X_train_s)
                    X_test_pca = pca.transform(X_test_s)
                except Exception:
                    from sklearn.decomposition import PCA
                    n_comp = min(X_train_s.shape[0], X_train_s.shape[1], 5)
                    pca_model = PCA(n_components=n_comp)
                    X_train_pca = pca_model.fit_transform(X_train_s)
                    X_test_pca = pca_model.transform(X_test_s)

                model = LogisticRegression(max_iter=1000).fit(X_train_pca, y_train)
                acc = accuracy_score(y_test, model.predict(X_test_pca))
                result = {
                    "method": "pca",
                    "components": getattr(pca, "get_num_components", lambda: n_comp)(),
                    "accuracy": acc,
            }


            elif method == "chi_square":
                chi = ChiSquareSelector(k_features=10)
                X_train_sel = chi.fit_transform(X_train, y_train)
                X_test_sel = chi.transform(X_test)
                model = LogisticRegression(max_iter=1000).fit(X_train_sel, y_train)
                acc = accuracy_score(y_test, model.predict(X_test_sel))
                result = {
                    "method": "chi_square",
                    "selected_features": chi.get_selected_features_names(),
                    "accuracy": acc,
                }

            else:
                raise ValueError("Unknown method. Choose: lasso, pca, chi_square.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running {method}: {e}")

    return JSONResponse(json.loads(json.dumps(result, default=lambda o: o.item() if isinstance(o, (np.integer, np.floating)) else o)))



@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/download/{job_id}")
async def download_result(job_id: str):
    job = JOBS.get(job_id)
    if not job or job.get("status") != "finished":
        raise HTTPException(status_code=404, detail="Result not ready")
    path = job.get("output_path")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Result file not found")
    return FileResponse(path, filename=os.path.basename(path), media_type="text/plain")


@app.get("/stream/{job_id}")
async def stream_progress(job_id: str):
    async def event_generator():
        last_len = 0
        while True:
            job = JOBS.get(job_id)
            if not job:
                yield "data: " + json.dumps({"error": "job not found"}) + "\n\n"
                return
            progress = job.get("progress", [])
            while last_len < len(progress):
                msg = progress[last_len]
                yield "data: " + json.dumps(msg) + "\n\n"
                last_len += 1
            if job.get("status") == "finished":
                yield "data: " + json.dumps({"status": "finished", "result": job.get("result")}) + "\n\n"
                return
            await asyncio.sleep(0.8)
    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


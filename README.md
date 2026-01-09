# Counter Cyber Crime Abusive Text Detection (Code-Mixed NLP)  

This repository implements a **production-ready MLOps pipeline** for detecting abusive text in code-mixed content (e.g., Hindi-English).  
It includes **data preprocessing, model training, model registry, FastAPI serving, and a simple frontend** for demonstration.  

---

## 🔹 Table of Contents

1. [Project Overview](#project-overview)  
2. [Folder Structure](#folder-structure)  
3. [Setup Environment](#setup-environment)  
4. [Data Preparation](#data-preparation)  
5. [Training the Model](#training-the-model)  
6. [Model Artifacts & Registry](#model-artifacts--registry)  
7. [Running the API](#running-the-api)  
8. [Frontend Demo](#frontend-demo)  
9. [Project Logs](#project-logs)  
10. [Future Steps / Dockerization](#future-steps--dockerization)  

---

## 🔹 Project Overview

- **Goal**: Detect abusive vs non-abusive text in code-mixed text.  
- **Model**: BiLSTM (TensorFlow) with embedding size 70.  
- **Preprocessing**:
  - Emoji normalization
  - URL, mentions, and abusive pattern replacement
  - Tokenization via `WhitespaceTokenizer`  
- **MLOps Features**:
  - Versioned model registry
  - Config-driven pipeline
  - Logging for training & inference
  - FastAPI endpoint
  - Frontend for demo

---
## 🔹 Folder Structure
project_root/
│
├── api/ # FastAPI code for serving
├── configs/ # YAML configs (training, logging)
├── data/raw/ # Raw CSV dataset
├── frontend/ # Simple HTML/CSS/JS UI
├── models/ # Model versions, registry.json
├── pipelines/ # Orchestration wrappers
├── scripts/ # CLI scripts for training / API
├── src/ # Core libraries
│ ├── common/ # Logger, utils
│ ├── data/ # Preprocessing, tokenizer, data loader
│ ├── inference/ # Model loader & predictor
│ └── training/ # train.py
├── requirements.txt # Python dependencies
└── README.md

---

## 🔹 Setup Environment

1. **Create conda environment**:

```bash
conda create -n abusive-nlp python=3.10 -y
conda activate abusive-nlp
```
2. **Install dependecies**:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
3. **NLTK setup**
```bash
python - <<EOF
import nltk
nltk.download('punkt')
```
4. **SETPYTHONPATH**
```bash
export PYTHONPATH=.
```
## 🔹 Data Prepration
- Download the dataset from https://github.com/ParrasTiwari/code_mix and keep it as data/raw/code_mix_abusive.csv

- CSV have two columns:
  - text → Input text
  - label → Binary: abusive / non-abusive
## 🔹 Training the model from scratch
- Entry point
  ```bash
  python src/training/train.py
  ```
- pipeline wrapper
  ```bash
  python pipelines/train_pipeline.py
  ```
- Logs: Saved to logs/training.log
- After training, the following will be saved under models/:
  - model/ → TensorFlow model
  - tokenizer.pkl → Tokenizer
  - metrics.json → Validation metrics
  - training_config.yaml → Training config snapshot
  - metadata.json → Model info
  - registry.json → Versioned registry
⚠️ We have not used optimized cutting-edge architecture in the train.py due to limited traininig resources. More architectures will be provided. We encourage community to contribute more effective architectures.
## 🔹 Model Artifact and registry
Example registry.json
```bash
{
  "latest": "v1",
  "models": {
    "v1": {
      "path": "models/v1",
      "val_accuracy": 0.85,
      "created_at": "2025-12-20T10:00:00"
    }
  }
}
```
- Allows rollbacks and multiple versions
- ModelLoader uses this for inference

## 🔹 Running the API
1. **Start FastAPI**
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
2. **Health check**
   ```bash
   http://localhost:8000/health
   ```
3. **Open Swagger docs**
   ```bash
   http://localhost:8000/docs
   ```
4. **Use /predict endpoint**
   ```bash
   curl -X POST http://localhost:8000/predict \ -H "Content-Type: application/json" \ -d '{"text":"you are useless"}'
  ``

## 🔹 Frontend Demo
```bash
cd frontend
python -m http.server 8080
```
Open in browser
```bash
http://localhost:8080
```
- Enter text → click Analyze
- Result:
  - Red → abusive
  - Green → non-abusive
  - Probability & latency displayed

⚠️ CORS is enabled in FastAPI middleware for local development

## 🔹 Project Logs

- Training logs: logs/training.log
- Inference logs: Printed to terminal (FastAPI)

## 🔹 Future Steps / Dockerization

- Dockerize API + Frontend
- Use docker-compose for unified start
- Add CI/CD pipelines for:
  - Automatic training
  - Model registry updates
  - Smoke tests
- Add monitoring / metrics (Prometheus + Grafana)
- Support multiple model versions in API



---

This README is **end-to-end**: setup, training, inference, frontend, logs, future MLOps steps.  

---

# END to END AI App for DA5402 Project Submission
### Roll No: DA24M010
### Name: Mohit Singh

# Surface Crack Detection AI App

This repository contains a full-stack AI application designed to detect cracks on surface images using deep learning. Built using MLOps principles, it supports containerization, model versioning, monitoring, feedback-based retraining, and seamless deployment.

## Features

- Upload and analyze surface images for cracks
- Deep learning-based crack classification
- Feedback loop for incorrect predictions
- Dockerized microservices (frontend, backend, DB, monitoring)
- Real-time monitoring via Grafana + Prometheus
- Model versioning with MLflow & DVC

---

## Tech Stack

- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Database**: PostgreSQL
- **Model Serving**: MLflow
- **Data Versioning and Model Training Pipelin**: DVC
- **Containerization**: Docker & Docker Compose
- **Monitoring**: Prometheus, Grafana

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/DA24M010/da5402_aiapp.git
cd da5402_aiapp
```

### 2. Install dependencies

```bash
pip install requirements.txt
```

### 3. Running model training pipeline

```bash
dvc repro
```

### 4. Serve the Model (Separate Terminal)

```bash
mlflow models serve -m "models:/surface_crack_detector/Production" --host 0.0.0.0 -p 5001 --no-conda
```

### 5. Launch Full App with Docker Compose

```bash
docker-compose up --build
```

Access the frontend at: [http://localhost:3000](http://localhost:3000)

---

## 📂 Project Structure

```bash
DA5402_AIAPP/
│
├── .dvc/                      # DVC metadata
├── artifacts/                 # Saved models, eda_reports, metric plots
├── data/                      # Raw and processed data
├── logs/                      # Training and app logs
├── mlruns/                    # MLflow tracking directory
├── services/                  # Microservices: backend, frontend, DB, monitoring
│   ├── backend/               # FastAPI backend
│   ├── frontend/              # React frontend
│   ├── grafana_data/          # Grafana persistent volume
│   ├── postgres_data/         # PostgreSQL volume
│   └── prometheus/            # Prometheus config
│
├── scripts/                   # Core ML & pipeline scripts
│   ├── ingestion_pipeline/
│   │   ├── data_eda.py             # Exploratory Data Analysis
│   │   ├── download_dataset.py     # Download images and labels
│   │   ├── dvc_utils.py            # DVC pipeline helpers
│   │   ├── feedback_handler.py     # Append flagged images from PG db to training set
│   │   └── run_ingest.py           # Orchestrates the data ingestion step
│   │
│   ├── pipeline/
│       ├── cnnmodel.py             # CNN architecture definition
│       ├── evaluate.py             # Evaluate model on test data
│       ├── load_dataset.py         # Load and format datasets
│       ├── preprocess_data.py      # Image transforms and preprocessing
│       ├── split_data.py           # Train-val-test split
│       ├── train.py                # Train the CNN model and handles mlflow logging and model registry
│       ├── utils.py                # Utility functions
│   ├── run_tests.py                # Runs model testing on test samples
│   ├── model_rebuild_scheduler.py  # Scheduled retraining logic
│   └── run_scheduler.py            # Orchestrates periodic retraining as a cronjob
│
├── params.yaml               # Training config (used by DVC)
├── requirements.txt          # requirements for running the training pipeline
├── docker-compose.yaml       # Docker service configuration
├── dvc.yaml                  # DVC pipeline stages
├── designDoc.pdf             # Design Documentation containing low and high level designes
├── manual.pdf                # User manual
└── readme.md
```

---

## How the App Works

1. **User uploads an image** via the UI.
2. The **backend** sends the image to the ML model (served via MLflow).
3. The **ML model** classifies the image as "Crack" or "No Crack."
4. The result is displayed to the user.
5. The user can **flag wrong predictions**, which are stored in PostgreSQL for feedback learning.
6. A scheduled pipeline retrains the model using these flagged samples.
7. The newly trained model registered on Mlflow as Production model is served using mlflow server automatically using the scheduler script.

---

## Monitoring Dashboards

- **Prometheus**: [http://localhost:9090](http://localhost:9090)
- **Grafana**: [http://localhost:3001](http://localhost:3001)
  - Default Admin Password: `mohit1234`

---

## Model Serving (MLflow)

ML model is served independently via:

```bash
mlflow models serve -m "models:/surface_crack_detector/Production" --host 0.0.0.0 -p 5001 --no-conda
```

The backend uses the `http://host.docker.internal:5001/invocations` endpoint to communicate with the ML model.

---

## Feedback Loop for Retraining

1. Users can flag incorrect predictions via the UI.
2. These are saved in the `feedback` table in PostgreSQL.
3. The retraining pipeline periodically pulls flagged data from PG db and updates the model.
4. Updated models are pushed to MLflow registry and served for prediction to the UI.

### Starting cronjob for model retraining using flagged data
```bash
python ./scripts/run_scheduler.py
```

#### *This runs every 5 minutes and the interval can be changed inside the script.*
---

## Authors & Acknowledgments

Developed by DA24M010  

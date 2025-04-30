import torch
import torch.nn as nn
from load_dataset import load_dataloaders
import mlflow
import mlflow.pytorch
import yaml
import json
import os
import logging
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from utils import load_params, plot_confusion_matrix

import numpy as np

# Setup logging
log_dir = "artifacts/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "evaluate.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=log_file,
    filemode="a"
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

def evaluate_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        params = load_params()
        eval_params = params["evaluate"]
        logging.info("Parameters loaded from config.")

        dataloaders = load_dataloaders("./artifacts/metadata/", test_batch_size=eval_params["test_batch_size"])
        test_loader = dataloaders["test"]
        logging.info("Test DataLoader loaded.")

        with open("./artifacts/run_id.txt", "r") as f:
            run_id = f.read().strip()
        logging.info(f"Using run ID: {run_id}")

        model_uri = f"runs:/{run_id}/model"
        model = mlflow.pytorch.load_model(model_uri).to(device)
        model.eval()
        logging.info("Model loaded from MLflow.")

        criterion = nn.BCELoss()
        test_loss = 0.0
        test_correct_preds = 0
        test_total_preds = 0
        all_test_labels = []
        all_test_preds = []

        with mlflow.start_run(run_id=run_id):
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs.view(-1), labels.float())
                    test_loss += loss.item() * inputs.size(0)

                    preds = (outputs.view(-1) > 0.5).float()
                    test_correct_preds += (preds == labels).sum().item()
                    test_total_preds += labels.size(0)

                    all_test_labels.extend(labels.cpu().numpy())
                    all_test_preds.extend(preds.cpu().numpy())

            test_loss /= len(test_loader.dataset)
            test_accuracy = test_correct_preds / test_total_preds
            precision = precision_score(all_test_labels, all_test_preds)
            recall = recall_score(all_test_labels, all_test_preds)
            f1 = f1_score(all_test_labels, all_test_preds)

            # Log metrics
            mlflow.log_metric("test_loss", test_loss)
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("test_precision", precision)
            mlflow.log_metric("test_recall", recall)
            mlflow.log_metric("test_f1_score", f1)

            logging.info(f"Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, "
                         f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

            # Save metrics to JSON
            final_metrics = {
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1_score": f1
            }

            metrics_file_path = "artifacts/metrics/metrics.json"
            os.makedirs(os.path.dirname(metrics_file_path), exist_ok=True)

            if os.path.exists(metrics_file_path):
                with open(metrics_file_path, "r") as f:
                    metrics_data = json.load(f)
            else:
                metrics_data = {}

            metrics_data.update(final_metrics)

            with open(metrics_file_path, "w") as f:
                json.dump(metrics_data, f, indent=4)
            logging.info("Metrics saved to metrics.json.")

            # Confusion matrix
            test_cm = confusion_matrix(all_test_labels, all_test_preds)
            cm_filename = "artifacts/metrics/confusion_matrix_test.png"
            plot_confusion_matrix(test_cm, classes=["Negative", "Positive"], filename=cm_filename)
            mlflow.log_artifact(cm_filename)
            logging.info("Confusion matrix saved and logged to MLflow.")

    except Exception as e:
        logging.exception("Error occurred during evaluation.")
        raise e

if __name__ == "__main__":
    evaluate_model()

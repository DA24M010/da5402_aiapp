import torch
import torch.nn as nn
from load_dataset import load_dataloaders
import mlflow
import mlflow.pytorch
import yaml
import json
import os
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# from scripts.pipeline.utils import plot_confusion_matrix
from utils import load_params, plot_confusion_matrix

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = load_params()
    eval_params = params["evaluate"]
    # Load DataLoader
    dataloaders = load_dataloaders("./artifacts/metadata/", test_batch_size=eval_params["test_batch_size"])
    test_loader = dataloaders["test"]

    # Read saved run_id
    with open("./artifacts/run_id.txt", "r") as f:
        run_id = f.read().strip()

    # Load model from MLflow
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pytorch.load_model(model_uri)
    model = model.to(device)
    model.eval()

    # Define loss function
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

                # Collect all labels and predictions for metrics calculation
                all_test_labels.extend(labels.cpu().numpy())
                all_test_preds.extend(preds.cpu().numpy())

        test_loss = test_loss / len(test_loader.dataset)
        test_accuracy = test_correct_preds / test_total_preds
        precision = precision_score(all_test_labels, all_test_preds)
        recall = recall_score(all_test_labels, all_test_preds)
        f1 = f1_score(all_test_labels, all_test_preds)

        # Log evaluation metrics to MLflow
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1_score", f1)

        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f} | "
              f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        # Save metrics to JSON file
        final_metrics = {
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1_score": f1
        }

        metrics_file_path = "artifacts/metrics/metrics.json"

        # Check if metrics.json exists
        if os.path.exists(metrics_file_path):
            # If it exists, load the current metrics and append new ones
            with open(metrics_file_path, "r") as f:
                metrics_data = json.load(f)
        else:
            # If it doesn't exist, create a new structure
            metrics_data = {}

        # Append evaluation metrics with the corresponding run_id
        metrics_data.update(final_metrics)

        # Save the updated metrics back to the file
        with open(metrics_file_path, "w") as f:
            json.dump(metrics_data, f, indent=4)

        # Save and log the confusion matrix
        test_cm = confusion_matrix(all_test_labels, all_test_preds)
        plot_confusion_matrix(test_cm, classes=["Negative", "Positive"], filename="artifacts/metrics/confusion_matrix_test.png")
        mlflow.log_artifact("artifacts/metrics/confusion_matrix_test.png")

        print("Evaluation completed and metrics saved to metrics.json!")

if __name__ == "__main__":
    evaluate_model()

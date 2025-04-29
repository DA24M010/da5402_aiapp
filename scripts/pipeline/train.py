import torch
import torch.optim as optim
import torch.nn as nn
import mlflow
import mlflow.pytorch
from load_dataset import load_dataloaders
from cnnmodel import CNNModel
from tqdm import tqdm 
import json  
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from utils import load_params, plot_confusion_matrix
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

def train_model():
    # Load parameters
    params = load_params()
    dataloader_params = params["dataloader"]
    train_params = params["train"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_sizes = {
        "train": dataloader_params["train_batch_size"],
        "val": dataloader_params["val_batch_size"]
    }
    num_workers = params["dataloader"]["num_workers"]
    image_size = params["dataloader"].get("image_size", 224)


    # Load DataLoader from saved file
    dataloaders = load_dataloaders("artifacts/metadata/", batch_sizes["train"], batch_sizes["val"], num_workers)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    # test_loader = dataloaders["test"]

    sample_batch = next(iter(train_loader))
    images, labels = sample_batch

    # Extract image size
    _, _, height, _ = images.shape
    image_size = height

    # Initialize model, loss, optimizer
    model = CNNModel(in_channel=3, num_classes=1, image_size=image_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=train_params["lr"])

    # Start MLflow logging
    with mlflow.start_run():
        with open("artifacts/run_id.txt", "w") as f:
            f.write(mlflow.active_run().info.run_id)

        mlflow.log_params(train_params)

        for epoch in range(train_params["epochs"]):
            model.train()
            running_loss = 0.0
            correct_preds = 0
            total_preds = 0
            all_train_labels = []
            all_train_preds = []

            # Progress bar for training
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_params['epochs']} [Training]")
            for inputs, labels in train_pbar:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels.float())
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                preds = (outputs.squeeze() > 0.5).float()
                correct_preds += (preds == labels).sum().item()
                total_preds += labels.size(0)

                # Collect all labels and predictions for metrics calculation
                all_train_labels.extend(labels.cpu().numpy())
                all_train_preds.extend(preds.cpu().numpy())

                # Update progress bar info
                train_pbar.set_postfix({
                    "loss": loss.item(),
                    "batch_acc": (preds == labels).float().mean().item()
                })

            train_loss = running_loss / len(train_loader.dataset)
            train_accuracy = correct_preds / total_preds
            precision = precision_score(all_train_labels, all_train_preds)
            recall = recall_score(all_train_labels, all_train_preds)
            f1 = f1_score(all_train_labels, all_train_preds)

            # Log metrics to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
            mlflow.log_metric("train_precision", precision, step=epoch)
            mlflow.log_metric("train_recall", recall, step=epoch)
            mlflow.log_metric("train_f1_score", f1, step=epoch)

            print(f"\n[Epoch {epoch+1}/{train_params['epochs']}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
                  f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct_preds = 0
            val_total_preds = 0
            val_all_labels = []
            val_all_preds = []

            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{train_params['epochs']} [Validation]")
            with torch.no_grad():
                for inputs, labels in val_pbar:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), labels.float())
                    val_loss += loss.item() * inputs.size(0)
                    preds = (outputs.squeeze() > 0.5).float()
                    val_correct_preds += (preds == labels).sum().item()
                    val_total_preds += labels.size(0)

                    # Collect all labels and predictions for metrics calculation
                    val_all_labels.extend(labels.cpu().numpy())
                    val_all_preds.extend(preds.cpu().numpy())

            val_loss = val_loss / len(val_loader.dataset)
            val_accuracy = val_correct_preds / val_total_preds
            val_precision = precision_score(val_all_labels, val_all_preds)
            val_recall = recall_score(val_all_labels, val_all_preds)
            val_f1 = f1_score(val_all_labels, val_all_preds)

            # Log validation metrics to MLflow
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
            mlflow.log_metric("val_precision", val_precision, step=epoch)
            mlflow.log_metric("val_recall", val_recall, step=epoch)
            mlflow.log_metric("val_f1_score", val_f1, step=epoch)

            print(f"\n[Epoch {epoch+1}/{train_params['epochs']}] "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f} | "
                  f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

                
        # Final metrics for JSON
        final_metrics = {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "train_precision": precision,
            "train_recall": recall,
            "train_f1_score": f1,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1_score": val_f1
        }

        sample_input = images.to(device)[:1]  # Single sample
        model.eval()
        with torch.no_grad():
            sample_output = model(sample_input)

        signature = infer_signature(sample_input.cpu().numpy(), sample_output.cpu().numpy())

        mlflow.pytorch.log_model(model, "model", signature=signature)
        result = mlflow.register_model(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
            name="surface_crack_detector"  
        )
        print(f"Registered model version: {result.version}")

        client = MlflowClient()
        client.transition_model_version_stage(
            name="surface_crack_detector",
            version=result.version,
            stage="Production"
        )
        print(f"Model version {result.version} promoted to Production.")

        torch.save(model, './artifacts/model.pth')

        # Save final metrics to JSON file
        with open("artifacts/metrics/metrics.json", "w") as f:
            json.dump(final_metrics, f, indent=4)

        # Save and log the confusion matrix for training phase at the end
        train_cm = confusion_matrix(all_train_labels, all_train_preds)
        plot_confusion_matrix(train_cm, classes=["Negative", "Positive"], filename="artifacts/metrics/confusion_matrix_train.png")
        mlflow.log_artifact("artifacts/metrics/confusion_matrix_train.png")

        # Save and log the confusion matrix for validation phase at the end
        val_cm = confusion_matrix(val_all_labels, val_all_preds)
        plot_confusion_matrix(val_cm, classes=["Negative", "Positive"], filename="artifacts/metrics/confusion_matrix_val.png")
        mlflow.log_artifact("artifacts/metrics/confusion_matrix_val.png")

        print("Training completed and final metrics saved to metrics.json!")

if __name__ == "__main__":
    train_model()

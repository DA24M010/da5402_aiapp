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
from utils import load_params, plot_confusion_matrix
import logging
import os
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

# Setup logging
os.makedirs("artifacts/logs", exist_ok=True)
logging.basicConfig(
    filename="artifacts/logs/train.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

def train_model():
    try:
        logging.info("Loading parameters and setting device...")
        params = load_params()
        dataloader_params = params["dataloader"]
        train_params = params["train"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        batch_sizes = {
            "train": dataloader_params["train_batch_size"],
            "val": dataloader_params["val_batch_size"]
        }

        # Load dataloaders
        dataloaders = load_dataloaders("artifacts/metadata/", batch_sizes["train"], batch_sizes["val"], dataloader_params["num_workers"])
        train_loader = dataloaders["train"]
        val_loader = dataloaders["val"]

        images, labels = next(iter(train_loader))
        image_size = images.shape[2]

        model = CNNModel(in_channel=3, num_classes=1, image_size=image_size).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=train_params["lr"])

        with mlflow.start_run():
            run_id = mlflow.active_run().info.run_id
            with open("artifacts/run_id.txt", "w") as f:
                f.write(run_id)
            logging.info(f"Started MLflow run: {run_id}")
            mlflow.log_params(train_params)

            for epoch in range(train_params["epochs"]):
                logging.info(f"Epoch {epoch+1}/{train_params['epochs']} - Training")
                model.train()
                running_loss, correct_preds, total_preds = 0.0, 0, 0
                all_train_labels, all_train_preds = [], []

                train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Training]")
                for inputs, labels in train_pbar:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs.view(-1), labels.float())
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    preds = (outputs.view(-1) > 0.5).float()
                    correct_preds += (preds == labels).sum().item()
                    total_preds += labels.size(0)

                    all_train_labels.extend(labels.cpu().numpy())
                    all_train_preds.extend(preds.cpu().numpy())

                    train_pbar.set_postfix({
                        "loss": loss.item(),
                        "batch_acc": (preds == labels).float().mean().item()
                    })

                train_loss = running_loss / len(train_loader.dataset)
                train_accuracy = correct_preds / total_preds
                precision = precision_score(all_train_labels, all_train_preds)
                recall = recall_score(all_train_labels, all_train_preds)
                f1 = f1_score(all_train_labels, all_train_preds)

                logging.info(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f}, "
                             f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "train_precision": precision,
                    "train_recall": recall,
                    "train_f1_score": f1
                }, step=epoch)

                # Validation phase
                logging.info(f"Epoch {epoch+1}/{train_params['epochs']} - Validation")
                model.eval()
                val_loss, val_correct_preds, val_total_preds = 0.0, 0, 0
                val_all_labels, val_all_preds = [], []

                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Validation]")
                with torch.no_grad():
                    for inputs, labels in val_pbar:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs.view(-1), labels.float())
                        val_loss += loss.item() * inputs.size(0)

                        preds = (outputs.view(-1) > 0.5).float()
                        val_correct_preds += (preds == labels).sum().item()
                        val_total_preds += labels.size(0)

                        val_all_labels.extend(labels.cpu().numpy())
                        val_all_preds.extend(preds.cpu().numpy())

                val_loss = val_loss / len(val_loader.dataset)
                val_accuracy = val_correct_preds / val_total_preds
                val_precision = precision_score(val_all_labels, val_all_preds)
                val_recall = recall_score(val_all_labels, val_all_preds)
                val_f1 = f1_score(val_all_labels, val_all_preds)

                logging.info(f"[Epoch {epoch+1}] Val Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f}, "
                             f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
                mlflow.log_metrics({
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_f1_score": val_f1
                }, step=epoch)

            # Save final metrics
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
            os.makedirs("artifacts/metrics", exist_ok=True)
            with open("artifacts/metrics/metrics.json", "w") as f:
                json.dump(final_metrics, f, indent=4)
            logging.info("Final metrics saved to artifacts/metrics/metrics.json")

            # Log model
            sample_input = images.to(device)[:1]
            model.eval()
            with torch.no_grad():
                sample_output = model(sample_input)
            signature = infer_signature(sample_input.cpu().numpy(), sample_output.cpu().numpy())

            mlflow.pytorch.log_model(model, "model", signature=signature)
            result = mlflow.register_model(
                model_uri=f"runs:/{run_id}/model",
                name="surface_crack_detector"
            )
            logging.info(f"Registered model version: {result.version}")

            client = MlflowClient()
            client.transition_model_version_stage(
                name="surface_crack_detector",
                version=result.version,
                stage="Production"
            )
            logging.info(f"Model version {result.version} promoted to Production")

            torch.save(model, './artifacts/model.pth')

            # Log confusion matrices
            train_cm = confusion_matrix(all_train_labels, all_train_preds)
            plot_confusion_matrix(train_cm, classes=["Negative", "Positive"], filename="artifacts/metrics/confusion_matrix_train.png")
            mlflow.log_artifact("artifacts/metrics/confusion_matrix_train.png")

            val_cm = confusion_matrix(val_all_labels, val_all_preds)
            plot_confusion_matrix(val_cm, classes=["Negative", "Positive"], filename="artifacts/metrics/confusion_matrix_val.png")
            mlflow.log_artifact("artifacts/metrics/confusion_matrix_val.png")

            logging.info("Training completed successfully!")

    except Exception as e:
        logging.error(f"Error occurred during training: {e}", exc_info=True)

if __name__ == "__main__":
    train_model()

import yaml
import matplotlib.pyplot as plt
import seaborn as sns

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

def plot_confusion_matrix(cm, classes, filename="confusion_matrix.png"):
    # Plot confusion matrix using seaborn
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(filename)
    plt.close()

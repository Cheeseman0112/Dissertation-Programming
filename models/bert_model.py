import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from data.splitter import loadSplit
from preprocessing.bert import preprocessBERT


#Confusion matrix plot
def plotConfusion(cm, title="Confusion Matrix - DistilBERT", fname = None):
    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        linewidths=0.5,
        linecolor="gray"
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()

    if fname:
        plt.savefig(fname, dpi=150)
    
    plt.show()


def computeMetrics(evalPred):
    logits, labels = evalPred
    preds = logits.argmax(axis = 1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )

    acc = accuracy_score(labels, preds)

    return{
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

#Run DistilBERT model
def runBERT(datasetName, plot_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    #Load dataset
    X_train, X_test, y_train, y_test = loadSplit(datasetName)
    
    #Tokenise using BERT tokeniser
    trainEnc, testEnc, tokenizer = preprocessBERT(X_train, X_test)

    #Convert to dataset format
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels.reset_index(drop=True)
        
        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels.iloc[idx])
            return item
        
        def __len__(self):
            return len(self.labels)
        
    trainDataset = Dataset(trainEnc, y_train)
    testDataset = Dataset(testEnc, y_test)

    #Load pretrained DistilBERT
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels = 2
    ).to(device)

    print("Model running on:", model.device)

    #Training configuration
    trainingArgs = TrainingArguments(
        output_dir = "./results",
        num_train_epochs = 2,
        per_device_train_batch_size = 4,
        per_device_eval_batch_size = 4,
        eval_strategy = "epoch",
        logging_dir = "./logs",
        fp16 = True,
        save_strategy="no"
    )

    #Trainer handles traiing loop
    trainer = Trainer(
        model = model,
        args = trainingArgs,
        train_dataset = trainDataset,
        eval_dataset = testDataset,
        compute_metrics = computeMetrics
    )

    #Training time
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()

    trainer.train()

    torch.cuda.synchronize() if device.type == "cuda" else None
    train_time = time.perf_counter() - t0

    #Evaluation timing
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()

    results = trainer.evaluate()

    torch.cuda.synchronize() if device.type == "cuda" else None
    test_time = time.perf_counter() - t0

    #Predicions
    preds_output = trainer.predict(testDataset)
    preds = np.argmax(preds_output.predictions, axis=1)

    #Confusion matrix
    cm = confusion_matrix(y_test, preds)
    class_names = ["Real", "Fake"]

    cm_df = pd.DataFrame(
        cm,
        index=[f"True_{c}" for c in class_names],
        columns=[f"Pred_{c}" for c in class_names]
    )

    plotConfusion(cm_df, title=f"Confusion - {datasetName}", fname=plot_path)

    #Classification report
    report = classification_report(y_test, preds, zero_division=0)

    print(f"\n[DistilBERT] Results for {datasetName}")
    print(f"  Accuracy       : {results['eval_accuracy']:.4f}")
    print(f"  Precision      : {results['eval_precision']:.4f}")
    print(f"  Recall         : {results['eval_recall']:.4f}")
    print(f"  F1 Score       : {results['eval_f1']:.4f}")
    print(f"  Train time     : {train_time:.2f}s")
    print(f"  Test time      : {test_time:.2f}s")

    print("\nClassification report:")
    print(report)

    return results
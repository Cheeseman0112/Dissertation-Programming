import torch
import torch.nn as nn
import torch.optim as optim

import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sys import getsizeof
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from data.splitter import loadSplit
from preprocessing.lstm import preprocessLSTM

#Utility functions
def _mem_mb(obj) -> float:
    return getsizeof(obj) / (1024 ** 2)

#Confusion matrix
def plotConfusion(cm: pd.DataFrame, title: str = "Confusion matrix for LSTM", fname: str | None = None) -> None:
    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        linewidths=0.5,
        linecolor="gray",
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title, pad=12)
    plt.tight_layout()

    if fname:
        plt.savefig(fname, dpi=150, bbox_inches="tight")

    plt.show()

def _estimate_lstm_flops(seq_len: int, embed_dim: int, hidden_dim: int) -> int:
    per_step = 4 * (embed_dim * hidden_dim + hidden_dim * hidden_dim + hidden_dim)
    return per_step * seq_len

#Define LSTM model architecture
class LSTMModel(nn.Module):
    def __init__(self, vocabSize, embedDim = 128, hiddenDim = 256):
        super().__init__()

        #Embedding layer converts word indices to dense vectors
        self.embedding = nn.Embedding(vocabSize + 1, embedDim)

        #LSTM layer process sequences
        self.lstm = nn.LSTM(embedDim, hiddenDim, batch_first=True)

        #Fully connected layer for classification
        self.fc = nn.Linear(hiddenDim, 1)

    def forward(self, x):
        x = self.embedding(x.long())
        _, (hidden, _) = self.lstm(x)
        x = self.fc(hidden[-1])
        return x

#Runs LSTM model on dataset
def runLSTM(datasetName, epochs = 3, plot_path: str | None = None):
    #Make it run on the gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    #Load dataset
    X_train, X_test, y_train, y_test = loadSplit(datasetName)

    #Preprocess into padded sequences
    X_train_pad, X_test_pad, wordIndex = preprocessLSTM(X_train, X_test)

    #Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_pad)
    y_train_tensor = torch.tensor(y_train.values).float().unsqueeze(1)

    X_test_tensor = torch.tensor(X_test_pad)
    y_test_tensor = torch.tensor(y_test.values).float().unsqueeze(1)

    #Shuffle training data
    perm = torch.randperm(len(X_train_tensor))
    X_train_tensor = X_train_tensor[perm]
    y_train_tensor = y_train_tensor[perm]

    print(y_train.value_counts())

    #Create model
    model = LSTMModel(vocabSize = len(wordIndex)).to(device)

    pos_weight = torch.tensor([
        (len(y_train) - y_train.sum()) / y_train.sum()
    ]).to(device)
    print("pos_weight:", pos_weight.item())

    #classification
    criterion = nn.BCEWithLogitsLoss()

    #Opimiser
    optimiser = optim.Adam(model.parameters(), lr=0.0005)

    batchSize = 8

    #Training loop
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()

    for epoch in range(epochs):
        model.train()

        for i in range(0, len(X_train_tensor), batchSize):
            X_batch = X_train_tensor[i:i+batchSize].to(device)
            y_batch = y_train_tensor[i:i+batchSize].to(device)

            optimiser.zero_grad()

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimiser.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    torch.cuda.synchronize() if device.type == "cuda" else None
    train_time = time.perf_counter() - t0

    #Evalutation
    model.eval()

    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()

    with torch.no_grad():
        logits = model(X_test_tensor.to(device))
    torch.cuda.synchronize() if device.type == "cuda" else None
    test_time = time.perf_counter() - t0

    #Apply sigmoid after model
    predictions = torch.sigmoid(logits)
    predictions = (predictions > 0.5).int().cpu().numpy()
    print("Unique predictions:", set(predictions.flatten()))
    probs = torch.sigmoid(logits).cpu().numpy()
    print("Min prob:", probs.min())
    print("Max prob:", probs.max())
    print("Mean prob:", probs.mean())

    #Metrics
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)

    #Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    class_names = ["Real", "Fake"]

    cm_df = pd.DataFrame(
        cm, 
        index=[f"True_{c}" for c in class_names],
        columns= [f"Pred_{c}" for c in class_names]
    )
    
    plotConfusion(cm_df, title=f"Confusion - {datasetName}", fname=plot_path)

    #Computational metrics
    seq_len = X_test_pad.shape[1]
    embed_dim = model.embedding.embedding_dim
    hidden_dim = model.lstm.hidden_size

    flops_per_pred = _estimate_lstm_flops(seq_len, embed_dim, hidden_dim)

    #Memory usage
    mem_model_mb = (
        _mem_mb(model.embedding.weight) +
        _mem_mb(model.lstm.weight_ih_l0) +
        _mem_mb(model.lstm.weight_hh_l0) +
        _mem_mb(model.lstm.bias_ih_l0) +
        _mem_mb(model.lstm.bias_hh_l0) +
        _mem_mb(model.fc.weight) +
        _mem_mb(model.fc.bias)
    )
    mem_data_mb = (
        _mem_mb(X_train_tensor) +
        _mem_mb(X_test_tensor) +
        _mem_mb(y_train_tensor) +
        _mem_mb(y_test_tensor)
    )

    total_mem_mb = mem_data_mb +mem_data_mb

    #Outputs
    print("\n[LSTM] Results")
    print(f"  Accuracy               : {accuracy:.4f}")
    print(f"  Train time (GPU/CPU)   : {train_time:.2f}s")
    print(f"  Test  time (GPU/CPU)   : {test_time:.4f}s")
    print(f"  FLOPs per prediction   : {flops_per_pred:,}")
    print(f"  Model memory (MiB)     : {mem_model_mb:.2f}")
    print(f"  Data memory (MiB)      : {mem_data_mb:.2f}")
    print(f"  Total memory (MiB)     : {total_mem_mb:.2f}")

    print("\nClassification report:")
    print(report)

    return accuracy, report
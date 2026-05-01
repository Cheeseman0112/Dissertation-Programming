import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support

from sys import getsizeof

from data.splitter import loadSplit
from preprocessing.svm import preprocessSVM

#Estimate the memory taken by a sparse CSR matrix
def _mem_mb(obj):
    return getsizeof(obj) / (1024 ** 2)

#Confusion matrix
def plot_confusion(cm, class_names, title = "Confusion matrix for SVM", fname = None):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        linewidths=0.5,
        linecolor="gray",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Precited")
    plt.ylabel("Actual")
    plt.title(title, pad=15)
    plt.tight_layout()

    if fname:
        plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()

#Runs SVM on a given dataset
def runSVM(datasetName, plot_path = None):
    X_train, X_test, y_train, y_test = loadSplit(datasetName)

    X_train_vec, X_test_vec, vectorizer = preprocessSVM(X_train, X_test)

    #Create SVM model, linear kernal used for text classification
    model = SVC(
        kernel = "linear",
        C = 1.0,
        class_weight = "balanced",
        probability = False
    )

    #Train the model
    t0 = time.perf_counter()
    model.fit(X_train_vec, y_train)
    trainTime = time.perf_counter() - t0

    #Make predictions
    t0 = time.perf_counter()
    predictions = model.predict(X_test_vec)
    testTime = time.perf_counter() - t0

    #Evaluate performace
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average=None, zero_division=0)

    #Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    class_names = [str(c) for c in model.classes_]

    plot_confusion(cm, class_names, title=f"SVM - {datasetName}", fname=plot_path)

    #Computational cost estimates
    n_support = model.n_support_.sum()
    n_features = X_train_vec.shape[1]
    flops_per_pred = n_support * n_features

    mem_model_mb = (
        _mem_mb(model.coef_)
        + _mem_mb(model.intercept_)
        + _mem_mb(model.support_vectors_)
    )
    mem_vectors_mb = _mem_mb(X_train_vec) + _mem_mb(X_test_vec)
    mem_total_mb = mem_model_mb + mem_vectors_mb

    print(f"\n[SVM] Results for {datasetName}")
    print(f"  Accuracy               : {accuracy:.4f}")
    print(f"  Train time (CPU)       : {trainTime:.2f}s")
    print(f"  Test time (CPU)        : {testTime:.4f}s")
    print(f"  # support vectors      : {n_support}")
    print(f"  FLOPs per prediction   : {flops_per_pred:,}")
    print(f"  Model memory (MiB)     : {mem_model_mb:.2f}")
    print(f"  TF‑IDF vectors memory  : {mem_vectors_mb:.2f}")
    print(f"  Total memory mb        : {mem_total_mb:.2f}")
    print("\nClassification report:")
    print(report)

    return accuracy, report
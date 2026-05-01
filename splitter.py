import os
import pandas as pd
from sklearn.model_selection import train_test_split

#Creates a fixed split and saves it
def createAndSaveSplit(df, datasetName, testSize = 0.2, randomState = 42):
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size = testSize,
        random_state = randomState,
        stratify = df["label"]
    )

    splitDir = f"data/splits/{datasetName}"
    os.makedirs(splitDir, exist_ok = True)

    #Save splits
    pd.DataFrame({"text": X_train, "label": y_train}).to_csv(
        f"{splitDir}/train.csv", index = False
    )
    pd.DataFrame({"text": X_test, "label": y_test}).to_csv(
        f"{splitDir}/test.csv", index = False
    )

    print(f"Splits saved for {datasetName}")

#Loads pre-saved splits
def loadSplit(datasetName):
    trainPath = f"data/splits/{datasetName}/train.csv"
    testPath = f"data/splits/{datasetName}/test.csv"

    train_df = pd.read_csv(trainPath)
    test_df = pd.read_csv(testPath)

    return(
        train_df["text"],
        test_df["text"],
        train_df["label"],
        test_df["label"]
    )
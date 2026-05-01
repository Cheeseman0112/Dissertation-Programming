import pandas as pd
from .utils import basicClean, standardiseLabels, validateDataset

#Function to load and prepare a dataset in a standard format
def loadDataset(config):
    print("DEBUG: loadDataset is running")

    df = pd.read_csv(config["path"])

    #Make all columns lowercase
    df.columns = df.columns.str.lower().str.strip()
    print("DEBUG: columns after lower():", df.columns)

    labelCol = config["labelCol"].lower()

    #Combine multiple text columns
    df["text"] = ""
    for col in config["textCols"]:
        if col in df.columns:
            df["text"] += " " + df[col].fillna("")
    
    #Keep only required columns
    df = df[["text", labelCol]].dropna()
    print("DEBUG: columns after trimming:", df.columns)

    df = df[df["text"].str.strip() != ""]

    #Validate + clean
    df = validateDataset(df, "text", config["labelCol"])
    df["text"] = df["text"].apply(basicClean)

    #Standardise labels
    df = standardiseLabels(df, config["labelCol"])

    #Rename label column to standard name
    df = df.rename(columns={config["labelCol"]: "label"})

    print(f"{config['name']} loaded: {len(df)} samples")

    return df
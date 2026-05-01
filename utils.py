import re
import pandas as pd

def basicClean(text):
    #Basic cleaning of datasets
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def standardiseLabels(df, labelCol):
    df[labelCol] = df[labelCol].replace({
        "FAKE": 0, "REAL": 1,
        "Fake": 0, "Real": 1,
        "fake": 0, "real": 1
    })

    return df

def validateDataset(df, textCol, labelCol):
    #Basic validation to prevent runtime errors
    if textCol not in df.columns:
        raise ValueError(f"Missing text column: {textCol}")
    if labelCol not in df.columns:
        raise ValueError(f"Missing label column: {labelCol}")
    
    df = df[[textCol, labelCol]].dropna()

    return df
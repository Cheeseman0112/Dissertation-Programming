import pandas as pd

# -----------------------------
# CHANGE THESE PATHS
# -----------------------------
datasets = [
    {
    "name": "WELFake_Dataset",
    "path": "C:/Users/epic-/OneDrive/Documents/University/Dissertation/Diss Programming/data/datasets/WELFake_Dataset.csv",
    "textCols": ["title", "text"],
    "labelCol": "label"
    },
    
    {
    "name": "news_articles",
    "path": "C:/Users/epic-/OneDrive/Documents/University/Dissertation/Diss Programming/data/datasets/news_articles.csv",
    "textCols": ["title", "test"],
    "labelCol": "label"
    },
    {
    "name": "FakeNewsNet",
    "path":"C:/Users/epic-/OneDrive/Documents/University/Dissertation/Diss Programming/data/datasets/FakeNewsNet.csv",
    "textCols": ["title"],
    "labelCol": "real"
    }
]

# -----------------------------
# PROCESS EACH DATASET
# -----------------------------
for dataset in datasets:
    print(f"\nDataset: {dataset['name']}")
    
    df = pd.read_csv(dataset["path"])

    # Make sure column names are consistent
    df.columns = df.columns.str.lower()
    labelCol = dataset["labelCol"].lower()

    # Drop missing labels just in case
    df = df.dropna(subset=[labelCol])

    # Get counts
    counts = df[labelCol].value_counts()

    print("Label Distribution:")
    print(counts)

    # Optional: percentages (nice for dissertation)
    percentages = df[labelCol].value_counts(normalize=True) * 100
    print("\nPercentage Distribution:")
    print(percentages.round(2))
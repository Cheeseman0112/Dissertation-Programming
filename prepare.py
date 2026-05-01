from data.loader import loadDataset
from data.splitter import createAndSaveSplit

#Define dataset config
datasetConfig = [
    {
    "name": "WELFake_Dataset",
    "path": "C:/Users/epic-/OneDrive/Documents/University/Dissertation/Diss Programming/data/datasets/WELFake_Dataset.csv",
    "textCols": ["title", "text"],
    "labelCol": "label"
    },
    
    {
    "name": "news_articles",
    "path": "C:/Users/epic-/OneDrive/Documents/University/Dissertation/Diss Programming/data/datasets/news_articles.csv",
    "textCols": ["title", "text"],
    "labelCol": "label"
    },
    {
    "name": "FakeNewsNet",
    "path":"C:/Users/epic-/OneDrive/Documents/University/Dissertation/Diss Programming/data/datasets/FakeNewsNet.csv",
    "textCols": ["title"],
    "labelCol": "real"
    }
]

#Loop through datasets
for config in datasetConfig:
    #Load + Clean dataset
    df = loadDataset(config)
    #Create and save aplit
    createAndSaveSplit(df, config["name"])
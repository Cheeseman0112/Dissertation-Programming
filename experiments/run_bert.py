import argparse
from models.bert_model import runBERT

def main():
    parser = argparse.ArgumentParser(description = "Run DistilBERT model")

    parser.add_argument(
        "--dataset",
        type = str,
        required = True,
        help = "Dataset name"
    )

    args = parser.parse_args()

    runBERT(args.dataset)

if __name__ == "__main__":
    main()
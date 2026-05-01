import argparse
from models.lstm_model import runLSTM

def main():
    parser = argparse.ArgumentParser(description = "Run LSTM model")

    parser.add_argument(
        "--dataset",
        type = str,
        required = True,
        help = "Dataset name"
    )

    parser.add_argument(
        "--epochs",
        type = int,
        default = 3,
        help = "Number of training epochs"
    )

    args = parser.parse_args()

    runLSTM(args.dataset, epochs = args.epochs)

if __name__ == "__main__":
    main()
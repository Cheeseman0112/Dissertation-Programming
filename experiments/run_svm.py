import argparse
from models.svm_model import runSVM

def main():
    parser = argparse.ArgumentParser(description="Run SVM model")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (must match data/splits folder)"
    )

    args = parser.parse_args()
    
    runSVM(args.dataset)

if __name__ == "__main__":
    main()
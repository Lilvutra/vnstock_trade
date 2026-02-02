from ..src.vnstock_trade.fetch.fetch_vnstock import fetch_data
from ..src.vnstock_trade.features.build_features import build_features
from ..src.vnstock_trade.labels.generate import generate_labels
from ..src.vnstock_trade.training.train_model import train

def main():
    fetch_data()
    build_features()
    generate_labels()
    train()

if __name__ == "__main__":
    main()

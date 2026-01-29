from vnstock.fetch.fetch_vnstock import fetch_data
from vnstock.features.build_features import build_features
from vnstock.labels.generate_label import generate_label
from vnstock.training.train_model import train

def main():
    fetch_data()
    build_features()
    generate_label()
    train()

if __name__ == "__main__":
    main()

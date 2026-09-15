from app.pipeline.data_loader import DataLoader


def main():
    loader = DataLoader()

    df = loader.load()

    loader.dataset_info(df)

    print(df.head())


if __name__ == "__main__":
    main()
    
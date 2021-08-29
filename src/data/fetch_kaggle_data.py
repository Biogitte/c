import kaggle
import argparse


def fetch_kaggle_data(dataset: str, output_path: str) -> str:
    """
    Download Kaggle datasets
    :param dataset: Name of Kaggle dataset to download
    :param output_path: Location where dataset will be saved
    :return: The downloaded file and a string "Download Complete"

    """
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset, path=output_path, unzip=True)
    return 'Download Complete'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str,
                        help="Name of the Kaggle dataset to download.")
    parser.add_argument("output_path", type=str,
                        help="Path to where to download the dataset.")
    args = parser.parse_args()
    fetch_kaggle_data(args.dataset, args.output_path)


if __name__ == "__main__":
    main()

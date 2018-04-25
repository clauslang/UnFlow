import os
import gzip
import pickle


def main(pkl_file_path, destination_dir):
    with gzip.open(pkl_file_path) as src_file:
        memories = pickle.load(src_file, encoding="bytes")
        i = 9


if __name__ == "__main__":
    main(pkl_file_path="../blue200_original.pkl", destination_dir="data/nao_raw/blue200")

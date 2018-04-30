import gzip
import pickle
from PIL import Image


def main(pkl_file_path, destination_dir):
    # todo: print status
    with gzip.open(pkl_file_path) as src_file:
        memories = pickle.load(src_file, encoding="bytes")
        for i, memory in enumerate(memories):
            name = destination_dir + "nao_img_" + str(i).zfill(6) + ".png"
            image = memory[b'image'][:, :, ::-1]    # convert from BGR to RGB
            Image.fromarray(image).save(name)


if __name__ == "__main__":
    main(pkl_file_path="../grey400_original.pkl", destination_dir="../data/nao_raw/grey400/")

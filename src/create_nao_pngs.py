import gzip
import pickle
from PIL import Image


def main(pkl_file_path, destination_dir):
    # todo: print status
    print('opening file...')
    with gzip.open(pkl_file_path) as src_file:
        print('loading images...')
        memories = pickle.load(src_file, encoding="bytes")
        print('extracting {} images..'.format(len(memories)))
        for i, memory in enumerate(memories):
            if i % 100 == 0:
                print('extracted {} / {} images'.format(i, len(memories)))
            name = destination_dir + "nao_img_" + str(i).zfill(6) + ".png"
            image = memory[b'image'][:, :, ::-1]    # convert from BGR to RGB
            Image.fromarray(image).save(name)


if __name__ == "__main__":
    main(pkl_file_path="../grey400_original.pkl", destination_dir="../data/nao_raw/grey400/")

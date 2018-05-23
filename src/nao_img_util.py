import os
import gzip
import pickle
import numpy as np
from PIL import Image


def pickle_pngs(directory, destination_path):
    memories = []
    file_names = [file_name for file_name in os.listdir(directory) if file_name.endswith('.png')]
    file_names.sort()
    for i, file_name in enumerate(file_names):
        if i % 100 == 0:
            print('converted {} / {} images'.format(i, len(file_names)))
        print(file_name)
        img = Image.open(directory + file_name)
        img.load()
        array = np.asarray(img)
        memories.append({'image': array, 'sensor_angles': [0, 0, 0, 0]})
    with gzip.open(destination_path, 'wb') as destination_file:
        pickle.dump(memories, destination_file, 2)


def create_pngs(pkl_file_path, destination_dir):
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
    # create_pngs(pkl_file_path="../grey400_original.pkl", destination_dir="../data/nao_raw/grey400/")
    pickle_pngs('../out/css_nao/', '../grey400_flow_thr5.pkl')

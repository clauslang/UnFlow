import os
import numpy as np
import tensorflow as tf
from e2eflow.core.input import Input, frame_name_to_num, read_png_image, random_crop


class NaoInput(Input):

    def __init__(self, data, batch_size, dims, dir_name, *, num_threads=1, normalize=True, skipped_frames=False):
        super().__init__(data, batch_size, dims,
                         num_threads=num_threads, normalize=normalize, skipped_frames=skipped_frames)
        self.dir_name = dir_name

    def input_consecutive(self, sequence=True, shift=0, skip=0, return_shape=True):
        """Assumes that paired images are next to each other after ordering the
        files.
        """

        if not isinstance(skip, list):
            skip = [skip]

        data_dirs = self.data.get_raw_dirs(dir_name=self.dir_name)

        filenames = []
        for dir_path in data_dirs:
            files = os.listdir(dir_path)
            files.sort()
            if sequence:
                steps = [1 + s for s in skip]
                stops = [len(files) - s for s in steps]
            else:
                steps = [2]
                stops = [len(files)]
                assert len(files) % 2 == 0
            for step, stop in zip(steps, stops):
                for i in range(0, stop, step):
                    if self.skipped_frames and sequence:
                        assert step == 1
                        num_first = frame_name_to_num(files[i])
                        num_second = frame_name_to_num(files[i + 1])
                        if num_first + 1 != num_second:
                            continue
                    fn1 = os.path.join(dir_path, files[i])
                    fn2 = os.path.join(dir_path, files[i + 1])
                    filenames.append((fn1, fn2))

        print("Loading {} frame pairs.".format(len(filenames)))

        shift = shift % len(filenames)
        filenames = list(np.roll(filenames, shift))

        filenames_1, filenames_2 = zip(*filenames)
        filenames_1 = list(filenames_1)
        filenames_2 = list(filenames_2)

        input_1 = read_png_image(filenames_1)
        input_2 = read_png_image(filenames_2)
        image_1 = self._preprocess_image(input_1)
        image_2 = self._preprocess_image(input_2)

        if return_shape:
            return tf.train.batch(
                [image_1, image_2, tf.shape(image_1)],
                batch_size=self.batch_size,
                num_threads=self.num_threads,
                allow_smaller_final_batch=True)
        else:
            return tf.train.batch(
                [image_1, image_2],
                batch_size=self.batch_size,
                num_threads=self.num_threads,
                allow_smaller_final_batch=True)

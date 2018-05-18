import os
import numpy as np
import tensorflow as tf
from e2eflow.core.input import Input, frame_name_to_num, read_png_image, random_crop


class NaoInput(Input):

    def input_train_2012(self, hold_out_inv=None):
        return self._input_test('nao_raw/grey400', hold_out_inv)

    def input_debug(self, hold_out_inv=None):
        return self._input_test('nao_raw/greyfew', hold_out_inv)

    def input_consecutive(self, sequence=True, needs_crop=True, shift=0, skip=0):
        """Constructs input of raw data.

        Args:
            sequence: Assumes that image file order in data_dirs corresponds to
                temporal order, if True. Otherwise, assumes uncorrelated pairs of
                images in lexicographical ordering.
            shift: number of examples to shift the input queue by.
                Useful to resume training.
        Returns:
            image_1: batch of first images
            image_2: batch of second images
            input_shape: shape of input images
        """
        if not isinstance(skip, list):
            skip = [skip]

        data_dirs = self.data.get_raw_dirs()
        height, width = self.dims

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

        print("Training on {} frame pairs.".format(len(filenames)))

        filenames_extended = []
        for fn1, fn2 in filenames:
            filenames_extended.append((fn1, fn2))

        shift = shift % len(filenames_extended)
        filenames_extended = list(np.roll(filenames_extended, shift))

        filenames_1, filenames_2 = zip(*filenames_extended)
        filenames_1 = list(filenames_1)
        filenames_2 = list(filenames_2)

        with tf.variable_scope('train_inputs'):
            image_1 = read_png_image(filenames_1)
            image_2 = read_png_image(filenames_2)

            if needs_crop:
                # if center_crop:
                #    image_1 = tf.image.resize_image_with_crop_or_pad(image_1, height, width)
                #    image_2 = tf.image.resize_image_with_crop_or_pad(image_1, height, width)
                # else:
                image_1, image_2 = random_crop([image_1, image_2], [height, width, 3])
            else:
                image_1 = tf.reshape(image_1, [height, width, 3])
                image_2 = tf.reshape(image_2, [height, width, 3])

            if self.normalize:
                image_1 = self._normalize_image(image_1)
                image_2 = self._normalize_image(image_2)

            return tf.train.batch(
                [image_1, image_2, tf.shape(image_1)],
                batch_size=self.batch_size,
                num_threads=self.num_threads)

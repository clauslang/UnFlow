from e2eflow.core.input import Input

import tensorflow as tf


class NaoInput(Input):

    def input_test(self):
        input_shape, im1, im2 = self._input_images('nao_raw/blue200')
        # flow, mask = self._input_flow()
        return tf.train.batch(
            # [im1, im2, input_shape, flow, mask],
            [im1, im2, input_shape],
            batch_size=self.batch_size,
            num_threads=self.num_threads,
            allow_smaller_final_batch=True)


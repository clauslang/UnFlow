import tensorflow as tf
from e2eflow.core.input import Input


class NaoInput(Input):

    def input_train_2012(self, hold_out_inv=None):
        return self._input_test('nao_raw/grey400', hold_out_inv)

    def input_debug(self, hold_out_inv=None):
        return self._input_test('nao_raw/greyfew', hold_out_inv)

    def input_consecutive(self, hold_out_inv=None):
        im1, im2 = self.input_raw(swap_images=False)
        return im1, im2, tf.shape(im1)

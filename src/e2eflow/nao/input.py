from e2eflow.core.input import Input

import tensorflow as tf


class NaoInput(Input):

    def input_test(self, hold_out_inv=None):
        return self._input_test('data/nao_raw/blue200', hold_out_inv)


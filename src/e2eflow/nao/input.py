from e2eflow.core.input import Input


class NaoInput(Input):

    def input_test(self, hold_out_inv=None):
        return self._input_test('nao_raw/grey400', hold_out_inv)

    def input_train_2012(self, hold_out_inv=None):
        return self._input_test('nao_raw/blue200', hold_out_inv)

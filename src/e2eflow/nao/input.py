from e2eflow.core.input import Input


class NaoInput(Input):

    def input_test(self, hold_out_inv=None):
        return self._input_test('nao_raw/grey400', hold_out_inv)

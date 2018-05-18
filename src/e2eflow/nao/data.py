from ..core.data import Data


class NaoData(Data):

    def __init__(self, data_dir, stat_log_dir=None,
                 development=True, fast_dir=None):
        super().__init__(data_dir, stat_log_dir,
                         development=development,
                         fast_dir=fast_dir)

    def _fetch_if_missing(self):
        pass

    def get_raw_dirs(self, dir_name='grey400'):
        # todo: custom data dir from config.ini
        return ['../data/nao_raw/' + dir_name]

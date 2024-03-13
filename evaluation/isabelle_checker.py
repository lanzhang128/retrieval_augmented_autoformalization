import os
from tqdm import tqdm
from isabelle import Isabelle, write_to_thy_file, write_error_to_file


class IsabelleChecker:
    def __init__(self,
                 session_name='IsarMathLib',
                 server_log_file='server.log',
                 isabelle_dirs=None,
                 dependency_file=None):
        self.checker = Isabelle(session_name=session_name, log_file=server_log_file, dirs=isabelle_dirs)
        self.dependency_file = dependency_file if dependency_file is not None else '../base.thy'

    def evaluate(self, files_dir, keys, texts, statements):
        if not os.path.exists(files_dir):
            os.mkdir(files_dir)
        if len(statements) == 0:
            raise ValueError('Statements are empty!')

        count = 0
        pass_count = 0
        for key, text, statement in zip(tqdm(keys), texts, statements):
            thy_file_path = os.path.join(files_dir, f'test_{key}.thy')
            write_to_thy_file(thy_file_path, f'test_{key}', self.dependency_file, text, statement)
            response, inference_time = self.checker.get_response(theories=[f'test_{key}'], master_dir=files_dir)
            is_valid, error_lines, error_details, _ = self.checker.check_error(isabelle_response=response)
            error_log_path = os.path.join(files_dir, f'test_{key}.error.log')
            write_error_to_file(error_log_path, is_valid, error_lines, error_details, inference_time)
            count += 1
            if is_valid:
                pass_count += 1
        return {'Pass': pass_count / count}

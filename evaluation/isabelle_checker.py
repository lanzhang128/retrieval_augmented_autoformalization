import os
from tqdm import tqdm
from isabelle import Isabelle, write_to_thy_file, write_error_to_file
import signal


def handler(signum, frame):
    raise Exception('Isabelle response timed out at: ')


class IsabelleChecker:
    def __init__(self,
                 session_name='IsarMathLib',
                 server_log_file='server.log',
                 isabelle_dirs=None,
                 watchdog_timeout=60,
                 timeout=120):
        self.checker = Isabelle(session_name=session_name,
                                log_file=server_log_file,
                                dirs=isabelle_dirs,
                                watchdog_timeout=watchdog_timeout)
        self.timeout = timeout

    def evaluate(self, files_dir, keys, imports, texts, statements):
        if not os.path.exists(files_dir):
            os.mkdir(files_dir)
        if len(statements) == 0:
            raise ValueError('Statements are empty!')

        count = 0
        pass_count = 0
        for key, import_thy, text, statement in zip(tqdm(keys), imports, texts, statements):
            thy_file_path = os.path.join(files_dir, f'test_{key}.thy')
            error_log_path = os.path.join(files_dir, f'test_{key}.error.log')

            if os.path.exists(error_log_path):
                with open(error_log_path, 'r') as f:
                    is_valid = f.readlines()[0].split()[-1]
                is_valid = True if is_valid == 'True' else False

            else:
                write_to_thy_file(thy_file_path, f'test_{key}', import_thy, text, statement)
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(self.timeout)

                try:
                    response, inference_time = self.checker.get_response(theories=[f'test_{key}'], master_dir=files_dir)
                    signal.alarm(0)
                except Exception as e:
                    print(e, end=f'test_{key}.thy\n')
                    response, inference_time = [], self.timeout
                    self.checker.shutdown()
                    self.checker.restart()

                is_valid, error_lines, error_details, _ = self.checker.check_error(isabelle_response=response)

                write_error_to_file(error_log_path, is_valid, error_lines, error_details, inference_time)

            count += 1
            if is_valid:
                pass_count += 1
        return {'Pass Count': pass_count, 'Pass Rate': pass_count / count}

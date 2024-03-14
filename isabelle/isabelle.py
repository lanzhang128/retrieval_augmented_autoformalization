import json
import time
from isabelle_client import start_isabelle_server, get_isabelle_client


class Isabelle:
    def __init__(self,
                 isabelle_name='test',
                 port=8888,
                 log_file='server.log',
                 session_name='HOL',
                 dirs=None,
                 verbose=True,
                 options=None,
                 timeout=60):
        self.isabelle_name = isabelle_name
        self.port = port
        self.log_file = log_file
        self.session_name = session_name
        self.dirs = dirs if dirs is not None else ['./Isabelle2023']
        self.verbose = verbose
        self.options = options if options is not None else []
        self.timeout = timeout
        self._init_client()
        self._init_session()

    def _init_client(self):
        start_time = time.time()
        server_info, _ = start_isabelle_server(name=self.isabelle_name, port=self.port, log_file=self.log_file)
        self.isabelle = get_isabelle_client(server_info)
        print(f'Isabelle server started with info: {server_info} in {time.time() - start_time:.2f}s.')

    def _init_session(self):
        start_time = time.time()
        self.isabelle.session_build(session=self.session_name, dirs=self.dirs, verbose=self.verbose,
                                    options=self.options)
        self.session_id = self.isabelle.session_start(session=self.session_name)
        print(f'Isabelle session started in {time.time() - start_time:.2f}s.')

    def get_response(self, theories, master_dir):
        start_time = time.time()
        isabelle_response = self.isabelle.use_theories(session_id=self.session_id,
                                                       theories=theories,
                                                       master_dir=master_dir,
                                                       watchdog_timeout=self.timeout)
        inference_time = time.time() - start_time
        return isabelle_response, inference_time

    @staticmethod
    def check_error(isabelle_response, proof_code_file_path=None):
        is_valid = True
        error_details = []
        error_lines = []
        stuck_error_line = None

        finished_response = next((item for item in isabelle_response if item.response_type == 'FINISHED'), None)

        if finished_response:
            response_body = json.loads(finished_response.response_body)
            if response_body.get('nodes') and 'percentage' in response_body['nodes'][0]['status']:
                percentage = int(response_body['nodes'][0]['status']['percentage'])
            else:
                percentage = 0
            print(f'The finishing percentage is: {percentage}')

            if response_body.get('errors'):
                for error in response_body['errors']:
                    message, line = error['message'], error['pos']['line']
                    error_details.append(f'Error on line {line}: {message}')
                    error_lines.append(line)
                    is_valid = False
            elif percentage != 100:
                if proof_code_file_path is not None:
                    proof_line_number = None
                    qed_line_number = None
                    with open(proof_code_file_path, 'r') as file:
                        for i, line in enumerate(file, start=1):
                            if 'proof -' in line:
                                proof_line_number = i
                            if 'qed' in line:
                                qed_line_number = i

                    if not error_details:
                        if proof_line_number is not None:
                            if qed_line_number is not None:
                                number = round((qed_line_number - proof_line_number - 1) * (percentage / 100))
                            else:
                                number = 0
                            stuck_error_line = proof_line_number + number
                        else:
                            stuck_error_line = 0
                        is_valid = False

                if percentage == 0:
                    error_details = ['Not processed']
                    is_valid = False

            for node in response_body.get('nodes', []):
                for message in node.get('messages', []):
                    if message['kind'] == 'warning':
                        warning_message, line = message['message'], message['pos']['line']
                        error_details.append(f'Error on line {line}: {warning_message}')
                        error_lines.append(line)
                        is_valid = False
        else:
            print('Wrong theory name')
            is_valid = False
        return is_valid, error_lines, error_details, stuck_error_line

    def shutdown(self):
        self.isabelle.session_stop(session_id=self.session_id)
        self.isabelle.shutdown()
        print('Isabelle session is shut down.')

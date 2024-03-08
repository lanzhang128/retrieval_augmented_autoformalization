from isabelle_client import start_isabelle_server, get_isabelle_client
import json
import time

class Isabelle:
    def __init__(self, isabelle_name="test", session_name="HOL", port=8888, log_file="server.log", dirs=["./Isabelle2023"], verbose=True, options=[], timeout=60):
        self.session_name = session_name
        self.port = port
        self.log_file = log_file
        self.dirs = dirs
        self.verbose = verbose
        self.options = options
        self.timeout = timeout
        server_info, _ = start_isabelle_server(name=isabelle_name, port=self.port, log_file=self.log_file)
        self.isabelle = get_isabelle_client(server_info)
        self.isabelle.session_build(session=self.session_name, dirs=self.dirs, verbose=self.verbose, options=self.options)
        self.start_id = self.isabelle.session_start(session=self.session_name)
    
    def check_HOL_session_syntax_error(self, theory_name, directory):
        start_time = time.time()
        isabelle_response = self.isabelle.use_theories(session_id=self.start_id, session=self.session_name, theories=[theory_name], master_dir=directory, watchdog_timeout=self.timeout)
        time_taken = time.time() - start_time 
        
        has_syntax_error = False
        has_warning_error = False
        error_details = []
        error_lines = []
        finished_response = next((item for item in isabelle_response if item.response_type == 'FINISHED'), None)
        
        error_keywords = ["Type unification failed", "Inner lexical error", "Outer syntax error", "Inner syntax error", "Outer lexical error", "Malformed command syntax", "Undefined type name"]
        warning_keywords = ["Introduced fixed type variable"]
        
        if finished_response:
            response_body = json.loads(finished_response.response_body)
            if response_body.get('errors'):
                for error in response_body['errors']:
                    message, line = error['message'], error['pos']['line']
                    if any(keyword in message for keyword in error_keywords):
                        error_details.append(f"Error on line {line}: {message}")
                        error_lines.append(line)  
                        has_syntax_error = True
            
            for node in response_body.get('nodes', []):
                for message in node.get('messages', []):
                    if message['kind'] == 'warning':
                        warning_message, line = message['message'], message['pos']['line']
                        if any(keyword in warning_message for keyword in warning_keywords):
                            error_details.append(f"Error on line {line}: {warning_message}")
                            error_lines.append(line)  
                            has_warning_error = True
                            has_syntax_error = True
        else:
            print("Wrong theory name")
            return False, [], []
        
        return has_syntax_error, error_details, error_lines
    
    def check_HOL_session_logical_error(self, theory_name, directory, proof_code_file_path):
        start_time = time.time()
        isabelle_response = self.isabelle.use_theories(session_id=self.start_id, session=self.session_name, theories=[theory_name], master_dir=directory,watchdog_timeout=self.timeout)
        inference_time = time.time() - start_time 
        
        error_details = []
        error_lines = []
        stuck_error_line = []
        is_valid = False
        proof_line_number = None
        qed_line_number = None
                
        finished_response = next((item for item in isabelle_response if item.response_type == 'FINISHED'), None)
        
        if finished_response:
            response_body = json.loads(finished_response.response_body)
            percentage = int(response_body['nodes'][0]['status']['percentage']) if response_body.get('nodes') and 'percentage' in response_body['nodes'][0]['status'] else 0
            print(f'The finishing percentage is: {percentage}')
         
            if response_body.get('errors'):
                for error in response_body['errors']:
                    message, line = error['message'], error['pos']['line']
                    error_details.append(f"Error on line {line}: {message}")
                    error_lines.append(line)
            elif percentage != 100:  
                is_valid = False
                with open(proof_code_file_path, 'r') as file:
                    for i, line in enumerate(file, start=1):
                        if "proof -" in line: proof_line_number = i
                        if "qed" in line: qed_line_number = i
                            
                if percentage != 100 and not error_details:
                    number = round((qed_line_number - proof_line_number - 1) * (percentage / 100)) if qed_line_number and proof_line_number else 0
                    stuck_error_line.append(proof_line_number + number if proof_line_number else 0)
            else:
                is_valid = True
                
            for node in response_body.get('nodes', []):
                for message in node.get('messages', []):
                    if message['kind'] == 'warning':
                        warning_message, line = message['message'], message['pos']['line']
                        error_details.append(f"Error on line {line}: {warning_message}")
                        error_lines.append(line)
                        is_valid = False
        else:
            print("Wrong theory name")
            return False, [], [], [], 999999  
        
        return is_valid, error_lines, error_details, stuck_error_line, inference_time
    
    def check_ZF_Session_error(self, theory_name, directory):
        start_time = time.time()
        isabelle_response = self.isabelle.use_theories(session_id=self.start_id, session="ZF", theories=[theory_name], master_dir=directory,watchdog_timeout=self.timeout)
        inference_time = time.time() - start_time 
            
        error_details = []
        error_lines = []
        is_valid = False
        
        finished_response = next((item for item in isabelle_response if item.response_type == 'FINISHED'), None)
        
        if finished_response:
            response_body = json.loads(finished_response.response_body)
            nodes = response_body.get('nodes')
            percentage = int(nodes[-1]['status']['percentage']) if nodes and len(nodes) > 0 and 'percentage' in nodes[-1]['status'] else 0
            print(f'The finishing percentage is: {percentage}')
           
            if response_body.get('errors'):
                for error in response_body['errors']:
                    message, line = error['message'], error['pos']['line']
                    error_details.append(f"Error on line {line}: {message}")
                    error_lines.append(line)
            elif percentage == 0:
                 is_valid = False
                 error_details = ['Not processed']
            else:
                is_valid = True
        else:
            print("Wrong theory name")
            return False, [], [], 999999  
        
        return is_valid, error_lines, error_details, inference_time
    
    def shutdown(self):
        self.isabelle.shutdown()
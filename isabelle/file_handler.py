import ast
import os


def write_to_thy_file(file_path, theory_name, import_thy, text, statement):
    with open(file_path, 'w') as f:
        imports = '\n'.join(import_thy)
        f.write(f'theory {theory_name}\nimports\n{imports}\nbegin\n\n{text}\n\n{statement}')


def write_error_to_file(file_path, is_valid, error_lines, error_details, inference_time):
    with open(file_path, 'w') as file:
        file.write(f'logical validity: {is_valid}\n')
        file.write(f'error lines: {error_lines}\n')
        file.write(f'errors details: {error_details}\n')
        file.write(f'isabelle inference time: {inference_time:.2f}s')


def parse_error_file(error_log_path, thy_file_path):
    if os.path.exists(error_log_path):
        with open(error_log_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if 'error lines: ' in line:
                    error_lines = ast.literal_eval(line[len('error lines: '):])
                if 'errors details: ' in line:
                    errors_details = ast.literal_eval(line[len('errors details: '):])
                if 'logical validity: ' in line:
                    validity = ast.literal_eval(line[len('logical validity: '):])

        with open(thy_file_path, 'r', encoding='utf-8') as f:
            thy_lines = f.readlines()

        all_syntax_error = ''
        first_syntax_error = ''
        for i, line_number in enumerate(error_lines):
            error_code_start_line = error_lines[i] - 1
            error_code_content = []

            for index, line in enumerate(thy_lines[error_code_start_line:], start=error_code_start_line):
                error_code_content.append(line)
                if line.strip() == '' or index == len(thy_lines) - 1:
                    break

            error_code = ''.join(error_code_content)

            syntax_error = errors_details[i].replace('\<^here>', '')

            syntax_error = f'Identified Syntax Error: \n{syntax_error}\n\nCode Causing Error: \n{error_code}'

            all_syntax_error += f'{syntax_error}\n'
            if i == 0:
                first_syntax_error += syntax_error
    else:
        validity = False
        all_syntax_error = ''
        first_syntax_error = ''

    return validity, first_syntax_error, all_syntax_error

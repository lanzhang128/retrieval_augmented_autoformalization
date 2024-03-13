def write_to_thy_file(file_path, theory_name, dependency_file, text, statement):
    with open(file_path, 'w') as f1:
        with open(dependency_file, 'r') as f2:
            dependency = f2.read()
        f1.write(f'theory {theory_name}\n{dependency}\n{text}\n{statement}')


def write_error_to_file(file_path, is_valid, error_lines, error_details, inference_time):
    with open(file_path, 'w') as file:
        file.write(f'logical validity: {is_valid}\n')
        file.write(f'error lines: {error_lines}\n')
        file.write(f'errors details: {error_details}\n')
        file.write(f'isabelle inference time: {inference_time:.2f}s')

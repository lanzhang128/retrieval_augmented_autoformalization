import json
import argparse
import os.path
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from isabelle.file_handler import parse_error_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoformalization with LLMs')
    parser.add_argument('--model_name', default='mistral',
                        help='name of the LLM')
    parser.add_argument('--round', default='1.1',
                        help='round indicator:\n0 zero-shot\n1 3-shot\n2 retrieved prompt')
    parser.add_argument('--result_json', default='results/mistral_t_0_1.1.json',
                        help='json file to store results')
    parser.add_argument('--test_json', default='results/mistral_t_auto_0.json',
                        help='json file containing test data')
    args = parser.parse_args()

    model_name = args.model_name
    mode = args.round

    with open(args.test_json, 'r', encoding='utf-8') as f:
        json_dic = json.load(f)

    with open(f'prompts/round_{mode}.json', 'r', encoding='utf-8') as f:
        prompt = json.load(f)

    result_dic = {}
    if model_name == 'mistral' or model_name == 'mixtral' or model_name == 'llemma-7B' or model_name == 'llemma-34B':
        if model_name == 'mistral':
            model_id = 'mistralai/Mistral-7B-Instruct-v0.2'
        elif model_name == 'llemma-7B':
            model_id = 'EleutherAI/llemma_7b'
        elif model_name == 'llemma-34B':
            model_id = 'EleutherAI/llemma_34b'
        else:
            model_id = 'mistralai/Mixtral-8x7B-Instruct-v0.1'

        model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

        print(f'Generating with {model_name} and {model.dtype} precision..')
        count = 0
        start_time = time.time()

        for key in json_dic.keys():
            text = json_dic[key]['text']
            statement = json_dic[key]['statement']

            thy_file_path = os.path.join(args.result_json[:-5], f'test_{key}.thy')
            error_log_path = os.path.join(args.result_json[:-5], f'test_{key}.error.log')
            validity, syntax_error = parse_error_file(error_log_path, thy_file_path)

            if mode == 'refine' and validity:
                refined_code = statement
            else:
                messages = []
                content = prompt['user'].replace('{text}', text)
                content = content.replace('{isabelle_code}', statement)
                content = content.replace('{syntax_error}', syntax_error)
                messages.append({'role': 'user', 'content': content})

                encodeds = tokenizer.apply_chat_template(messages, return_tensors='pt')
                model_inputs = encodeds.to('cuda')
                generated_ids = model.generate(model_inputs, max_new_tokens=1000,
                                               do_sample=False, pad_token_id=tokenizer.eos_token_id)
                decoded = tokenizer.batch_decode(generated_ids)

                refined_code = decoded[0]
                template = tokenizer.batch_decode(encodeds)[0]
                refined_code = refined_code[refined_code.find(template) + len(template):]
                if refined_code[-4:] == '</s>':
                    refined_code = refined_code[:-4]

            result_dic[key] = {'text': text, 'statement': refined_code}

            count += 1
            if count % 1 == 0:
                print(f'Finished {count}/{len(json_dic)} samples, time elapsed {time.time() - start_time:.2f}s.')

        with open(args.result_json, 'w', encoding='utf-8') as f:
            json.dump(result_dic, f, ensure_ascii=False, indent=4)
    else:
        raise ValueError(f'{model_name} is currently not supported.')

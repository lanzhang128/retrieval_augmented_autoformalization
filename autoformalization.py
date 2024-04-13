import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoformalization with LLMs')
    parser.add_argument('--model_name', default='mistral',
                        help='name of the LLM')
    parser.add_argument('--mode', type=int, default=0,
                        help='prompt mode:\n0 zero-shot\n1 3-shot\n2 retrieved prompt')
    parser.add_argument('--result_json', default='results/mistral_0_auto.json',
                        help='json file to store results')
    parser.add_argument('--test_json', default='data/IsarMathLib/mistral_inf/test.json',
                        help='json file containing test data')
    parser.add_argument('--shot_json', default='data/IsarMathLib/3-shot.json',
                        help='json file containing few-shot data')
    parser.add_argument('--retrieval_folder', default='results/BM25_retrieval_tis',
                        help='retrieval results folder')
    args = parser.parse_args()

    model_name = args.model_name
    mode = args.mode
    retrieval_folder = args.retrieval_folder

    with open(args.test_json, 'r', encoding='utf-8') as f:
        json_dic = json.load(f)

    with open('prompts/autoformalization.json', 'r', encoding='utf-8') as f:
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

        print(f'Autoformalizing test set with {model_name} and {model.dtype} precision..')
        for key in tqdm(json_dic.keys()):
            text = json_dic[key]['text']
            messages = []
            if mode != 0:
                if mode == 1:
                    with open(args.shot_json, 'r', encoding='utf-8') as f:
                        temp_dic = json.load(f)
                elif mode == 2:
                    with open(f'{retrieval_folder}/{key}.json', 'r', encoding='utf-8') as f:
                        temp_dic = json.load(f)
                for i in temp_dic.keys():
                    temp_text = temp_dic[i]['text']
                    messages.append({'role': 'user', 'content': prompt['user'].replace('{text}', temp_text)})
                    temp_statement = temp_dic[i]['statement']
                    messages.append(
                        {'role': 'assistant', 'content': prompt['assistant'].replace('{statement}', temp_statement)})

            messages.append({'role': 'user', 'content': prompt['user'].replace('{text}', text)})

            encodeds = tokenizer.apply_chat_template(messages, return_tensors='pt')
            model_inputs = encodeds.to('cuda')
            generated_ids = model.generate(model_inputs, max_new_tokens=1000,
                                           do_sample=False, pad_token_id=tokenizer.eos_token_id)
            decoded = tokenizer.batch_decode(generated_ids)

            formal = decoded[0]
            template = tokenizer.batch_decode(encodeds)[0]
            formal = formal[formal.find(template) + len(template):]
            if formal[-4:] == '</s>':
                formal = formal[:-4]

            result_dic[key] = {'text': text, 'statement': formal}

        with open(args.result_json, 'w', encoding='utf-8') as f:
            json.dump(result_dic, f, ensure_ascii=False, indent=4)
    else:
        raise ValueError(f'{model_name} is currently not supported.')

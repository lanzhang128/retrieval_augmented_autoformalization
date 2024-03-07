import os
import json
import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Informalization with HuggingFace LLMs')
    parser.add_argument('--model_id', default='mistralai/Mistral-7B-Instruct-v0.2', help='name of the LLM')
    parser.add_argument('--data_folder', default='data/IsarMathLib/mistral_inf', help='the results storage folder')
    args = parser.parse_args()

    if not os.path.exists(args.data_folder):
        os.mkdir(args.data_folder)

    model_id = args.model_id
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    with open('prompts/informalization.json', 'r', encoding='utf-8') as f:
        prompt = json.load(f)

    for split in ['train', 'test']:
        print(f'Informalizing {split} set with {model_id} and {model.dtype} precision.')
        with open(f'data/IsarMathLib/extraction/{split}.json', 'r', encoding='utf-8') as f:
            json_dic = json.load(f)
        count = 0
        start_time = time.time()

        for key in json_dic.keys():
            statement = json_dic[key]['statement']
            messages = [{'role': 'user', 'content': prompt['user'].replace('{statement}', statement)}]
            encodeds = tokenizer.apply_chat_template(messages, return_tensors='pt')
            model_inputs = encodeds.to('cuda')
            generated_ids = model.generate(model_inputs, max_new_tokens=1000,
                                           do_sample=True, pad_token_id=tokenizer.eos_token_id)
            decoded = tokenizer.batch_decode(generated_ids)

            informal = decoded[0]
            template = tokenizer.batch_decode(encodeds)[0]

            if 'Mistral' in model_id:
                informal = informal[informal.find(template)+len(template):]
                if informal[-4:] == '</s>':
                    informal = informal[:-4]

            json_dic[key]['informal'] = informal
            count += 1
            if count % 1 == 0:
                print(f'Finished {count}/{len(json_dic)} samples, time elapsed {time.time()-start_time:.2f}s.')

        with open(f'{args.data_folder}/{split}.json', 'w', encoding='utf-8') as f:
            json.dump(json_dic, f, ensure_ascii=False, indent=4)

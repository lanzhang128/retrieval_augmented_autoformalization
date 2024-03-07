import os
import json
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrieving documents using HuggingFace Models')
    parser.add_argument('--org_json', default='data/IsarMathLib/mistral_inf/train.json',
                        help='original json file to construct knowledge base')
    args = parser.parse_args()

    with open(args.org_json, 'r', encoding='utf-8') as f:
        json_dic = json.load(f)

    folders = ['text', 'text+statement', 'informal+statement', 'text+informal+statement']
    if not os.path.exists(f'data/KB'):
        os.mkdir(f'data/KB')
    for folder in folders:
        if not os.path.exists(f'data/KB/{folder}'):
            os.mkdir(f'data/KB/{folder}')

    for key in json_dic.keys():
        dic = json_dic[key]
        for folder in folders:
            temp_dic = {}
            for subkey in folder.split('+'):
                temp_dic[subkey] = dic[subkey]
            with open(f'data/KB/{folder}/{key}.txt', 'w', encoding='utf-8') as f:
                f.write(json.dumps(temp_dic))

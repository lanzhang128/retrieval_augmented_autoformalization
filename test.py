import os
import json
import argparse
from evaluation.utils import postprocess_model_output, preprocess_metric_input
from evaluation.common_metric import BLEU, ChrF, RUBY
from evaluation.model_based_metric import CodeBERTScore


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model evaluation')
    parser.add_argument('--ref_json', default='data/IsarMathLib/extraction/test.json',
                        help='json file that stores reference data')
    parser.add_argument('--result_json', default='results/mistral_0_auto.json',
                        help='json file that stores results')
    parser.add_argument('--metrics', nargs='+', default=['BLEU', 'ChrF', 'RUBY', 'CodeBERTScore'],
                        help='metrics to evaluate results')
    args = parser.parse_args()

    metric_class = {
        'BLEU': BLEU(),
        'ChrF': ChrF(),
        'RUBY': RUBY()
    }

    with open(args.ref_json, 'r', encoding='utf-8') as f:
        ref_json = json.load(f)

    if os.path.exists(args.result_json[:-4]+'post.json'):
        with open(args.result_json[:-4]+'post.json', 'r', encoding='utf-8') as f:
            can_json = json.load(f)
    else:
        with open(args.result_json, 'r', encoding='utf-8') as f:
            can_json = json.load(f)
        for key in can_json.keys():
            can_json[key]['statement'] = postprocess_model_output(can_json[key]['statement'])
        with open(args.result_json[:-4]+'post.json', 'w', encoding='utf-8') as f:
            json.dump(can_json, f, ensure_ascii=False, indent=4)

    nl_texts = []
    ref_texts = []
    can_texts = []
    for key in ref_json.keys():
        nl_texts.append(preprocess_metric_input(ref_json[key]['text']))
        ref_texts.append(preprocess_metric_input(ref_json[key]['statement']))
        can_texts.append(preprocess_metric_input(can_json[key]['statement']))

    score_dic = {}
    for metric in args.metrics:
        print(f'Evaluating with {metric} metric.')
        if metric == 'CodeBERTScore':
            score_dic.update(CodeBERTScore().evaluate(ref_texts, can_texts, nl_texts))
        else:
            score_dic.update(metric_class[metric].evaluate(ref_texts, can_texts))
    print(score_dic)

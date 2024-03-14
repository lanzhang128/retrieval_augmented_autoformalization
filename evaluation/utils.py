import re


def postprocess_model_output(text):
    res = re.sub('\(\*.*?\*\)', '', text, flags=re.DOTALL)
    if 'proof' in res:
        res = res[:res.find('proof')]
    if 'Note:' in res:
        res = res[:res.find('Note:')]
    return res


def preprocess_metric_input(text):
    return ' '.join(text.split())

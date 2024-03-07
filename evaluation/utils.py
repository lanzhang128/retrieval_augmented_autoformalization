import re


def postprocess_model_output(text):
    res = re.sub('\(\*.*?\*\)', '', text, flags=re.DOTALL)
    if 'proof' in res:
        res = res[:res.find('proof')]
    if 'Note:' in res:
        res = res[:res.find('Note:')]
    res.replace('\n\n', '\n')
    return res


def preprocess_metric_input(text):
    res = text.replace('\n', ' ')

    temp = res.split()
    res = []
    for i in temp:
        if i != '':
            res.append(i)
    res = ' '.join(res)
    return res

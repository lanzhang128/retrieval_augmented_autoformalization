import os
import json
import re
import random


def parse_thy_file(file_name, types, source):
    assert file_name[-4:] == '.thy'
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.read()

    comments = []
    start = 0
    while start < len(lines) - 1:
        if lines[start:start+2] == '(*':
            count = 1
            end = start + 2
            while end < len(lines) - 1 and count > 0:
                if lines[end:end+2] == '*)':
                    count -= 1
                    end = end + 2
                elif lines[end:end + 2] == '(*':
                    count += 1
                    end = end + 2
                else:
                    end = end + 1
            comments.append((start, end))
            start = end
        start += 1

    new_lines = ''
    start = 0
    for (l, r) in comments:
        new_lines += lines[start:l]
        start = r
    new_lines += lines[start:]

    lines = [_ +'\n' for _ in new_lines.split('\n')]

    chunks = []
    start = 0
    while start < len(lines):
        end = start
        while end < len(lines):
            if lines[end].rstrip().replace(' ', '') == '':
                break
            else:
                end += 1
        if end == len(lines):
            chunks.append(''.join(lines[start:end]))
        elif end != start:
            chunks.append(''.join(lines[start:end]))
        start = end + 1

    items = []
    for i in range(len(chunks)):
        temp = chunks[i].split()
        if temp:
            if temp[0] in types:
                item = {'type': temp[0]}
                if chunks[i-1][:4] == 'text':
                    item['text'] = chunks[i-1]
                else:
                    item['text'] = ''
                if 'assumes' in chunks[i]:
                    if 'shows' in chunks[i]:
                        item['assumes'] = chunks[i][chunks[i].find('assumes'):chunks[i].find('shows')]
                    elif 'obtains' in chunks[i]:
                        item['assumes'] = chunks[i][chunks[i].find('assumes'):chunks[i].find('obtains')]
                    else:
                        item['assumes'] = chunks[i][chunks[i].find('assumes'):]
                else:
                    item['assumes'] = ''
                item['using'] = []
                for s in re.findall('using.*?by', chunks[i], flags=re.DOTALL):
                    s = s[:-2]
                    s = s.replace('\n', '')
                    s = s.replace('unfolding', '')
                    s = s.replace('using', '')
                    s = s.split()
                    item['using'] += s
                item['using'] = list(dict.fromkeys(item['using']))
                if 'proof' in chunks[i] and 'qed' in chunks[i]:
                    item['statement'] = chunks[i][:chunks[i].find('proof')]
                    item['proof'] = chunks[i][chunks[i].find('proof'):]
                else:
                    if 'using' in chunks[i]:
                        item['statement'] = chunks[i][:chunks[i].find('using')]
                        item['proof'] = chunks[i][chunks[i].find('using'):]
                    else:
                        item['statement'] = chunks[i]
                        item['proof'] = ''
                item['source'] = f'{source}/{os.path.basename(file_name)}'
                items.append(item)
    return items


if __name__ == '__main__':
    data_id = 0
    json_dic = {}
    types = ['lemma', 'definition', 'corollary', 'theorem']
    for root, _, files in os.walk('IsarMathLib'):
        files.sort()
        for file in files:
            if file[-4:] == '.thy':
                items = parse_thy_file(os.path.join(root, file), types, 'IsarMathLib')
                for item in items:
                    if item['text'] == '':
                        continue
                    item['id'] = data_id
                    json_dic[f'{data_id}'] = item
                    data_id += 1

    if not os.path.exists('data/IsarMathLib'):
        os.mkdir('data/IsarMathLib')
    if not os.path.exists('data/IsarMathLib/extraction'):
        os.mkdir('data/IsarMathLib/extraction')

    train_dic = {}
    test_dic = {}

    random.seed(2024)
    test_ids = random.sample(range(len(json_dic)), int(0.1 * len(json_dic)))
    train_ids = list(set(range(len(json_dic))) - set(test_ids))

    for i in range(len(train_ids)):
        train_dic[f'{i}'] = json_dic[f'{train_ids[i]}']
    for i in range(len(test_ids)):
        test_dic[f'{i}'] = json_dic[f'{test_ids[i]}']

    with open('data/IsarMathLib/extraction/all.json', 'w', encoding='utf-8') as f:
        json.dump(json_dic, f, ensure_ascii=False, indent=4)
    with open('data/IsarMathLib/extraction/train.json', 'w', encoding='utf-8') as f:
        json.dump(train_dic, f, ensure_ascii=False, indent=4)
    with open('data/IsarMathLib/extraction/test.json', 'w', encoding='utf-8') as f:
        json.dump(test_dic, f, ensure_ascii=False, indent=4)

import os
import json
import time
import argparse
from llama_index import SimpleDirectoryReader, ServiceContext
from llama_index.retrievers import BM25Retriever


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BM25 retrieval')
    parser.add_argument('--json_file', default='results/mistral_0_auto.json',
                        help='json file containing information for test')
    parser.add_argument('--mode', type=int, default=1,
                        help='retrieval mode:\n0 text\n1 text+statement')
    parser.add_argument('--org_json', default='data/IsarMathLib/mistral_inf/train.json',
                        help='original json file to construct knowledge base')
    parser.add_argument('--kb_folder', default='data/KB/text+informal+statement',
                        help='data folder to be used as knowledge base')
    parser.add_argument('--retrieval_folder', default='results/BM25_retrieval_tis',
                        help='retrieval results folder')
    args = parser.parse_args()

    mode = args.mode
    if not os.path.exists(args.retrieval_folder):
        os.mkdir(args.retrieval_folder)

    service_context = ServiceContext.from_defaults(embed_model=None, llm=None)
    documents = SimpleDirectoryReader(args.kb_folder).load_data()
    nodes = service_context.node_parser.get_nodes_from_documents(documents)
    retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=3)

    with open(args.json_file, 'r', encoding='utf-8') as f:
        json_dic = json.load(f)

    with open(args.org_json, 'r', encoding='utf-8') as f:
        kb_dic = json.load(f)

    count = 0
    start_time = time.time()
    for key in json_dic:
        text = json_dic[key]['text']
        statement = json_dic[key]['statement']
        if mode == 0:
            nodes = retriever.retrieve(f'text: {text}')
        elif mode == 1:
            nodes = retriever.retrieve(f'text: {text}\nstatement: {statement}')
        ids = []
        for node in nodes:
            ids.append(int(os.path.basename(node.metadata['file_path'])[:-4]))
        res = {}
        for i in range(len(ids)):
            res[i] = kb_dic[f'{ids[i]}']
        with open(f'{args.retrieval_folder}/{key}.json', 'w') as f:
            json.dump(res, f, ensure_ascii=False, indent=4)
        count += 1
        if count % 20 == 0:
            print(f'Finished {count}/{len(json_dic)} samples, time elapsed {time.time() - start_time:.2f}s.')

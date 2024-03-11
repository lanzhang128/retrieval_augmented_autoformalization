from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class CodeBERTScore:
    def __init__(self):
        model_id = 'EleutherAI/llemma_7b'
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.float16)
        self.model = self.model.model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    @torch.no_grad()
    def evaluate(self, ref_texts, can_texts, nl_texts):
        score_dic = {}
        if len(ref_texts) == 0:
            raise ValueError('References are empty!')
        if len(can_texts) == 0:
            raise ValueError('Candidates are empty!')

        Ps, Rs, F1s, F3s = [], [], [], []
        for text, ref, can in zip(tqdm(nl_texts), ref_texts, can_texts):
            if len(can) == 0:
                Ps.append(0)
                Rs.append(0)
                F1s.append(0)
                F3s.append(0)
            else:
                text_tokens_length = len(self.tokenizer(text)['input_ids'])
                ref_inputs = self.tokenizer(text, ref, return_tensors='pt').to('cuda')
                can_inputs = self.tokenizer(text, can, return_tensors='pt').to('cuda')

                ref_vectors = self.model(**ref_inputs)['last_hidden_state'][0, text_tokens_length + 1:, :]
                can_vectors = self.model(**can_inputs)['last_hidden_state'][0, text_tokens_length + 1:, :]
                ref_vectors = ref_vectors / torch.linalg.vector_norm(ref_vectors, dim=1, keepdim=True)
                can_vectors = can_vectors / torch.linalg.vector_norm(can_vectors, dim=1, keepdim=True)

                cosine_similarity = torch.mm(ref_vectors, can_vectors.T)
                token_precision = torch.max(cosine_similarity, dim=0)[0]
                token_recall = torch.max(cosine_similarity, dim=1)[0]
                P, R = torch.mean(token_precision).item(), torch.mean(token_recall).item()
                F1 = 2 * P * R / (P + R)
                F3 = 10 * P * R / (9 * P + R)
                Ps.append(P)
                Rs.append(R)
                F1s.append(F1)
                F3s.append(F3)

        score_dic['CodeBERTScore-F1'] = sum(F1s) / len(F1s)
        score_dic['CodeBERTScore-F3'] = sum(F3s) / len(F3s)
        return score_dic

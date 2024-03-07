import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.chrf_score import corpus_chrf
from nltk.metrics.distance import edit_distance


class BLEU:
    @staticmethod
    def evaluate(ref_texts, can_texts):
        score_dic = {}
        bleu_references = [[ref.split()] for ref in ref_texts]
        bleu_candidates = [can.split() for can in can_texts]
        score_dic['BLEU-1'] = corpus_bleu(bleu_references, bleu_candidates, weights=(1, 0, 0, 0))
        score_dic['BLEU-2'] = corpus_bleu(bleu_references, bleu_candidates, weights=(0.5, 0.5, 0, 0))
        score_dic['BLEU-4'] = corpus_bleu(bleu_references, bleu_candidates, weights=(0.25, 0.25, 0.25, 0.25))
        return score_dic


class ChrF:
    @staticmethod
    def evaluate(ref_texts, can_texts):
        score_dic = {}
        chrf_references = [ref.split() for ref in ref_texts]
        chrf_candidates = [can.split() for can in can_texts]
        score_dic['ChrF'] = corpus_chrf(chrf_references, chrf_candidates)
        return score_dic


class RUBY:
    @staticmethod
    def evaluate(ref_texts, can_texts):
        score_dic = {}
        if len(ref_texts) == 0:
            raise ValueError('RUBY references are empty!')
        if len(can_texts) == 0:
            raise ValueError('RUBY candidates are empty!')
        scores = []
        for ref, can in zip(ref_texts, can_texts):
            sed_score = edit_distance(ref, can)
            scores.append(1 - sed_score / max(len(ref), len(can)))
        score_dic['RUBY'] = np.mean(scores)
        return score_dic

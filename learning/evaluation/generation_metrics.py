import os
from IPython import embed
from typing import List, Tuple
from nltk.translate.bleu_score import sentence_bleu

import torch
from bert_score import BERTScorer

scorer = BERTScorer(lang="en", rescale_with_baseline=True)

def calc_bleu(references: torch.tensor, hypothesis: torch.tensor, tokenizer):
    """
    calc blue-4
    """
    ref = [tokenizer.decode(tk) for tk in references.tolist() if tk != 0 and tk != tokenizer.eos_token_id] # TODO: ideally this entorphy should be calculated over samples
    hyp = [tokenizer.decode(tk) for tk in hypothesis.tolist() if tk != 0 and tk != tokenizer.eos_token_id] # TODO: ideally this entorphy should be calculated over samples
    return sentence_bleu([ref], hyp, weights=(0.25, 0.25, 0.25, 0.25)) # Warning: only len(hyp) >= 4


def calc_bert_score(references: List[str], hypothesis: List[str]) -> Tuple[List[float], List[float], List[float]]:
    """
    ref: https://github.com/Tiiiger/bert_score/blob/master/bert_score/scorer.py#L169
    """
    P, R, F1 = scorer.score(hypothesis, references)
    return P.tolist(), R.tolist(), F1.tolist()

def calc_perplexity(sent_prob: float, sent_len:int):
    if sent_len == 0.:
        return 1.
    if  sent_prob == 0:
        return 1. # TODO not correct
    return sent_prob ** (-1/sent_len)

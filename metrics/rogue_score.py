# import torch
# import torch.nn as nn
# from typing import Dict, List
import torch
import torch.nn as nn
from typing import Dict, List

class RougeN(nn.Module):
    def __init__(self, n:int=1):
        super().__init__()
        self.n = n
        
    def _get_ngrams(self, sequence:List[int], n:int)-> List[tuple]:
        return [tuple(sequence[i:i+n] ) for i in range(len(sequence)-n+1)]

    def forward(self, reference:torch.Tensor, candidate:torch.Tensor) -> Dict[str, float]:
        ref_tokens = reference.split()
        cand_tokens = candidate.split()
        ref_ngrams = self._get_ngrams(ref_tokens, self.n)
        cand_ngrams =  self._get_ngrams(cand_tokens, self.n)
        ref_set = set(ref_ngrams)
        cand_set = set(cand_ngrams)
        overlap = len(ref_set & cand_set)
        recalll = overlap/len(ref_set) if len(ref_set) > 0 else 0.0
        precision = overlap /len(cand_set) if len(cand_set) > 0 else 0.0
        f1 = 2*recalll*precision /(recalll+precision) if (recalll+precision) > 0 else 0.0
        return {"precision": precision, "recall": recalll, "f1": f1}

sentence1 = "Hello how you"
sentence2 = "Hello who are you, lets meet tommorow"
rg_n = RougeN()
print(rg_n(sentence1, sentence2))

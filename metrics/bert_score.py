# import torch
# from transformers import AutoTokenizer, AutoModel
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any

class BERTScore:
    def __init__(self, model_name="bert-base-uncased", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        
    def _embed(self, sentences:List[str]):
        enc = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
        print("Encoded: ",enc)
        enc = {k:v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc)
        embeddings = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).expand(embeddings.size()).float()
        embeddings = embeddings*mask
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        return embeddings, mask.squeeze(-1)

    def score(self, candidates, references):
        cand_emb, cand_mask = self._embed(candidates)
        ref_emb, ref_mask = self._embed(references)
        batch_size = cand_emb.size(0)
        precisions, recalls, f1s = [],[],[]
        for i in range(batch_size):
            cand_vecs = cand_emb[i][cand_mask[i]>0]
            ref_vecs = ref_emb[i][ref_mask[i]>0]
            sim = torch.matmul(cand_vecs, ref_vecs.T)
            precision = sim.max(dim=1).values.mean()
            recall = sim.max(dim=0).values.mean()
            f1 = 2*precision*recall /(precision+recall+1e-8)
            precisions.append(precision.item())
            recalls.append(recall.item())
            f1s.append(f1.item())
        return precisions, recalls,f1s
bert_score = BERTScore(model_name="bert-base-uncased")


candidates = ["the cat is on the mat", "there is a cat on the mat"]
references = ["a cat is sitting on the mat", "a cat is on the mat"]
P, R, F1 = bert_score.score(candidates, references)

for i in range(len(candidates)):
    print(f"Candidate: {candidates[i]}")
    print(f"References: {references[i]}")
    print(f"Precision: {P[i]:.4f}, Recall: {R[i]:.4f} F1: {F1[i]:.4f}")
    print("------")
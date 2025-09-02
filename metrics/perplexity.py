# import torch
# import torch.nn as nn
import torch
import torch.nn as nn


class Perplexity:
    def __init__(self, ignore_index:int=-100, reduction:str="mean"):
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
        
    def __call__(self, logits:torch.Tensor, targets:torch.Tensor) ->float:
        vocab_size = logits.size(-1)
        loss = self.loss_fn(logits.view(-1, vocab_size), targets.view(-1))
        perplexity = torch.exp(loss)
        return perplexity.item()

if __name__ == "__main__":
    batch_size, seq_len, vocab_size = 2, 5, 10
    torch.manual_seed(0)
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    ppl_metric = Perplexity(ignore_index=-100)
    ppl = ppl_metric(logits, targets)
    print(ppl)
    print(f"Perplexity: {ppl:.4f}")
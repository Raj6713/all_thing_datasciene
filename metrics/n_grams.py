from typing import List

class NGrams:
    def __init__(self,do_lowercase=False):
        self.do_lowercase=do_lowercase
        
    def create_n_grams(self,sentence:str,n:int)->List[List[str]]:
        if self.do_lowercase:
            tokens = sentence.lower().split(" ")
        else:
            tokens = sentence.split(" ")
        print(tokens)
        assert type(n)==int, "The value should be integer."
        n_gs=[]
        for n_level in range(1,n):
            print("N level: ",n_level)
            items = [" ".join(tokens[i:i+n_level]) for i in range(0, len(tokens)-n_level)]
            print(items)
        
        
sentence = "Hey How are you doing, we are going to create the fist AI model someday"
ngrams = NGrams(do_lowercase=True)
ngrams.create_n_grams(sentence,3)

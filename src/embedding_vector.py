from typing import List, Literal
import numpy as np 
import SentenceTransformer

class EmbeddingVectorizer:
    
    def __init__(self, model_name : str = 'intfloat/multilingual-e5-base', normalize = True):
        
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize
        
    def format_inputs(self, texts : List[str], mode : Literal['query', 'passage']) -> List[str]:  # model E5 trained with these prefix: query for finding and passage for storing
        
        if mode not in {'query', 'passage'}: # use set to reduce complexity
            raise ValueError('mode not accepted')
        return [f'{mode} : ' + text.strip() for text in texts] # add prefix in front of each doc
    
    def transform(self, texts, mode : Literal['query', 'passage'] = 'query') -> List[List[str]]: # always set query like the instruction of model
        
        inputs = self.format_inputs(texts, mode)
        embeddings = self.model.encode(inputs, normalize = self.normalize)
        return np.array(embeddings.tolist())
    
    
    
    
        
    
    
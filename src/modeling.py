from pytools import memoize_method
import torch
import torch.nn.functional as F
import modeling_util
import pytorch_pretrained_bert
from transformers import AutoTokenizer, RwkvModel
from sentence_transformers import SentenceTransformer, util

class Rwkv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rwkv = RwkvModel.from_pretrained("RWKV/rwkv-4-169m-pile")
        self.tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-4-169m-pile")
        '''self.rwkv = RwkvModel.from_pretrained("sgugger/rwkv-430M-pile")
        self.tokenizer = AutoTokenizer.from_pretrained("sgugger/rwkv-430M-pile")'''
        self.vocab = self.tokenizer.get_vocab()
        self.RWKV_SIZE = 768
    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            if state[key].requires_grad:
                state[key] = state[key].data
            else:
                del state[key]
        torch.save(state, path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)

    @memoize_method
    def tokenize(self, text):
        toks = self.tokenizer.encode(text, add_special_tokens=False)
        return toks

    def encode_rwkv(self, query_tok, query_mask, doc_tok, doc_mask ):
       
        
        SEPS = torch.full_like(query_tok[:, :1], self.tokenizer.eos_token_id)
        
        Query = torch.cat([torch.tensor(self.tokenize('query:')) for _ in range(len(query_tok))], dim=0).view(-1,2).to('cuda:0')
        Doc = torch.cat([torch.tensor(self.tokenize('document:')) for _ in range(len(query_tok))], dim=0).view(-1,2).to('cuda:0')
        d_ONES = torch.ones_like(query_mask[:, :2])
        ONES = torch.ones_like(query_mask[:, :1])
        toks = torch.cat([Query, query_tok, Doc, doc_tok ,SEPS], dim=1)
        mask = torch.cat([d_ONES, query_mask, d_ONES, doc_mask ,ONES], dim=1)
        toks[toks == -1] = 0 
        result = self.rwkv(input_ids=toks, attention_mask=mask)
        result=torch.mean(result.last_hidden_state,dim = 1)
        return result

class Vanilla_rwkv(Rwkv):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.05)
        self.linear = torch.nn.Linear(self.RWKV_SIZE, 1)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        emb_reps = self.encode_rwkv(query_tok, query_mask, doc_tok, doc_mask)
        x = self.linear(self.dropout(emb_reps))
        return x




from transformers import AutoTokenizer,RwkvModel
import torch
import numpy as np
from mteb import MTEB
TASK_LIST_RERANKING = [
   
    "MindSmallReranking",
   
]
TASK_LIST_RETRIEVAL = [
    "ClimateFEVER",
    "CQADupstackEnglishRetrieval",
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MSMARCO",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
]

class MyModel(torch.nn.Module):
    def __init__(self):                     
        super().__init__()
        self.rwkv = RwkvModel.from_pretrained("RWKV/rwkv-4-world-430m")
        self.tokenizer =  AutoTokenizer.from_pretrained("RWKV/rwkv-4-world-430m")
        self.tokenizer.pad_token = self.tokenizer.eos_token
    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)

    def encode(self, sentences: list[str], batch_size: 64 , **kwargs) -> np.ndarray:
        all_embeddings = []

        self.rwkv.eval()
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch_sentences = sentences[i:i + batch_size]
                inputs = self.tokenizer(text=batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.cuda() for k, v in inputs.items()}
                result = self.rwkv(**inputs, output_hidden_states=True)
                print(result.last_hidden_state.shape)
                batch_embeddings = torch.mean(result.last_hidden_state, dim=1).cpu().numpy()
                all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings)

model = MyModel()
model.cuda()

evaluation = MTEB(tasks=TASK_LIST_RETRIEVAL)
evaluation.run(model,output_folder="retrival/",batch_size = 64)





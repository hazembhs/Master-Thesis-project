import math
import torch
from collections import defaultdict
import torch.nn.functional as F



def subbatch(toks, maxlen):
    _, DLEN = toks.shape[:2]
    SUBBATCH = math.ceil(DLEN / maxlen)
    S = math.ceil(DLEN / SUBBATCH) if SUBBATCH > 0 else 0 
    stack = []
    if SUBBATCH == 1:
        return toks, SUBBATCH
    else:
        for s in range(SUBBATCH):
            stack.append(toks[:, s*S:(s+1)*S])
            if stack[-1].shape[1] != S:
                nulls = torch.zeros_like(toks[:, :S - stack[-1].shape[1]])
                stack[-1] = torch.cat([stack[-1], nulls], dim=1)
        return torch.cat(stack, dim=0), SUBBATCH

def un_subbatch(embed, toks, maxlen):
    BATCH, DLEN = toks.shape[:2]
    SUBBATCH = math.ceil(DLEN / maxlen)
    if SUBBATCH == 1:
        return embed
    else:
        embed_stack = []
        for b in range(SUBBATCH):
            embed_stack.append(embed[b*BATCH:(b+1)*BATCH])
        embed = torch.cat(embed_stack, dim=1)
        embed = embed[:, :DLEN]
        return embed


def compute_rr_at_k(ranked_list, qrels, k=100):
    ranked_list = list(ranked_list) 
    for rank, doc_id in enumerate(ranked_list[:k], 1):
        
        if qrels.get(doc_id, 0) > 0: 

            return 1.0 / rank 
    return 0.0 

def compute_mrr_at_100(rerank_run, valid_qrels):
    total_queries = len(rerank_run)
    mrr_sum = 0.0 
    for query_id, ranked_list in rerank_run.items():
        rr = compute_rr_at_k(ranked_list, valid_qrels.get(query_id, {}))
        mrr_sum += rr

    mrr_at_100 = mrr_sum / total_queries
    return mrr_at_100

def compute_cosine_similarity(query_dict, doc_dict, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = defaultdict(dict)
    query_ids = list(query_dict.keys())
    doc_ids = list(doc_dict.keys())

    query_tensors = torch.stack([query_dict[qid] for qid in query_ids]).to(device)
    doc_tensors1 = torch.stack([doc_dict[did][0] for did in doc_ids]).to(device)
    doc_tensors2 = torch.stack([doc_dict[did][1] for did in doc_ids]).to(device)

    for i in range(0, len(query_ids), batch_size):
        query_batch = query_tensors[i:i + batch_size]
        for j in range(0, len(doc_ids), batch_size):
            doc_batch1 = doc_tensors1[j:j + batch_size]
            doc_batch2 = doc_tensors2[j:j + batch_size]

            similarity_batch1 = F.cosine_similarity(query_batch.unsqueeze(1), doc_batch1.unsqueeze(0), dim=-1)
            similarity_batch2 = F.cosine_similarity(query_batch.unsqueeze(1), doc_batch2.unsqueeze(0), dim=-1)
            
            mean_similarity_batch = (similarity_batch1 + similarity_batch2) / 2.0

            for k, qid in enumerate(query_ids[i:i + batch_size]):
                for l, did in enumerate(doc_ids[j:j + batch_size]):
                    results[qid][did] = mean_similarity_batch[k, l].item()

    return results
def compute_cosine_similarity_between_dense_vecs(query_dict, doc_dict, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = defaultdict(dict)
    query_ids = list(query_dict.keys())
    doc_ids = list(doc_dict.keys())

    query_tensors = torch.stack([query_dict[qid] for qid in query_ids]).to(device)
    doc_tensors1 = torch.stack([torch.tensor(doc_dict[did]) for did in doc_ids]).to(device)

    for i in range(0, len(query_ids), batch_size):
        query_batch = query_tensors[i:i + batch_size]
        for j in range(0, len(doc_ids), batch_size):
            doc_batch1 = doc_tensors1[j:j + batch_size]
           
            similarity_batch1 = F.cosine_similarity(query_batch.unsqueeze(1), doc_batch1.unsqueeze(0), dim=-1)
            
            for k, qid in enumerate(query_ids[i:i + batch_size]):
                for l, did in enumerate(doc_ids[j:j + batch_size]):
                    results[qid][did] = similarity_batch1[k, l].item()

    return results




def listwise_softmax_cross_entropy_loss(y_true, y_pred):
    """
    Listwise softmax cross entropy loss
    """
    loss = 0.0
    batch_size, num_docs = y_true.shape

    for i in range(batch_size):
        softmax_scores = F.softmax(y_pred[i], dim=0)
        log_softmax_scores = torch.log(softmax_scores)
        loss += -torch.sum(y_true[i] * log_softmax_scores)

    return loss / batch_size




def pairwise_logistic_loss(y_true, y_pred):
    batch_size = y_true.size(0) 
    total_loss = 0.0

    for i in range(1, batch_size):  
      
        loss = torch.log(1 + torch.exp(y_pred[i] - y_pred[0]))  
        total_loss += loss

    return total_loss 


def pointwise_cross_entropy(y_true, y_pred):
    
    y_pred_sigmoid = torch.sigmoid(y_pred)
   
    positive_loss = -torch.sum(torch.log(y_pred_sigmoid[y_true == 1] + 1e-10))  
    negative_loss = -torch.sum(torch.log(1 - y_pred_sigmoid[y_true == 0] + 1e-10))  
    total_loss =  positive_loss + negative_loss
    return total_loss

def write_run(rerank_run, runf):
   
    with open(runf, 'wt') as runfile:
        for qid in rerank_run:
            scores = list(sorted(rerank_run[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))
            for i, (did, score) in enumerate(scores):
                runfile.write(f'{qid} 0 {did} {i+1} {score} run\n')

def normalize(tensor, eps=1e-9):
 
    return tensor / (torch.norm(tensor, dim=-1, keepdim=True) + eps)

def min_max_normalization(logits):
    
    min_val = torch.min(logits).item()
    max_val = torch.max(logits).item()
    normalized_logits = (logits - min_val) / (max_val - min_val)
    return normalized_logits





import math

def compute_dcg_at_k(ranked_list, qrels, k=100):
    ranked_list = list(ranked_list) 
    dcg = 0.0
    for i, doc_id in enumerate(ranked_list[:k]):
        relevance = qrels.get(doc_id, 0)
        if relevance > 0:
            dcg += (2 ** relevance - 1) / math.log2(i + 2)
    return dcg

def compute_idcg_at_k(qrels, k=100):
    # Ideal ranking of the top k relevant documents
    ideal_rels = sorted(qrels.values(), reverse=True)
    idcg = 0.0
    for i, relevance in enumerate(ideal_rels[:k]):
        if relevance > 0:
            idcg += (2 ** relevance - 1) / math.log2(i + 2)
    return idcg

def compute_ndcg_at_k(ranked_list, qrels, k=100):
    dcg = compute_dcg_at_k(ranked_list, qrels, k)
    idcg = compute_idcg_at_k(qrels, k)
    return dcg / idcg if idcg > 0 else 0.0

def compute_ndcg_at_100(rerank_run, valid_qrels):
    total_queries = len(rerank_run)
    ndcg_sum = 0.0
    for query_id, ranked_list in rerank_run.items():
        ndcg = compute_ndcg_at_k(ranked_list, valid_qrels.get(query_id, {}), k=100)
        ndcg_sum += ndcg

    ndcg_at_100 = ndcg_sum / total_queries
    return ndcg_at_100



def compute_precision_at_k(ranked_list, qrels, k=100):
    ranked_list = list(ranked_list) 
    relevant_count = 0
    for doc_id in ranked_list[:k]:
        if qrels.get(doc_id, 0.0) > 0:
            relevant_count += 1
    return relevant_count / k

def compute_precision_at_100(rerank_run, valid_qrels):
    total_queries = len(rerank_run)
    precision_sum = 0.0
    for query_id, ranked_list in rerank_run.items():
        precision = compute_precision_at_k(ranked_list, valid_qrels.get(query_id, {}), k=100)
        precision_sum += precision

    precision_at_100 = precision_sum / total_queries
    return precision_at_100

def compute_recall_at_100(rerank_run, valid_qrels):
    total_queries = len(rerank_run)
    recall_sum = 0.0

    for query_id, ranked_list in rerank_run.items():
        ranked_list = list(ranked_list)

        relevant_docs = set(valid_qrels.get(query_id, {}).keys())

        retrieved_docs = set(ranked_list[:100])

        retrieved_relevant_docs = retrieved_docs & relevant_docs

        if len(relevant_docs) > 0:
            recall = len(retrieved_relevant_docs) / len(relevant_docs)
        else:
            recall = 0.0

        recall_sum += recall

    recall_at_100 = recall_sum / total_queries
    return recall_at_100

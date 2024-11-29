"""
here you can evaluate the following dense retrival and reranking  pipeline, you will find in the sh file all the runing command ; 

splade+bert
splade+bert_small
spalde+rwkv_169M
splade+rwkv_430M
splade+T5_small
spalde+T5_base
spalde+T5_small trained with pointwise loss reranking
splade+T5_base  trained with pointwise loss reranking

"""
from pyserini.search.lucene import LuceneSearcher
from pyserini.search import SimpleSearcher
import splade
import os
import argparse
import subprocess
import random
from tqdm import tqdm
import torch
from statistics import mean
from collections import defaultdict
import sys
import modeling
import data
import torch.nn.functional as F
import modeling_util
from BM25 import BM25Okapi
import time
QUERY_NUM = 200
SIM_BATCH = 32
MODEL_BATCH = 128
NUM_DOC_FIRST_STAGE_OUT = 150
NUM_DOC_LAST_STAGE_OUT = 100

MODEL_MAP = {
    'vanilla_bert': modeling.VanillaBertRanker,
    'spalde_model': splade.Splade,
    'rwkv_model': modeling.Vanilla_rwkv,
    'T5_with_reranking_loss': modeling.T5_with_reranking_loss,
    'vanilla_t5': modeling.T5Ranker
}


def rerank(model, dataset, run, qrels,model_name):
    start_time = time.time()
    run_filtred = {query_id: top_docs for query_id, top_docs in run.items() if any(doc_id in qrels.get(query_id, {}) for doc_id in top_docs)}
    
    print( "the number of query they dont dont have relevant document after the first stage",len(run) - len(run_filtred))

    run_scores = run_model(model, dataset, run_filtred,model_name)
    
    rank = {qid: {docid: float(score) for docid, score in sorted(docs.items(), key=lambda x: x[1], reverse=True)[:NUM_DOC_LAST_STAGE_OUT]} for qid, docs in run_scores.items()}
    
    mrr_at_100 = modeling_util.compute_mrr_at_100(rank, qrels)
    ndcg100 = modeling_util.compute_ndcg_at_100(rank, qrels)
    precision100 = modeling_util.compute_precision_at_100(rank, qrels)
    recall_at_100 = modeling_util.compute_recall_at_100(rank, qrels)
    duration =time.time()-start_time
    
    print('Time taken for the dense model:', duration)
    print('MRR@100 ', mrr_at_100)
    print('NDCG@100:', ndcg100)
    print('Precision@100:', precision100)
    print('Recall@100:', recall_at_100)
    
    return mrr_at_100

def run_model(model, dataset, run, model_name,desc='reranking'):
    
    rerank_run = defaultdict(dict)
    with torch.no_grad(), tqdm(total=sum(len(r) for r in run.values()), ncols=80, desc=desc) as pbar:
        model.eval()
        
        for records in data.iter_valid_records(model, dataset, run,MODEL_BATCH):
           if model_name == 'T5_with_reranking_loss':
             true_id = model.tokenizer.get_vocab()[model.tokenizer.tokenize("true")[0]]
             false_id = model.tokenizer.get_vocab()[model.tokenizer.tokenize("false")[0]]
             logits = model.generate(records['query_tok'].to('cuda:0'),
                                    records['query_mask'].to('cuda:0'),
                                    records['doc_tok'].to('cuda:0'),
                                    records['doc_mask'].to('cuda:0'))  
             scores = logits[:, true_id].unsqueeze(dim=-1)
           elif model_name == 'vanilla_t5':
              true_id = model.tokenizer.get_vocab()[model.tokenizer.tokenize("true")[0]]
              false_id = model.tokenizer.get_vocab()[model.tokenizer.tokenize("false")[0]]
              logits = model.generate(records['query_tok'].to('cuda:0'),
                                    records['query_mask'].to('cuda:0'),
                                    records['doc_tok'].to('cuda:0'),
                                    records['doc_mask'].to('cuda:0'))  
              
              false_logits = logits[:, false_id].unsqueeze(dim=-1)
              true_logits = logits[:, true_id].unsqueeze(dim=-1)
              tf_logits = torch.cat((true_logits, false_logits), dim=-1)
              scores = tf_logits.log_softmax(dim=-1)[:, 0]
    
           elif model_name == 'rwkv_model' or model_name =='vanilla_bert' :
            scores = model(records['query_tok'].to('cuda:0'),
                                    records['query_mask'].to('cuda:0'),
                                    records['doc_tok'].to('cuda:0'),
                                    records['doc_mask'].to('cuda:0')) 
           else :
            print("no model name match")
            break
           for qid, did, score in zip(records['query_id'], records['doc_id'], scores):
                rerank_run[qid][did] = score.item()
           pbar.update(len(records['query_id']))
      
    return rerank_run



def main(bm25_searcher, model,dataset, querys, qrels, model_name):

  
    run = defaultdict(dict)
    print("Starting first-stage retrieval with BM25...")
    start_time = time.time()
    #querys = dict(random.sample(list(querys.items()),QUERY_NUM))
    with tqdm(total=len(querys), ncols=80, desc='BM25 retrieval', leave=False) as pbar:
        for query_id, query_text in querys.items():
            bm25_results = bm25_searcher.search(query_text, k=NUM_DOC_FIRST_STAGE_OUT)
            for doc in bm25_results:
                run[query_id][doc.docid] = doc.score  # BM25 scores
            pbar.update(1)
    rank = {qid: {docid: float(score) for docid, score in sorted(docs.items(), key=lambda x: x[1], reverse=True)[:NUM_DOC_FIRST_STAGE_OUT]} for qid, docs in run.items()}
    duration =time.time()-start_time
    mrr_at_100 = modeling_util.compute_mrr_at_100(rank, qrels)
    ndcg100 = modeling_util.compute_ndcg_at_100(rank, qrels)
    precision100 = modeling_util.compute_precision_at_100(rank, qrels)
    recall_at_100 = modeling_util.compute_recall_at_100(rank, qrels)
    
    
    print('Time taken in first stage', duration)
    print('MRR@100 :', mrr_at_100)
    print('NDCG@100:', ndcg100)
    print('Precision@100:', precision100)
    print('Recall@100:', recall_at_100)
    
    print("Starting reranking...")
    try:
        valid_score = rerank(model, dataset, rank, qrels, model_name)
    except ValueError as e:
        print(f"Error during reranking: {e}")
    
    return valid_score

'''import json

def save_dict_to_file(docs, file_path):
    with open(file_path, 'w') as f:
        for doc_id, text in docs.items():
            # Create a dictionary for each document
            doc_json = {
                'id': doc_id,
                'contents': text
            }
            # Write the dictionary as a JSON object to the file
            f.write(json.dumps(doc_json) + '\n')

    _,docs,_ = data.read_datafiles(args.datafiles)
    save_dict_to_file(docs, './data/corpus.json')
#save_dict_to_file(docs, 'corpus.json')'''
def main_cli():
    parser = argparse.ArgumentParser('reranking pipeline')
    parser.add_argument('--model', choices=MODEL_MAP.keys(), default='vanilla_bert')
    parser.add_argument('--datafiles', type=argparse.FileType('rt'), nargs='+')
    parser.add_argument('--qrels_valid', type=argparse.FileType('rt'))
    parser.add_argument('--query', type=argparse.FileType('rt'))
    parser.add_argument('--initial_model_weights', type=argparse.FileType('rb'))
    parser.add_argument('--index_dir', type=str, required=True, help='Path to the BM25 index directory')

    args = parser.parse_args()

    model_name = args.model
    
    qrels = data.read_qrels_dict(args.qrels_valid)
    
    query = data.read_query(args.query)
    
    dataset = data.read_datafiles(args.datafiles)
    
    bm25_searcher = LuceneSearcher(args.index_dir)

    model = MODEL_MAP[model_name]().cuda()
    model.load(args.initial_model_weights)

    main(bm25_searcher, model, dataset, query, qrels, model_name)


if __name__ == '__main__':
    main_cli()





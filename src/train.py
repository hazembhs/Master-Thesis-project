import os
import argparse
import subprocess
import random
import tempfile
from tqdm import tqdm
import torch
import modeling
import data
from statistics import mean
from collections import defaultdict
from torch.optim.lr_scheduler import ReduceLROnPlateau,OneCycleLR
import torch.nn.functional as F
import loss as loss_functions
import modeling_util


SEED = 42
min_LR = 0.001
max_LR = 0.004
max_rwkv_LR = 0.0005
min_rwkv_LR = 0.0001
FACTOR=0.1
MAX_EPOCH = 10
BATCH_VALID = 124
PCT_START = 0.1
BATCH_SIZE = 124
BATCHES_PER_EPOCH = 500
GRAD_ACC_SIZE = 20
PATIENCE = 4
LR_PATIENCE = 2
TOTAL_STEPS = MAX_EPOCH*BATCHES_PER_EPOCH

VALIDATION_METRIC = 'ndcg_cut_10'


torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)

MODEL_MAP = {
    'rwkv': modeling.Vanilla_rwkv
}


def main(model, dataset, train_pairs, qrels_train, valid_run, qrels_valid, model_out_dir):
   
    if isinstance(model, str):
        model = MODEL_MAP[model]().cuda()
    if model_out_dir is None:
        model_out_dir = tempfile.mkdtemp()
   
    params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    non_rwkv_params = {'params': [v for k, v in params if 'linear' in k ],'lr':min_LR}
    rwkv_params = {'params': [v for k, v in params if not 'linear' in k], 'lr': min_rwkv_LR}
    
    
    optimizer = torch.optim.Adam([non_rwkv_params, rwkv_params])
    
    one_cycle_scheduler = OneCycleLR(optimizer, max_lr=[max_LR, max_rwkv_LR], total_steps=TOTAL_STEPS, pct_start=PCT_START)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=FACTOR, patience=LR_PATIENCE, verbose=True)

    epoch = 0
    top_valid_score = None

    print(f'Starting training, upto {MAX_EPOCH} epochs, patience {PATIENCE} LR={max_LR} rwkv_LR={max_rwkv_LR}', flush=True)
    for epoch in range(MAX_EPOCH):

        loss = train_iteration(model, optimizer, dataset, train_pairs, qrels_train,one_cycle_scheduler)
        print(f'train epoch={epoch} loss={loss}', flush=True)

        valid_score = validate(model, dataset, valid_run, qrels_valid, epoch)
        print(f'validation epoch={epoch} score={valid_score}', flush=True)
        scheduler.step(valid_score)

        if top_valid_score is None or valid_score > top_valid_score:
            top_valid_score = valid_score
            print('new top validation score, saving weights', flush=True)
            model.save(os.path.join(model_out_dir, 'weights.p'))
            top_valid_score_epoch = epoch
        if top_valid_score is not None and epoch - top_valid_score_epoch > PATIENCE:
            print(f'no validation improvement since {top_valid_score_epoch}, early stopping', flush=True)
            break

    
    if top_valid_score_epoch != epoch:
        model.load(os.path.join(model_out_dir, 'weights.p'))
    return (model, top_valid_score_epoch)





def train_iteration(model, optimizer, dataset, train_pairs, qrels,one_cycle_scheduler):
    total = 0
   
    model.train()
    total_loss = 0.
    with tqdm('training', total=BATCH_SIZE * BATCHES_PER_EPOCH, ncols=80, desc='train') as pbar:
        for record in data.iter_train_pairs_with_labels_list_loss(model, dataset, train_pairs, qrels, GRAD_ACC_SIZE):
            
            scores = model(record['query_tok'],
                           record['query_mask'],
                           record['doc_tok'],
                           record['doc_mask'] )
            relevance = record['extra_labels'] 
            loss = loss_functions.pairwise_logistic_loss(relevance,scores)
            count = len(record['query_id'])           
            loss.backward()
            total_loss += loss.item()
            total += count
            if total % BATCH_SIZE == 0:
                for param_group in optimizer.param_groups:
                    print(f"Learning rate: {param_group['lr']}")
                optimizer.step()
                optimizer.zero_grad()
                one_cycle_scheduler.step()
            pbar.update(count)
            if total >= BATCH_SIZE * BATCHES_PER_EPOCH:
                return total_loss


def validate(model, dataset, run, valid_qrels, epoch):

    run_filtred = {query_id: top_docs for query_id, top_docs in run.items() if any(doc_id in valid_qrels.get(query_id, {}) for doc_id in top_docs)}
    run_scores = run_model(model, dataset, run_filtred)
    rank = {qid: {docid: float(score) for docid, score in sorted(docs.items(), key=lambda x: x[1], reverse=True)[:100]} for qid, docs in run_scores.items()}
    mrr_at_100 = modeling_util.compute_mrr_at_100(rank, valid_qrels)

    print("MRR@100:", mrr_at_100)
    
    return mrr_at_100

def run_model(model, dataset, run, desc='valid'):
    
    run_dev = {qid: run[qid] for qid in random.sample(list(run.keys()), int(len(run) * 0.1))}
    rerank_run = defaultdict(dict)
    with torch.no_grad(), tqdm(total=sum(len(r) for r in run_dev.values()), ncols=80, desc=desc) as pbar:
        model.eval()
        for records in data.iter_valid_records(model, dataset, run_dev, BATCH_VALID):
            logits = model(records['query_tok'],
                                    records['query_mask'],
                                    records['doc_tok'],
                                    records['doc_mask'])  
           
            for qid, did, score in zip(records['query_id'], records['doc_id'], logits):
                rerank_run[qid][did] = score.item()
            pbar.update(len(records['query_id']))
    
    return rerank_run




def main_cli():
    parser = argparse.ArgumentParser('rwkv model training and validation')
    parser.add_argument('--model', choices=MODEL_MAP.keys(), default='rwkv')
    parser.add_argument('--datafiles', type=argparse.FileType('rt'), nargs='+')
    parser.add_argument('--qrels', type=argparse.FileType('rt'))
    parser.add_argument('--qrels_valid', type=argparse.FileType('rt'))
    parser.add_argument('--train_pairs', type=argparse.FileType('rt'))
    parser.add_argument('--valid_run', type=argparse.FileType('rt'))
    parser.add_argument('--initial_rwkv_weights', type=argparse.FileType('rb'))
    parser.add_argument('--model_out_dir', type=str, required=True, help='Directory to save the trained model')
    args = parser.parse_args()
    model = MODEL_MAP[args.model]().cuda()
   
    if args.initial_rwkv_weights is not None:
        model.load(args.initial_rwkv_weights.name)
    os.makedirs(args.model_out_dir, exist_ok=True)
   
    dataset = data.read_datafiles(args.datafiles)
    qrels = data.read_qrels_dict(args.qrels)
    qrels_valid = data.read_qrels_dict(args.qrels_valid)
    train_pairs = data.read_pairs_dict(args.train_pairs)
    valid_run = data.read_run_dict(args.valid_run)

    main(model, dataset, train_pairs, qrels, valid_run, qrels_valid, args.model_out_dir)


if __name__ == '__main__':
    main_cli()


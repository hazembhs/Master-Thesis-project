import random
from tqdm import tqdm
import torch
import ast
import itertools
import json
def read_query(file):
    queries = {}
    for line in tqdm(file, desc=f'Loading querys (by line)', leave=False):
        cols = line.rstrip().split('\t')
        if len(cols) < 2:
            tqdm.write(f'Skipping line: `{line.rstrip()}`')
            continue
        c_id, c_text = cols
        queries[c_id] = c_text
    return queries

def read_datafiles(files):
    queries_train = {}
    docs = {}
    queries_dev = {}

    for file_obj in files:
        for line in tqdm(file_obj, desc=f'Loading file (by line)', leave=False):
            cols = line.rstrip().split('\t')
            if len(cols) < 2:
                tqdm.write(f'Skipping line: `{line.rstrip()}`')
                continue
            if file_obj.name.endswith('train.tsv'):
                c_id, c_text = cols
                queries_train[c_id] = c_text
            elif file_obj.name.endswith('test-queries.tsv'):
                c_id, c_text = cols
                queries_dev[c_id] = c_text
            elif file_obj.name.endswith('docs.tsv'):
                if len(cols) == 4:
                    c_id, url, title, body = cols
                    docs[c_id] = f"{url} {title} {body}"
                elif  len(cols) == 3:
                    c_id, url , title= cols
                    docs[c_id] = f"{url} {title}"
                else:
                     c_id, c_text = cols
                     docs[c_id] = c_text
              
    return queries_train, docs, queries_dev

def read_qrels_dict(file):
    result = {}
    for line in tqdm(file, desc='loading qrels (by line)', leave=False):
        qid, _, docid, score = line.split()
        result.setdefault(qid, {})[docid] = int(score)
    return result


def read_run_dict(file):
    result = {}
    for line in tqdm(file, desc='loading run (by line)', leave=False):
        qid, _, docid, rank, score, _ = line.split()
        result.setdefault(qid, {})[docid] = float(score)
    return result


def read_pairs_dict(file):
    result = {}
    for line in tqdm(file, desc='loading pairs (by line)', leave=False):
        qid, _, docid, rank, score, _ = line.split()
        result.setdefault(qid, {})[docid] = 1
    return result


def iter_train_pairs(model, dataset, train_pairs, qrels, batch_size):
    batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': []}
    for qid, did, query_tok, doc_tok in _iter_train_pairs(model, dataset, train_pairs, qrels):
        batch['query_id'].append(qid)
        batch['doc_id'].append(did)
        batch['query_tok'].append(query_tok)
        batch['doc_tok'].append(doc_tok)
        if len(batch['query_id']) // 2 == batch_size:
            yield _pack_n_ship(batch)
            batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': []}



def _iter_train_pairs(model, dataset, train_pairs, qrels):
    ds_queries, ds_docs, _ = dataset
    while True:
        qids = list(qrels.keys())
        random.shuffle(qids)
        for qid in qids:
            pos_id  = list(qrels[qid].keys())[0]
            neg_ids = [did for did in train_pairs[qid] if did not in qrels.get(qid, {})]
            neg_id = random.choice(neg_ids)
            query_tok = model.tokenize(ds_queries[qid])
            pos_doc = ds_docs.get(pos_id)
            neg_doc = ds_docs.get(neg_id)
            if pos_doc is None:
                tqdm.write(f'missing doc {pos_id}! Skipping')
                continue
            if neg_doc is None:
                tqdm.write(f'missing doc {neg_id}! Skipping')
                continue
            yield qid, pos_id, query_tok, model.tokenize(pos_doc)
            yield qid, neg_id, query_tok, model.tokenize(neg_doc)


def iter_valid_records(model, dataset, run, batch_size):
    batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': []}
    for qid, did, query_tok, doc_tok in _iter_valid_records(model, dataset, run):
        batch['query_id'].append(qid)
        batch['doc_id'].append(did)
        batch['query_tok'].append(query_tok)
        batch['doc_tok'].append(doc_tok)
        if len(batch['query_id']) == batch_size:
            yield _pack_n_ship(batch)
            batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': []}
    if len(batch['query_id']) > 0:
        yield _pack_n_ship(batch)


def _iter_valid_records(model, dataset, run):
    _, ds_docs, dev_queries, = dataset
    for qid in run:
        query_tok = model.tokenize(dev_queries[qid])
        for did in run[qid]:
            doc = ds_docs.get(did)
            if doc is None:
                tqdm.write(f'missing doc {did}! Skipping')
                continue
            doc_tok = model.tokenize(doc)
            yield qid, did, query_tok, doc_tok


def _pack_n_ship(batch):
    QLEN = 35
    MAX_DLEN = 980
    DLEN = min(MAX_DLEN, max(len(b) for b in batch['doc_tok']))
    return {
        'query_id': batch['query_id'],
        'doc_id': batch['doc_id'],
        'query_tok': _pad_crop(batch['query_tok'], QLEN),
        'doc_tok': _pad_crop(batch['doc_tok'], DLEN),
        'query_mask': _mask(batch['query_tok'], QLEN),
        'doc_mask': _mask(batch['doc_tok'], DLEN),
    }


def _pad_crop(items, l):
    result = []
    for item in items:
        if len(item) < l:
            item = item + [-1] * (l - len(item))
        if len(item) > l:
            item = item[:l]
        result.append(item)
    return torch.tensor(result)


def _mask(items, l):
    result = []
    for item in items:
        if len(item) < l:
            mask = [1. for _ in item] + ([0.] * (l - len(item)))
        if len(item) >= l:
            mask = [1. for _ in item[:l]]
        result.append(mask)
    return torch.tensor(result)

def iter_train_pairs_with_labels(model, dataset, train_pairs, qrels, batch_size):
    batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': [], 'labels': [], 'extra_labels': []}
    for qid, did, query_tok, doc_tok, labels,  extra_labels in _iter_train_pairs_with_labels(model, dataset, train_pairs, qrels):
        batch['query_id'].append(qid)
        batch['doc_id'].append(did)
        batch['query_tok'].append(query_tok)
        batch['doc_tok'].append(doc_tok)
        batch['labels'].append(labels)
        batch['extra_labels'].append(extra_labels)
        if len(batch['query_id']) // 2 == batch_size:
            yield _pack_n_ship_with_labels(batch)
            batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': [], 'labels': [], 'extra_labels': []}


def _iter_train_pairs_with_labels(model, dataset, train_pairs, qrels):
    ds_queries, ds_docs, _ = dataset
    while True:
        qids = list(qrels.keys())
        random.shuffle(qids)
        for qid in qids:
            pos_id  = list(qrels[qid].keys())[0]
            neg_ids = [did for did in train_pairs.get(qid, []) if did not in qrels.get(qid, {})]
            neg_id = random.choice(neg_ids)
            query_tok = model.tokenize(ds_queries[qid])
            pos_doc = ds_docs.get(pos_id)
            neg_doc = ds_docs.get(neg_id)
            if pos_doc is None:
                tqdm.write(f'missing doc {pos_id}! Skipping')
                continue
            if neg_doc is None:
                tqdm.write(f'missing doc {neg_id}! Skipping')
                continue
            yield qid, pos_id, query_tok, model.tokenize(pos_doc), model.tokenize("true"), 1
            yield qid, neg_id, query_tok, model.tokenize(neg_doc), model.tokenize("false"), 0

def iter_train_pairs_with_labels_list_loss(model, dataset, train_pairs, qrels, batch_size):
    batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': [], 'labels': [], 'extra_labels': []}
    for qid, did, query_tok, doc_tok, labels,  extra_labels in _iter_train_pairs_with_labels_list_loss(model, dataset, train_pairs, qrels):
        batch['query_id'].append(qid)
        batch['doc_id'].append(did)
        batch['query_tok'].append(query_tok)
        batch['doc_tok'].append(doc_tok)
        batch['labels'].append(labels)
        batch['extra_labels'].append(extra_labels)
        if len(batch['query_id']) == batch_size:
            yield _pack_n_ship_with_labels(batch)
            batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': [], 'labels': [], 'extra_labels': []}

def _iter_train_pairs_with_labels_list_loss(model, dataset, train_pairs, qrels, num_negatives=19):
    ds_queries, ds_docs, _ = dataset
    while True:
        qids = list(qrels.keys())
        random.shuffle(qids)
        for qid in qids:
            pos_id = list(qrels[qid].keys())[0]
            pos_doc = ds_docs.get(pos_id)
            
            if pos_doc is None:
                tqdm.write(f'missing doc {pos_id}! Skipping')
                continue
            
            query_tok = model.tokenize(ds_queries[qid])
            yield qid, pos_id, query_tok, model.tokenize(pos_doc), model.tokenize("true"), 1
            
            neg_ids = [did for did in train_pairs.get(qid, []) if did not in qrels.get(qid, {})]
            neg_ids = random.sample(neg_ids, min(num_negatives, len(neg_ids)))
            
            for neg_id in neg_ids:
                neg_doc = ds_docs.get(neg_id)
                
                if neg_doc is None:
                    tqdm.write(f'missing doc {neg_id}! Skipping')
                    continue
                
                yield qid, neg_id, query_tok, model.tokenize(neg_doc), model.tokenize("false"), 0


def _pack_n_ship_with_labels(batch):
    QLEN = 35
    MAX_DLEN = 980
    DLEN = min(MAX_DLEN, max(len(b) for b in batch['doc_tok']))
    return {
        'query_id': batch['query_id'],
        'doc_id': batch['doc_id'],
        'query_tok': _pad_crop(batch['query_tok'], QLEN),
        'doc_tok': _pad_crop(batch['doc_tok'], DLEN),
        'query_mask': _mask(batch['query_tok'], QLEN),
        'doc_mask': _mask(batch['doc_tok'], DLEN),
        'labels': _pad_crop(batch['labels'], 1),
        'extra_labels': torch.tensor(batch['extra_labels']).long().cuda()
    }


def read_doc(files):
    docs = {}
    for line in tqdm(files, desc='Loading file (by line)', leave=False):
            cols = line.rstrip().split('\t')
            if len(cols) < 2:
                tqdm.write(f'Skipping line: `{line.rstrip()}`')
                continue
            else:
                if len(cols) == 4:
                    c_id, url, title, body = cols
                    docs[c_id] = f"{url} {title} {body}"
                elif  len(cols) == 3:
                    c_id, url , title= cols
                    docs[c_id] = f"{url} {title}"
                else:
                     c_id, c_text = cols
                     docs[c_id] = c_text   
    return docs


def get_doc(model, dataset, docs ,batch_size):
    batch = {'doc_id': [], 'doc_tok': []}
    for did,  doc_tok in _get_doc(model, dataset,docs):
        batch['doc_id'].append(did)
        batch['doc_tok'].append(doc_tok)
        if len(batch['doc_id']) == batch_size:
            yield _pack_n_doc(batch)
            batch = {'doc_id': [], 'doc_tok': []}

def _get_doc(model, dataset,docs):
    doc= dataset
    for id in docs:
       doc_tok = model.transformer_rep.tokenizer(dataset.get(id))['input_ids']
       yield  id, doc_tok

def _pack_n_doc(batch):
    MAX_DLEN =40
    DLEN = min(MAX_DLEN, max(len(b) for b in batch['doc_tok']))
    return {
        'doc_id': batch['doc_id'],
        'doc_tok': pad(batch['doc_tok'], DLEN),
        'doc_mask': mask(batch['doc_tok'], DLEN),
    }

def pad(items, l):
    result = []
    for item in items:
        if len(item) < l:
            item = item + [-1] * (l - len(item))
        if len(item) > l:
            item = item[:l]
        result.append(item)
    return torch.tensor(result)


def mask(items, l):
    result = []
    for item in items:
        if len(item) < l:
            mask = [1. for _ in item] + ([0.] * (l - len(item)))
        if len(item) >= l:
            mask = [1. for _ in item[:l]]
        result.append(mask)
    return torch.tensor(result)

def read_dev_doc(file):
    result = []
    for line in tqdm(file, desc='loading qrels (by line)', leave=False):
        qid, _, docid, score = line.split()
        result.append(docid)
    
    return result

def read_doc_vec(file_path):
    embeddings = {}
    with open(file_path, 'r') as f:
        # Skip the header line
        next(f)
        for line in f:
            parts = line.rstrip().split('\t')
            if len(parts) != 3:
                print(f"Skipping malformed line: {line.strip()}")
                continue
            
            did = parts[0].strip()
            try:
                vector1 = torch.tensor([float(val) for val in parts[1].strip('[]\n').split(', ')])
                vector2 = torch.tensor([float(val) for val in parts[2].strip('[]\n').split(', ')])
                embeddings[did] = (vector1, vector2)
            except ValueError as e:
                print(f"Error parsing line: {line.strip()} - {e}")
                continue
    
    return embeddings




def get_query(model,query,SPLADE_BATCH,QUERY_NUM):
    batch = {'query_id': [], 'query_tok': []}
    for id , text in itertools.islice(query.items(), QUERY_NUM):
        batch['query_id'].append(id)
        batch['query_tok'].append(model.transformer_rep.tokenizer(text)['input_ids'])
        if len(batch['query_id']) == SPLADE_BATCH:
            yield _pack_n_query(batch)
            batch = {'query_id': [], 'query_tok': []}

def get_query_sentence_bert(query,BERT_BATCH,QUERY_NUM):
    batch = {'query_id': [], 'query_tok': []}
    for id , text in itertools.islice(query.items(), QUERY_NUM):
        batch['query_id'].append(id)
        batch['query_tok'].append(text)
        if len(batch['query_id']) == BERT_BATCH:
            yield batch
            batch = {'query_id': [], 'query_tok': []}


def _pack_n_query(batch):
    MAX_DLEN =40
    DLEN = min(MAX_DLEN, max(len(b) for b in batch['query_tok']))
    return {
        'query_id': batch['query_id'],
        'query_tok': pad(batch['query_tok'], DLEN),
        'query_mask': mask(batch['query_tok'], DLEN),
    }

def read_json_embedding(json_file_path):
    doc_embeddings = {}
    with open(json_file_path, 'r') as f:
       doc_embeddings = json.load(f)
    return doc_embeddings
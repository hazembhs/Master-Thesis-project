
# reranking-and-dense-retrieval-system

# Deep Learning Models for Document Retrieval and Reranking

## Overview
This project implements various deep learning models for document retrieval and reranking. The goal is to evaluate their effectiveness in improving retrieval performance based on multiple metrics, such as Mean Reciprocal Rank (MRR), Normalized Discounted Cumulative Gain (NDCG), and Precision/Recall.

## Models Implemented
- **BM25** : Combination of traditional BM25 scoring with deep learning models to enhance relevance ranking.
- **RWKV**:  is an RNN with Transformer-level LLM performance, which can also be directly trained like a GPT transformer (parallelizable). And it's 100% attention-free

## Metrics
- **Mean Reciprocal Rank (MRR@100)**: Measures the average rank of the first relevant document.
- **Normalized Discounted Cumulative Gain (NDCG@100)**: Evaluates ranking quality based on the position of relevant documents.
- **Precision and Recall**: Assess the accuracy of the retrieved documents.

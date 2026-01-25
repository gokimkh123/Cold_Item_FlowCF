# evaluate.py
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
import time
import yaml
from tqdm import tqdm

sys.path.append(os.getcwd())

from recbole.config import Config
from recbole.data import create_dataset
from recbole.utils import init_seed
from model.flowcf import FlowCF
try:
    from model.diffcf import DiffCF
except ImportError:
    DiffCF = None

def compute_metrics(top_k_items, ground_truth_items, k):
    """
    User-to-Item Metrics 계산 (Recall, NDCG, Precision, Hit)
    """
    hits = 0
    sum_r = 0.0
    gt_set = ground_truth_items
    
    for i, item_id in enumerate(top_k_items):
        if item_id in gt_set:
            hits += 1
            sum_r += 1.0 / np.log2(i + 2)

    # 1. Recall: 맞춘 개수 / 전체 정답 개수
    recall = hits / len(gt_set) if len(gt_set) > 0 else 0.0
    
    # 2. NDCG
    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(gt_set), k))])
    ndcg = sum_r / idcg if idcg > 0 else 0.0
    
    # 3. Precision: 맞춘 개수 / K
    precision = hits / k
    
    # 4. Hit Rate: 하나라도 맞췄으면 1, 아니면 0
    hit = 1.0 if hits > 0 else 0.0
    
    return recall, ndcg, precision, hit

def print_pretty_result(model_name, n_steps, s_steps, valid_user_count, results, inference_time):
    """지표 4개(Recall, NDCG, Precision, Hit)를 포함한 결과 출력"""
    
    s_str = str(s_steps) if s_steps is not None else "-"
    
    print("\n" + "="*50)
    if model_name == 'FlowCF':
        print(f"n_steps: {n_steps} | s_steps: {s_str}")
    else:
        print(f"n_steps: {n_steps} (Fixed)")
        
    print("="*50)
    print("Final User-to-Item Cold-Start Evaluation Results")
    print(f"Model: {model_name}")
    print(f"Evaluated Users: {valid_user_count}")
    print("="*50)
    
    print(f"{'Metric':<10} {'@10':<10} {'@20':<10}")
    print("-" * 30)
    
    # Recall
    r10 = results.get('recall@10', 0.0)
    r20 = results.get('recall@20', 0.0)
    print(f"{'Recall':<10} {r10:.4f}     {r20:.4f}")
    
    # NDCG
    n10 = results.get('ndcg@10', 0.0)
    n20 = results.get('ndcg@20', 0.0)
    print(f"{'NDCG':<10} {n10:.4f}     {n20:.4f}")
    
    # Precision [추가]
    p10 = results.get('precision@10', 0.0)
    p20 = results.get('precision@20', 0.0)
    print(f"{'Precision':<10} {p10:.4f}     {p20:.4f}")
    
    # Hit Rate [추가]
    h10 = results.get('hit@10', 0.0)
    h20 = results.get('hit@20', 0.0)
    print(f"{'Hit':<10} {h10:.4f}     {h20:.4f}")
    
    print("="*50)
    print(f"Inference Time: {inference_time:.2f}s")
    print("="*50 + "\n")

def evaluate_cold_start(model, dataset, test_file_path, k_list=[10, 20]):
    try:
        df_test = pd.read_csv(test_file_path, sep='\t')
        df_test.columns = [col.split(':')[0] for col in df_test.columns]
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    user_ground_truth = df_test.groupby('uid')['mid'].apply(set).to_dict()
    test_mids = df_test['mid'].unique()
    
    movie_field = dataset.uid_field 
    user_field = dataset.iid_field   

    model.eval()
    
    start_time = time.time()
    
    score_matrix_rows = []
    valid_movie_indices = []

    with torch.no_grad():
        for mid in tqdm(test_mids, desc="Predicting Movies"):
            try:
                internal_mid = dataset.token2id(movie_field, str(mid))
            except (ValueError, KeyError):
                continue
            
            scores = model.predict_cold_item(int(internal_mid))
            score_matrix_rows.append(scores.cpu().numpy().flatten())
            valid_movie_indices.append(mid)

    end_time = time.time()
    inference_time = end_time - start_time 

    if not score_matrix_rows:
        return None

    score_matrix = np.vstack(score_matrix_rows)

    # 지표 저장소 초기화
    metrics_name = ['recall', 'ndcg', 'precision', 'hit']
    results = {k: {m: 0.0 for m in metrics_name} for k in k_list}
    
    valid_user_count = 0
    
    for uid_raw, true_mids_set in user_ground_truth.items():
        try:
            internal_uid = dataset.token2id(user_field, str(uid_raw))
        except (ValueError, KeyError):
            continue
            
        if internal_uid >= score_matrix.shape[1]:
            continue 
            
        user_scores = score_matrix[:, internal_uid]
        max_k = max(k_list)
        top_indices = np.argsort(user_scores)[-max_k:][::-1]
        recommended_mids = [valid_movie_indices[idx] for idx in top_indices]
        
        for k in k_list:
            rec, ndcg, prec, hit = compute_metrics(recommended_mids[:k], true_mids_set, k)
            results[k]['recall'] += rec
            results[k]['ndcg'] += ndcg
            results[k]['precision'] += prec
            results[k]['hit'] += hit
            
        valid_user_count += 1

    final_results = {}
    if valid_user_count > 0:
        for k in k_list:
            for m in metrics_name:
                final_results[f'{m}@{k}'] = results[k][m] / valid_user_count
        
        # 출력
        model_name = model.__class__.__name__
        n_steps = getattr(model, 'n_steps', '?')
        s_steps = getattr(model, 's_steps', None)
        
        print_pretty_result(model_name, n_steps, s_steps, valid_user_count, final_results, inference_time)
        
        # 파싱용 출력 (Recall@20 기준으로 그래프를 그리므로 이것들은 유지)
        print(f"__PARSE_RESULT__:{final_results['recall@10']},{final_results['recall@20']},{inference_time}")
        
    return final_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='flowcf.yaml')
    parser.add_argument('--test_file', type=str, default='dataset/ML1M/BPR_cv/BPR_cv.test.inter')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--steps', type=int, default=None, help='Override inference steps (n_steps)')
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint)
    saved_config = checkpoint['config']

    if 'model' in saved_config:
        model_name = saved_config['model']
    else:
        model_name = 'FlowCF'
    print(f"Detected Model Type: {model_name}")

    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            current_yaml = yaml.safe_load(f)

        if model_name == 'DiffCF':
            print(f">>> [DiffCF Protected] Ignoring n_steps from {args.config} to preserve training settings.")
        else:
            if 'n_steps' in current_yaml:
                print(f">>> Updating n_steps from YAML: {saved_config['n_steps']} -> {current_yaml['n_steps']}")
                saved_config['n_steps'] = current_yaml['n_steps']
                
            if 's_steps' in current_yaml:
                old_s = saved_config['s_steps'] if 's_steps' in saved_config else 'None'
                print(f">>> Updating s_steps from YAML: {old_s} -> {current_yaml['s_steps']}")
                saved_config['s_steps'] = current_yaml['s_steps']

    if args.steps is not None:
        print(f">>> [Force Override] n_steps from Command Line: {saved_config['n_steps']} -> {args.steps}")
        saved_config['n_steps'] = args.steps

    if 'eval_batch_size' in saved_config and saved_config['eval_batch_size'] < 4096:
        saved_config['eval_batch_size'] = 4096

    init_seed(saved_config['seed'], saved_config['reproducibility'])
    dataset = create_dataset(saved_config)
    
    if model_name == 'FlowCF':
        model = FlowCF(saved_config, dataset).to(saved_config['device'])
    elif model_name == 'DiffCF' and DiffCF is not None:
        model = DiffCF(saved_config, dataset).to(saved_config['device'])
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.load_state_dict(checkpoint['state_dict'])
    
    evaluate_cold_start(model, dataset, args.test_file, k_list=[10, 20])
# src/metrics.py
import numpy as np

def compute_metrics(top_k_items, ground_truth_items, k_list=[10, 20]):
    """
    Recall, NDCG, Precision, Hit Rate 계산
    """
    # Set 변환 (속도 최적화)
    if not isinstance(ground_truth_items, set):
        gt_set = set(ground_truth_items)
    else:
        gt_set = ground_truth_items

    # 정답이 없으면 0 반환
    if not gt_set:
        return {
            f"Recall@{k}": 0.0 for k in k_list
        } | {
            f"NDCG@{k}": 0.0 for k in k_list
        } | {
            f"Precision@{k}": 0.0 for k in k_list
        } | {
            f"Hit@{k}": 0.0 for k in k_list
        }

    # Hit 여부 리스트 (1 or 0)
    # top_k_items가 내부 ID(int)든 문자열(str)이든, gt_set과 타입만 맞으면 됨
    hit_list = []
    for item_id in top_k_items:
        if item_id in gt_set:
            hit_list.append(1)
        else:
            hit_list.append(0)

    results = {}
    for k in k_list:
        # k개까지만 잘라서 계산
        k_hit_list = hit_list[:k]
        k_hits = sum(k_hit_list)
        
        # 1. Recall
        results[f"Recall@{k}"] = k_hits / len(gt_set)
        
        # 2. NDCG
        k_sum_r = sum([1.0 / np.log2(i + 2) for i, is_hit in enumerate(k_hit_list) if is_hit])
        idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(gt_set), k))])
        results[f"NDCG@{k}"] = k_sum_r / idcg if idcg > 0 else 0.0
        
        # 3. Precision (맞춘 개수 / 추천 개수 K)
        results[f"Precision@{k}"] = k_hits / k
        
        # 4. Hit Rate (하나라도 맞췄으면 1)
        results[f"Hit@{k}"] = 1.0 if k_hits > 0 else 0.0

    return results
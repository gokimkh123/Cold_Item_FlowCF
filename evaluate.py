import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm  # 진행상황 바 표시

# 현재 경로 추가
sys.path.append(os.getcwd())

from recbole.config import Config
from recbole.data import create_dataset
from recbole.utils import init_seed
from model.flowcf import FlowCF
try:
    from model.diffcf import DiffCF
except ImportError:
    DiffCF = None

def compute_recall_ndcg(top_k_items, ground_truth_items, k):
    """
    User-to-Item Metrics 계산
    :param top_k_items: 모델이 추천한 아이템 리스트 (Top-K)
    :param ground_truth_items: 유저가 실제로 본 아이템 리스트 (Set)
    :param k: @K
    """
    hits = 0
    sum_r = 0.0
    
    # 셋(Set)으로 변환되어 있다고 가정
    gt_set = ground_truth_items
    
    for i, item_id in enumerate(top_k_items):
        if item_id in gt_set:
            hits += 1
            sum_r += 1.0 / np.log2(i + 2)

    # Recall: 맞춘 개수 / 전체 정답 개수 (Standard Definition)
    recall = hits / len(gt_set) if len(gt_set) > 0 else 0.0
    
    # NDCG
    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(gt_set), k))])
    ndcg = sum_r / idcg if idcg > 0 else 0.0
    
    return recall, ndcg

def evaluate_cold_start(model, dataset, test_file_path, k_list=[10, 20]):
    """
    [핵심 수정] User-to-Item 평가 방식 (Standard Protocol)
    신규 아이템(Cold Item) N개가 업데이트되었을 때,
    각 유저에게 이 신규 아이템들을 추천해주고, 실제 봤는지 평가함.
    """
    print(f"\n[평가 시작] User-to-Item Cold-Start Evaluation")
    print(f"Target File: {test_file_path}")
    
    # 1. 데이터 로드 및 전처리
    try:
        df_test = pd.read_csv(test_file_path, sep='\t')
        df_test.columns = [col.split(':')[0] for col in df_test.columns] # 컬럼명 정리 (uid:token -> uid)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # 유저별 정답셋 생성: {uid: {mid1, mid2, ...}}
    # 여기서 uid는 '사람', mid는 '영화'입니다.
    user_ground_truth = df_test.groupby('uid')['mid'].apply(set).to_dict()
    
    # 평가 대상인 모든 신규 영화(Cold Items) 리스트 추출
    test_mids = df_test['mid'].unique()
    
    print(f"총 평가 대상 유저 수: {len(user_ground_truth)}")
    print(f"총 신규 영화(Cold Items) 수: {len(test_mids)}")

    # RecBole 필드명 매핑 (yaml 설정에 따름)
    # USER_ID_FIELD: mid (모델 입력), ITEM_ID_FIELD: uid (모델 출력)
    movie_field = dataset.uid_field 
    user_field = dataset.iid_field   

    model.eval()
    
    # =========================================================
    # 2. Score Matrix 생성 (Matrix Construction)
    # 목표: [Number_of_Movies, Number_of_Users] 크기의 점수판 만들기
    # =========================================================
    print("모든 신규 영화에 대해 점수 계산 중 (Matrix Construction)...")
    
    score_matrix_rows = []
    valid_movie_indices = [] # 실제로 예측에 성공한 영화의 원본 ID 리스트

    with torch.no_grad():
        for mid in tqdm(test_mids, desc="Predicting Movies"):
            try:
                # 영화 ID(토큰)를 내부 ID로 변환
                internal_mid = dataset.token2id(movie_field, str(mid))
            except (ValueError, KeyError):
                # 학습/테스트 범위 밖의 ID 등 예외 처리
                continue
            
            # 모델 예측: 이 영화(mid)에 대한 전체 유저의 점수 벡터 얻기
            # FlowCF의 predict_cold_item은 [1, n_users] 혹은 [n_users]를 반환
            scores = model.predict_cold_item(int(internal_mid))
            
            # CPU로 내려서 리스트에 저장 (메모리 관리)
            score_matrix_rows.append(scores.cpu().numpy().flatten())
            valid_movie_indices.append(mid)

    if not score_matrix_rows:
        print("평가 가능한 영화가 없습니다.")
        return

    # [N_movies, M_users] 크기의 행렬 완성
    # Row: 영화, Column: 유저
    score_matrix = np.vstack(score_matrix_rows)
    print(f"점수 행렬 생성 완료. Shape: {score_matrix.shape}")

    # =========================================================
    # 3. 유저별 랭킹 평가 (User-wise Evaluation)
    # =========================================================
    results = {k: {'recall': 0.0, 'ndcg': 0.0} for k in k_list}
    valid_user_count = 0
    
    print("유저별 랭킹 평가 진행 중 (User-wise Ranking)...")
    
    for i, (uid_raw, true_mids_set) in enumerate(tqdm(user_ground_truth.items(), desc="Evaluating Users")):
        try:
            # 유저 ID(토큰)를 내부 ID로 변환
            internal_uid = dataset.token2id(user_field, str(uid_raw))
        except (ValueError, KeyError):
            continue
            
        # [핵심 로직] 행렬에서 해당 유저의 컬럼(열)만 쏙 뽑아옵니다.
        # 이 벡터는 "이 유저가 각 신규 영화들에 대해 매긴 점수 리스트"입니다.
        if internal_uid >= score_matrix.shape[1]:
            continue 
            
        user_scores = score_matrix[:, internal_uid]
        
        # 랭킹 매기기: 점수가 높은 순서대로 영화의 '인덱스'를 뽑습니다.
        # np.argsort는 오름차순이므로 [::-1]로 뒤집어서 내림차순 정렬
        max_k = max(k_list)
        top_indices = np.argsort(user_scores)[-max_k:][::-1]
        
        # 인덱스를 실제 영화 ID(mid)로 변환
        recommended_mids = [valid_movie_indices[idx] for idx in top_indices]
        
        # 지표 계산
        for k in k_list:
            current_top_k = recommended_mids[:k]
            rec, ndcg = compute_recall_ndcg(current_top_k, true_mids_set, k)
            results[k]['recall'] += rec
            results[k]['ndcg'] += ndcg
            
        valid_user_count += 1

    # =========================================================
    # 4. 결과 출력
    # =========================================================
    if valid_user_count > 0:
        print("\n" + "="*50)
        print(f"Final User-to-Item Cold-Start Evaluation Results")
        print(f"Model: {model.__class__.__name__}")
        print(f"Evaluated Users: {valid_user_count}")
        print("="*50)
        print(f"{'Metric':<10} {'@10':<10} {'@20':<10}")
        print("-" * 30)
        
        rec_10 = results[10]['recall'] / valid_user_count
        rec_20 = results[20]['recall'] / valid_user_count
        ndcg_10 = results[10]['ndcg'] / valid_user_count
        ndcg_20 = results[20]['ndcg'] / valid_user_count
        
        print(f"{'Recall':<10} {rec_10:.4f}     {rec_20:.4f}")
        print(f"{'NDCG':<10} {ndcg_10:.4f}     {ndcg_20:.4f}")
        print("="*50 + "\n")
    else:
        print("평가 가능한 유저가 없습니다.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='flowcf.yaml')
    parser.add_argument('--test_file', type=str, default='dataset/ML1M/BPR_cv/BPR_cv.test.inter', help='평가할 데이터셋 경로 (test 또는 vali)')
    parser.add_argument('--checkpoint', type=str, required=True, help='학습된 모델 체크포인트 경로')
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.checkpoint}")
    try:
        checkpoint = torch.load(args.checkpoint)
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        exit()
    
    # 저장된 Config 로드
    saved_config = checkpoint['config']
    
    # 메모리 이슈 방지를 위해 배치 사이즈 조절 (필요 시)
    if 'eval_batch_size' in saved_config and saved_config['eval_batch_size'] < 4096:
        saved_config['eval_batch_size'] = 4096

    init_seed(saved_config['seed'], saved_config['reproducibility'])
    dataset = create_dataset(saved_config)
    
    # 모델 타입 확인 및 초기화
    try:
        model_name = saved_config['model']
    except KeyError:
        model_name = 'FlowCF'
    
    print(f"Detected Model Type: {model_name}")

    if model_name == 'FlowCF':
        model = FlowCF(saved_config, dataset).to(saved_config['device'])
    elif model_name == 'DiffCF' and DiffCF is not None:
        model = DiffCF(saved_config, dataset).to(saved_config['device'])
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.load_state_dict(checkpoint['state_dict'])
    
    # 평가 실행
    evaluate_cold_start(model, dataset, args.test_file, k_list=[10, 20])
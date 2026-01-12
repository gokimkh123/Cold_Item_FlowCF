import argparse
import torch
import numpy as np
import os
import sys
import random

# 경로 설정
sys.path.append(os.getcwd())

from recbole.config import Config
from recbole.data import create_dataset
from recbole.utils import init_seed
from recbole.data.interaction import Interaction
from model.flowcf import FlowCF

# ============================================================================
# [평가 지표 함수] Recall & NDCG 계산
# ============================================================================
def compute_metrics(top_indices, ground_truth_tokens, k=10):
    """
    top_indices: 모델이 추천한 유저 ID 리스트 (상위 K개)
    ground_truth_tokens: 실제로 좋아한 유저 ID 리스트 (정답)
    """
    # 1. Hit (맞췄는가?)
    hits = 0
    sum_r = 0.0
    
    # Ground Truth를 set으로 변환 (검색 속도 향상)
    gt_set = set(ground_truth_tokens)
    
    for i, idx in enumerate(top_indices):
        if idx in gt_set:
            hits += 1
            sum_r += 1.0 / np.log2(i + 2) # NDCG 분자

    # 2. Recall@K
    recall = hits / len(ground_truth_tokens) if len(ground_truth_tokens) > 0 else 0.0

    # 3. NDCG@K
    dcg = sum_r
    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(ground_truth_tokens), k))])
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return recall, ndcg

# ============================================================================
# 메인 평가 로직
# ============================================================================
if __name__ == '__main__':
    # 1. 설정 및 모델 로드
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pth file')
    parser.add_argument('--test_items', type=int, default=100, help='Number of items to test')
    args, _ = parser.parse_known_args()

    config = Config(model=FlowCF, config_file_list=['flowcf.yaml'])
    init_seed(config['seed'], config['reproducibility'])
    dataset = create_dataset(config)

    model = FlowCF(config, dataset).to(config['device'])
    
    # 가중치 로드
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=config['device'])
        model.load_state_dict(checkpoint['state_dict'])
        print(f">>> [System] 모델 로드 완료: {args.checkpoint}")
    else:
        print(">>> [Error] 모델 파일이 없습니다.")
        sys.exit(1)
        
    model.eval()

    # ----------------------------------------------------------------------
    # [Cold Start 평가 시작]
    # ----------------------------------------------------------------------
    print(f"\n>>> [Evaluation] 무작위 영화 {args.test_items}개를 뽑아 Cold Start 성능을 테스트합니다...")
    print("    (각 영화의 유저 중 80%를 힌트로 주고, 나머지 20%를 맞추는지 확인)")
    
    # Swap 되었으므로: dataset.user_num = 실제 영화(Item) 개수
    # dataset.item_num = 실제 유저(User) 개수
    
    # 전체 영화(가상의 유저 ID) 리스트
    all_movie_indices = np.arange(dataset.user_num)
    
    # 랜덤하게 테스트할 영화 뽑기
    np.random.shuffle(all_movie_indices)
    test_movie_indices = all_movie_indices[:args.test_items]

    total_recall = 0.0
    total_ndcg = 0.0
    valid_count = 0

    # Interaction Matrix (누가 뭘 봤는지 전체 데이터)
    # inter_feat는 DataFrame 형태
    df = dataset.inter_feat
    
    # 컬럼명 가져오기 (Swap된 상태 고려)
    # uid_field -> 실제 영화 ID 컬럼 / iid_field -> 실제 유저 ID 컬럼
    col_movie = dataset.uid_field 
    col_user = dataset.iid_field

    for movie_idx in test_movie_indices:
        # 1. 이 영화를 본 모든 유저(Token) 찾기
        # DataFrame에서 해당 movie_idx를 가진 행을 찾음
        mask = (df[col_movie] == movie_idx)
        users_who_liked = df[mask][col_user].values

        # 데이터가 너무 적으면(예: 5명 미만) 테스트에서 제외
        if len(users_who_liked) < 10:
            continue

        # 2. 80% 힌트(Seed) / 20% 정답(Truth) 분리
        np.random.shuffle(users_who_liked)
        split_point = int(len(users_who_liked) * 0.8)
        
        seed_users = users_who_liked[:split_point]
        ground_truth_users = users_who_liked[split_point:]

        if len(ground_truth_users) == 0:
            continue

        # 3. 모델 입력 만들기
        input_vector = torch.zeros((1, dataset.item_num)).to(config['device'])
        input_vector[0, seed_users] = 1.0 # 힌트 주입

        # 4. 모델 추론
        original_history = model.history_item_matrix if hasattr(model, 'history_item_matrix') else None
        model.history_item_matrix = input_vector
        
        dummy_inter = Interaction({dataset.uid_field: torch.tensor([0]).to(config['device'])})
        
        with torch.no_grad():
            scores = model.full_sort_predict(dummy_inter)
        
        if original_history is not None:
            model.history_item_matrix = original_history

        # 5. 점수 계산
        scores = scores.view(-1)
        scores[seed_users] = -np.inf # 힌트로 준 건 정답에서 제외
        
        top_k = 10
        _, top_indices = torch.topk(scores, top_k)
        top_indices = top_indices.cpu().numpy()

        # 지표 계산
        rec, ndcg = compute_metrics(top_indices, ground_truth_users, k=10)
        
        total_recall += rec
        total_ndcg += ndcg
        valid_count += 1
        
        if valid_count % 10 == 0:
            print(f"    -> 진행률: {valid_count}/{args.test_items} 완료...")

    # 최종 결과 출력
    if valid_count > 0:
        avg_recall = total_recall / valid_count
        avg_ndcg = total_ndcg / valid_count
        print("\n" + "="*50)
        print(f" [최종 성적표] 테스트한 영화 수: {valid_count}개")
        print(f" 🎯 Recall@10: {avg_recall:.4f}")
        print(f" 🌟 NDCG@10  : {avg_ndcg:.4f}")
        print("="*50)
        print(" 해석: Recall@10이 0.1(10%) 이상이면 꽤 쓸만한 모델입니다.")
    else:
        print("[Warning] 테스트할 수 있는 충분한 데이터가 있는 영화가 없습니다.")
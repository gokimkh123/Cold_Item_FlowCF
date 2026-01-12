import argparse
import torch
import numpy as np
import os
import sys

# [중요] model 패키지 경로 인식
sys.path.append(os.getcwd())

from recbole.config import Config
from recbole.data import create_dataset
from recbole.utils import init_seed
from recbole.data.interaction import Interaction
from model.flowcf import FlowCF  # FlowCF 클래스 직접 임포트

# --------------------------------------------------------------------------
# 유저/아이템 정보 로드 함수 (ML-1M 데이터셋 기준)
# --------------------------------------------------------------------------
def load_maps(dataset_path):
    # 아이템(영화) 정보 로드
    item_file = os.path.join(dataset_path, 'ml-1m.item')
    movie_id2title = {}
    if os.path.exists(item_file):
        with open(item_file, 'r', encoding='iso-8859-1') as f:
            lines = f.readlines()
            sep = '::' if '::' in lines[1] else '\t'
            start = 0 if '::' in lines[1] else 1
            for line in lines[start:]:
                parts = line.strip().split(sep)
                if len(parts) >= 2:
                    movie_id2title[parts[0]] = parts[1]
    
    # 유저 정보 로드
    user_file = os.path.join(dataset_path, 'ml-1m.user')
    user_id2info = {}
    if os.path.exists(user_file):
        with open(user_file, 'r', encoding='iso-8859-1') as f:
            lines = f.readlines()
            sep = '::' if '::' in lines[1] else '\t'
            start = 0 if '::' in lines[1] else 1
            for line in lines[start:]:
                parts = line.strip().split(sep)
                if len(parts) >= 3:
                    # 성별, 연령대 정보 포맷팅
                    user_id2info[parts[0]] = f"Gender:{parts[1]}, Age:{parts[2]}"
    return movie_id2title, user_id2info

# ==========================================================================
# 메인 실행 코드
# ==========================================================================
if __name__ == '__main__':
    # 1. 커맨드라인 인자 파싱 (파일명 입력받기)
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint file')
    # parse_known_args()를 써야 RecBole 내부 인자와 충돌하지 않음
    args, _ = parser.parse_known_args()

    # 2. 설정 로드
    print(">>> [System] 설정을 로드합니다...")
    config = Config(model=FlowCF, config_file_list=['flowcf.yaml'])
    init_seed(config['seed'], config['reproducibility'])

    # 3. 데이터셋 구조 로드 (ID 매핑용)
    dataset = create_dataset(config)

    # 4. 모델 초기화
    model = FlowCF(config, dataset).to(config['device'])
    
    # 5. [핵심] 학습된 가중치(Checkpoint) 로드
    checkpoint_path = args.checkpoint
    print(f">>> [System] 모델 파일을 로드합니다: {checkpoint_path}")
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=config['device'])
        model.load_state_dict(checkpoint['state_dict'])
        print(">>> [System] 가중치 로드 성공! (학습된 모델로 추론합니다)")
    else:
        print(f">>> [Error] 파일이 존재하지 않습니다: {checkpoint_path}")
        sys.exit(1)

    model.eval()

    # 6. 메타데이터(유저 정보) 로드
    movie_title_map, user_info_map = load_maps(config['data_path'])

    # --------------------------------------------------------------------------
    # Item Cold Start 시뮬레이션
    # --------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print(" >>> Item Cold Start 시뮬레이션 시작")
    print("=" * 50)

    # 시드 유저 입력
    seed_real_users = ['1', '10', '100', '55']
    print(f"\n[입력 상황] 신규 영화 개봉! 얼리어답터 유저들: {seed_real_users}")

    # 입력 벡터 생성 (Multi-hot)
    total_users = dataset.item_num  # Swap 되었으므로 item_num이 유저 수
    input_vector = torch.zeros((1, total_users)).to(config['device'])

    seed_tokens = []
    for real_uid in seed_real_users:
        try:
            # Swap 되었으므로 iid_field가 user_id
            token = dataset.token2id(dataset.iid_field, real_uid)
            input_vector[0, token] = 1.0
            seed_tokens.append(token)
        except ValueError:
            pass

    # 모델에 히스토리 주입 (Monkey Patching)
    original_history = model.history_item_matrix if hasattr(model, 'history_item_matrix') else None
    model.history_item_matrix = input_vector 

    dummy_interaction = Interaction({
        dataset.uid_field: torch.tensor([0]).to(config['device']), 
    })

    # 추론 실행
    with torch.no_grad():
        prediction_scores = model.full_sort_predict(dummy_interaction)
    
    # 히스토리 원상복구
    if original_history is not None:
        model.history_item_matrix = original_history

    # 결과 분석 (상위 10명 추출)
    scores = prediction_scores.view(-1)
    scores[seed_tokens] = -np.inf  # 이미 본 사람은 제외
    
    top_k = 10
    vals, indices = torch.topk(scores, top_k)
    
    print(f"\n[분석 결과] 이 영화를 좋아할 잠재 관객 Top {top_k}:")
    indices = indices.cpu().numpy()
    
    for rank, idx in enumerate(indices):
        real_user_id = dataset.id2token(dataset.iid_field, idx)
        user_info = user_info_map.get(real_user_id, "Unknown Info")
        print(f"  Rank {rank+1}: User {real_user_id} \t| {user_info}")
    
    print("\n[완료] 분석이 끝났습니다.")
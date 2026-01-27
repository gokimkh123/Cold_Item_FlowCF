# inference.py
import tensorflow as tf
import numpy as np
import yaml
import sys
from src.data_loader import ColdStartDataLoader
from src.model import FlowModel
from src.flow_logic import BernoulliFlow

def recommend(item_token, k=10):
    # 1. 설정 및 데이터 로드
    config = yaml.safe_load(open("config.yaml", 'r', encoding='utf-8'))
    loader = ColdStartDataLoader(config)
    num_items, num_users = loader.build()
    
    # 2. 모델 초기화 및 가중치 복원
    model = FlowModel(config['dims_mlp'] + [num_users], config['time_embedding_size'])
    
    # 레이어 빌드를 위한 Dummy Forward
    dummy_x = tf.zeros((1, num_users))
    dummy_cond = tf.zeros((1, loader.side_dim))
    dummy_t = tf.zeros((1, 1))
    model(dummy_x, dummy_cond, dummy_t, training=False)
    
    # 가중치 로드
    model.load_weights("saved_model/flow_model_bernoulli_epoch_200").expect_partial()
    
    flow = BernoulliFlow(loader.user_activity)
    
    # 3. 아이템 ID 확인 및 Side Info 추출
    token_str = str(item_token)
    if token_str not in loader.entity2id:
        print(f"\n[Error] 아이템 '{token_str}'을 데이터셋에서 찾을 수 없습니다.")
        return

    # 오타 수정: token_key -> token_str
    eid = loader.entity2id[token_str]
    cond = tf.expand_dims(loader.side_emb[eid], 0)
    
    # 4. Bernoulli Flow Inference
    print(f"\n>>> 아이템 '{token_str}'에 대한 유저 추천 생성 중 (20 steps)...")
    x_t = flow.get_prior_sample(1)
    steps = 20  # 추천 품질을 위해 20단계로 설정
    dt = 1.0 / steps
    
    for i in range(steps):
        t_val = i * dt
        t = tf.constant([[t_val]], dtype=tf.float32)
        pred = model(x_t, cond, t, training=False)
        x_t = flow.inference_step(x_t, pred, t_val, dt)
        
    # 5. 결과 랭킹 출력
    scores = x_t.numpy().flatten()
    top_indices = np.argsort(scores)[-k:][::-1]
    
    print("\n" + "="*45)
    print(f"아이템 '{item_token}' 추천 TOP {k}")
    print("-" * 45)
    for rank, idx in enumerate(top_indices):
        user_token = loader.target_tokens[idx]
        print(f"순위 {rank+1}: 유저 {user_token:<8} (점수: {scores[idx]:.4f})")
    print("="*45)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        recommend(sys.argv[1])
    else:
        print("사용법: python3 inference.py [item_id]")
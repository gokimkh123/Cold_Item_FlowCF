import pandas as pd
import os

# [수정] 찾은 경로로 업데이트 (중간에 /BPR_cv/ 폴더 추가됨)
train_file = "/app/dataset/ML1M/BPR_cv/BPR_cv.train.inter"
vali_file = "/app/dataset/ML1M/BPR_cv/BPR_cv.vali.inter"  # <--- 중요!

print(f">>> 검증 파일 경로: {vali_file}")

try:
    # 1. Validation 데이터 로드
    df_vali = pd.read_csv(vali_file, sep='\t')
    
    # 2. 영화(mid) 별 유저(uid) 수 카운트
    # 컬럼명 자동 감지
    mid_col = next((c for c in df_vali.columns if 'mid' in c), None)
    if not mid_col: mid_col = df_vali.columns[1]

    gt_counts = df_vali.groupby(mid_col).size()
    
    print(f"\n[Validation Set 분석]")
    print(f"총 인터랙션 수: {len(df_vali)}")
    print(f"총 영화(Target) 수: {len(gt_counts)}")
    print(f"영화 당 평균 시청 유저 수 (Ground Truth Size): {gt_counts.mean():.2f}")
    
    # 3. Recall@10의 이론적 최대치 계산
    max_recall_at_10 = (10 / gt_counts).apply(lambda x: min(x, 1.0)).mean()
    print(f"\n[이론적 한계 검증]")
    print(f"이 데이터셋에서 얻을 수 있는 이론적 최대 Recall@10: {max_recall_at_10:.4f}")
    print(f"현재 모델의 Recall@10: 0.063")
    
except Exception as e:
    print(f"에러 발생: {e}")
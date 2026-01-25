# trainer.py
import torch
import numpy as np
from recbole.trainer import Trainer
from tqdm import tqdm

class ColdStartTrainer(Trainer):
    def __init__(self, config, model):
        super(ColdStartTrainer, self).__init__(config, model)
        # 평가에 사용할 k 리스트 (설정 파일에서 가져오거나 기본값 사용)
        self.k_list = config['topk'] if 'topk' in config else [10, 20]

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        """
        RecBole의 기본 evaluate 대신, 우리의 커스텀 Cold-Start 평가 로직을 수행합니다.
        """
        if load_best_model:
            if model_file:
                checkpoint = torch.load(model_file)
                self.model.load_state_dict(checkpoint['state_dict'])
            self.model.eval()

        # 1. 검증 데이터에서 유저별 정답(Ground Truth) 추출
        # eval_data는 RecBole DataLoader 객체입니다.
        # 내부의 DataFrame에 접근하여 uid, mid 정보를 가져옵니다.
        df = eval_data.dataset.inter_feat
        # 컬럼명 매핑 (uid: 사람, mid: 영화) - yaml 설정에 따름
        # yaml에서 USER_ID_FIELD: mid, ITEM_ID_FIELD: uid로 되어 있음
        
        # 하지만 여기서 df는 텐서가 아닌 원본 데이터프레임 형태에 가까운 내부 데이터를 씁니다.
        # RecBole 데이터셋 구조상 직접 접근이 까다로우므로, 
        # 우리가 짠 evaluate.py와 동일한 로직을 수행하기 위해 
        # eval_data(Validation Set)에 있는 모든 mid(영화)를 예측합니다.
        
        # 평가 대상 영화 (Validation Set에 있는 모든 영화)
        # Note: RecBole의 dataset.inter_feat['mid']는 내부 ID로 매핑된 상태일 수 있습니다.
        # 안전하게 토큰(원본ID) 기준으로 하지 않고 내부 ID 기준으로 바로 계산합니다.
        
        # [중요] RecBole Trainer와 호환성을 위해 구조를 맞춥니다.
        self.model.eval()
        
        # --- Matrix Construction (Evaluate.py와 동일 로직) ---
        
        # 1. Validation 데이터에 존재하는 모든 영화(mid) 식별
        # dataset.inter_feat는 딕셔너리 형태의 텐서들을 담고 있음
        mid_field = self.config['USER_ID_FIELD'] # mid
        uid_field = self.config['ITEM_ID_FIELD'] # uid
        
        # 전체 인터랙션 가져오기
        try:
            mids = eval_data.dataset.inter_feat[mid_field].numpy()
            uids = eval_data.dataset.inter_feat[uid_field].numpy()
        except:
             # Tensor일 경우
            mids = eval_data.dataset.inter_feat[mid_field].cpu().numpy()
            uids = eval_data.dataset.inter_feat[uid_field].cpu().numpy()
            
        # 유저별 정답 딕셔너리 생성 {uid: {mid1, mid2...}}
        user_gt = {}
        for u, m in zip(uids, mids):
            if u not in user_gt:
                user_gt[u] = set()
            user_gt[u].add(m)
            
        unique_mids = np.unique(mids)
        
        # 2. Score Matrix 생성
        # score_matrix[mid_idx] -> [scores for all users]
        # 편의상 딕셔너리로 저장 {internal_mid: scores_vector}
        mid_score_map = {}
        
        # 모든 unique_mid에 대해 예측
        for mid in unique_mids:
            # predict_cold_item은 내부 ID(int)를 받아서 전체 유저 점수 리턴
            scores = self.model.predict_cold_item(int(mid))
            mid_score_map[mid] = scores.cpu().numpy().flatten()
            
        # 3. User-wise Evaluation
        total_recall = {k: 0.0 for k in self.k_list}
        total_ndcg = {k: 0.0 for k in self.k_list}
        valid_users = 0
        
        for uid, true_mids in user_gt.items():
            # 이 유저에 대한 모든 영화의 점수를 가져와야 함
            # -> score_matrix를 (N_movies, N_users)로 만들지 않고
            #    여기서는 (N_candidate_movies) 만큼만 비교하면 됨
            
            # 현재 Validation Set에 있는 영화들(unique_mids) 중에서 순위를 매겨야 함
            # (Standard Protocol: Test Set에 있는 아이템들 중에서 랭킹)
            
            candidate_mids = unique_mids
            candidate_scores = []
            
            # 이 유저(uid)에 대한 각 영화의 점수를 모음
            for mid in candidate_mids:
                # mid_score_map[mid] 는 [n_users] 크기
                # 그 중에서 uid 번째 점수만 가져옴
                score = mid_score_map[mid][uid]
                candidate_scores.append(score)
                
            # 랭킹 정렬
            candidate_scores = np.array(candidate_scores)
            top_indices = np.argsort(candidate_scores)[-max(self.k_list):][::-1]
            
            # Top-K 영화 ID 복원
            rec_mids = [candidate_mids[i] for i in top_indices]
            
            # Metric 계산
            for k in self.k_list:
                hits = 0
                sum_r = 0.0
                curr_recs = rec_mids[:k]
                
                for i, r_mid in enumerate(curr_recs):
                    if r_mid in true_mids:
                        hits += 1
                        sum_r += 1.0 / np.log2(i + 2)
                
                recall = hits / len(true_mids)
                
                idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(true_mids), k))])
                ndcg = sum_r / idcg if idcg > 0 else 0.0
                
                total_recall[k] += recall
                total_ndcg[k] += ndcg
                
            valid_users += 1
            
        # 4. 결과 리턴 (RecBole 형식인 OrderedDict로 반환해야 함)
        from collections import OrderedDict
        metric_dict = OrderedDict()
        
        # 대표 지표 (Early Stopping 기준) - 보통 Recall@20 사용
        # 키 이름은 RecBole 규칙을 따르거나 로그에서 식별 가능한 이름 사용
        for k in self.k_list:
            metric_dict[f'recall@{k}'] = total_recall[k] / valid_users
            metric_dict[f'ndcg@{k}'] = total_ndcg[k] / valid_users
            
        return metric_dict
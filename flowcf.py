import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils.enum_type import InputType
import math
import os

class FlowModel(nn.Module):
    def __init__(self, dims, time_emb_size, time_type="cat", dropout_prob=0.0, act_func="gelu", norm=False):
        super(FlowModel, self).__init__()
        self.dims = dims
        self.time_type = time_type
        self.time_emb_dim = time_emb_size
        self.norm = norm
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        
        # 활성화 함수 선택
        if act_func.lower() == 'leakyrelu':
            activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            activation = nn.GELU()

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(activation)
                layers.append(nn.Dropout(p=dropout_prob))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        time_emb = timestep_embedding_pi(t, self.time_emb_dim).to(x.device)
        time_emb = self.emb_layer(time_emb)
        if self.time_type == "cat":
            x = torch.cat([x, time_emb], dim=-1)
        out = self.mlp(x)
        return out

class FlowCF(GeneralRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(FlowCF, self).__init__(config, dataset)

        # 1. 설정 로드
        if 'prior_type' in config:
            self.prior_type = config['prior_type'].lower()
        else:
            self.prior_type = 'gaussian'

        if 'act_func' in config:
            self.act_func = config['act_func'].lower()
        else:
            self.act_func = 'gelu'
        
        print(f"\n[Model Config] Prior: {self.prior_type.upper()} | Activation: {self.act_func.upper()}\n")

        # 2. Side Info 로드 및 정렬
        npy_path = os.path.join("dataset", "ML1M", "mv-tag-emb.npy")
        if not os.path.exists(npy_path):
            npy_path = "/app/dataset/ML1M/mv-tag-emb.npy"
        
        raw_emb = np.load(npy_path)
        self.raw_side_emb = torch.FloatTensor(raw_emb).to(self.device)
        self.side_dim = self.raw_side_emb.shape[1]

        # Re-alignment
        n_movies = self.n_users 
        aligned_emb = torch.zeros((n_movies, self.side_dim)).to(self.device)
        for internal_id in range(1, n_movies):
            try:
                raw_token = dataset.id2token(dataset.uid_field, internal_id)
                raw_id = int(raw_token)
                if 0 <= raw_id < self.raw_side_emb.shape[0]:
                    aligned_emb[internal_id] = self.raw_side_emb[raw_id]
            except:
                pass
        self.side_emb = aligned_emb

        # 3. History Matrix 생성
        inter_matrix = dataset.inter_matrix(form='csr').astype('float32')
        self.history_matrix = torch.FloatTensor(inter_matrix.toarray()).to(self.device)

        # 4. Bernoulli Prior를 위한 유저 활동성 계산
        if self.prior_type == 'bernoulli':
            self.user_activity = self._get_user_activity(dataset)

        # 5. 모델 구축
        self.target_dim = self.n_items 
        self.input_dim = self.target_dim + self.side_dim
        self.n_steps = config["n_steps"]
        # yaml에 s_steps가 없으면 기본값 1 (모든 스텝 밟기)로 설정
        self.s_steps = config["n_steps"]
        self.time_emb_size = config["time_embedding_size"]
        
        # [작성자님 요청 수정] Time Steps 명시적 정의
        # n_steps가 5라면 [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] 생성
        self.time_steps = torch.linspace(0, 1, self.n_steps + 1).to(self.device)

        self.dims_mlp = [self.input_dim + self.time_emb_size] + config["dims_mlp"] + [self.target_dim]

        self.net = FlowModel(
            dims=self.dims_mlp,
            time_emb_size=self.time_emb_size,
            dropout_prob=config['dropout'],
            time_type="cat",
            act_func=self.act_func,
            norm=False
        ).to(self.device)

    def _get_user_activity(self, dataset):
        activity = self.history_matrix.mean(dim=0)
        return torch.clamp(activity, 1e-6, 1.0 - 1e-6)

    def calculate_loss(self, interaction):
        movie_ids = interaction[self.USER_ID]
        x_1 = self.history_matrix[movie_ids]
        cond = self.side_emb[movie_ids]
        batch_size = x_1.size(0)

        # 1. Discrete Time Sampling (0 ~ n_steps-1)
        steps = torch.randint(0, self.n_steps, (batch_size,), device=self.device)
        
        # 2. Time Fetching (Mixup을 위해 2차원으로 변환: [batch, 1])
        t = self.time_steps[steps].view(-1, 1)
        
        # 변수명 통일 (t_expand 사용)
        t_expand = t

        if self.prior_type == 'bernoulli':
            prior_probs = self.user_activity.expand(batch_size, -1)
            x_0 = torch.bernoulli(prior_probs).to(self.device)
            mask = torch.rand_like(x_1) <= t_expand
            x_t = torch.where(mask, x_1, x_0)
            target = x_1
        else:
            x_0 = torch.randn_like(x_1).to(self.device)
            x_t = (1 - t_expand) * x_0 + t_expand * x_1
            target = x_1 - x_0

        net_input = torch.cat([x_t, cond], dim=1)
        
        # [수정된 부분] 모델에 넣을 때는 1차원으로 차원 축소 (.squeeze(-1))
        # (Batch, 1) -> (Batch, )
        pred = self.net(net_input, t.squeeze(-1))

        loss = F.mse_loss(pred, target)
        return loss

    def full_sort_predict(self, interaction):
        movie_ids = interaction[self.USER_ID]
        cond = self.side_emb[movie_ids]
        batch_size = movie_ids.size(0)

        if self.prior_type == 'bernoulli':
            prior_probs = self.user_activity.expand(batch_size, -1)
            x_t = torch.bernoulli(prior_probs).to(self.device)
        else:
            x_t = torch.randn(batch_size, self.target_dim).to(self.device)

        # [수정] self.time_steps를 사용하여 루프 실행
        # 0부터 n_steps-1까지 반복
        for i in range(self.n_steps):
            # 현재 시간과 다음 시간 가져오기
            t_current = self.time_steps[i]
            t_next = self.time_steps[i+1]
            dt = t_next - t_current # 구간 크기 (일정하지만 안전하게 계산)

            # 텐서 형태로 변환 (배치 크기에 맞춤)
            t_tensor = t_current.expand(batch_size)
            
            net_input = torch.cat([x_t, cond], dim=1)
            pred = self.net(net_input, t_tensor)
            
            if self.prior_type == 'bernoulli':
                if t_current >= 1.0: break
                v_t = (pred - x_t) / (1.0 - t_current + 1e-5)
                x_t = x_t + v_t * dt
            else:
                x_t = x_t + pred * dt
        
        return x_t

    def predict(self, interaction):
        item = interaction[self.ITEM_ID]
        x_t = self.full_sort_predict(interaction)
        scores = x_t.gather(1, item.unsqueeze(1)).squeeze(1)
        return scores

    def predict_cold_item(self, item_id_or_emb, num_samples=10):
        # 1. Side Info (Condition) 준비
        if isinstance(item_id_or_emb, int):
            if item_id_or_emb < self.side_emb.shape[0]:
                 cond = self.side_emb[item_id_or_emb].unsqueeze(0)
            else:
                 cond = torch.zeros(1, self.side_dim).to(self.device)
        else:
            cond = torch.FloatTensor(item_id_or_emb).to(self.device)
            if cond.dim() == 1:
                cond = cond.unsqueeze(0)

        cond_expanded = cond.repeat(num_samples, 1)
        batch_size = num_samples
        
        # 2. 초기 노이즈 설정 (Cold Item은 무조건 Noise에서 시작)
        if self.prior_type == 'bernoulli':
            prior_probs = self.user_activity.expand(batch_size, -1)
            x_t = torch.bernoulli(prior_probs).to(self.device)
        else:
            x_t = torch.randn(batch_size, self.target_dim).to(self.device)

        # 3. Strided Inference (성큼성큼 걷기)
        # s_steps를 '보폭(Stride)'으로 사용합니다.
        # 예: n_steps=5, s_steps=2 -> indices: 0 -> 2 -> 4 -> 5 (도착)
        
        current_step_idx = 0
        
        with torch.no_grad():
            while current_step_idx < self.n_steps:
                # (1) 현재 시간 t 가져오기
                t_current = self.time_steps[current_step_idx]
                
                # (2) 다음 멈출 시간 t 결정 (Stride만큼 점프)
                # 마지막 스텝(n_steps)을 넘어가지 않도록 min 처리
                next_step_idx = min(current_step_idx + self.s_steps, self.n_steps)
                
                t_next = self.time_steps[next_step_idx]
                dt = t_next - t_current # 실제 이동할 거리 (예: 0.2가 아니라 0.4가 됨)

                # (3) 모델에게 "지금 t_current인데 어디로 갈까?" 물어보기
                t_tensor = t_current.expand(batch_size)
                net_input = torch.cat([x_t, cond_expanded], dim=1)
                pred = self.net(net_input, t_tensor)
                
                # (4) Euler Step Update (보폭 dt만큼 이동)
                if self.prior_type == 'bernoulli':
                    if t_current >= 1.0: break
                    v_t = (pred - x_t) / (1.0 - t_current + 1e-5)
                    x_t = x_t + v_t * dt
                else:
                    x_t = x_t + pred * dt
                
                # (5) 인덱스 업데이트 (점프!)
                current_step_idx = next_step_idx

        return x_t.mean(dim=0, keepdim=True)

def timestep_embedding_pi(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(timesteps.device) * 2 * math.pi
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
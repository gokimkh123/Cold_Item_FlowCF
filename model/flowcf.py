from recbole.model.init import xavier_normal_initialization
from recbole.model.layers import MLPLayers
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import AutoEncoderMixin, GeneralRecommender
from recbole.utils.enum_type import InputType
import math
import typing


class FlowModel(nn.Module):
    def __init__(
        self,
        dims: typing.List,
        time_emb_size: int,
        time_type="cat",
        act_func="tanh",
        norm=False,
        init_dropout=0
    ):
        # PyTorch의 nn.Module 클래스를 상속받아 초기화합니다.
        super(FlowModel, self).__init__()

        # MLP의 각 레이어 차원 크기 목록을 저장합니다. (예: [아이템 수, 300, 300, 아이템 수]가 될 것입니다.)
        self.dims = dims

        # 시간 임베딩을 입력과 합치는 방식입니다. 현재는 "cat" (Concatenation, 결합)만 구현되어 있습니다.
        self.time_type = time_type

        # 시간 임베딩 벡터의 차원 크기를 저장합니다.
        self.time_emb_dim = time_emb_size

        # 입력 벡터 x를 정규화(Normalization)할지 여부를 저장합니다.
        self.norm = norm

        # 시간 임베딩을 한 번 더 처리할 선형 레이어(Fully Connected Layer)를 정의합니다.
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        # 시간 임베딩 방식이 "결합(cat)"일 경우의 처리입니다.
        if self.time_type == "cat":
            # Concatenate timestep embedding with input
            '''
            MLP의 첫 번째 입력 차원에 시간 임베딩 차원을 더합니다. 
            이는 입력 벡터 x와 시간 임베딩이 결합된(concatenate된) 후 MLP의 입력으로 들어가기 때문입니다.
            '''
            self.dims[0] += self.time_emb_dim
        else:
            raise ValueError(
                "Unimplemented timestep embedding type %s" % self.time_type
            )

        '''
        RecBole에서 제공하는 MLP 레이어 정의 함수입니다. 
        layers는 업데이트된 self.dims를 사용하여 입력 x + t와 출력(아이템 수) 크기를 갖는 MLP를 구성합니다.
        '''
        self.mlp_layers = MLPLayers(
            layers=self.dims, dropout=0.1, activation=act_func, last_activation=False
        )

        # 입력에 적용할 초기 드롭아웃 레이어를 정의합니다.
        self.init_dropout = nn.Dropout(init_dropout)

        # 모델의 모든 가중치를 Xavier 초기화 방식으로 초기화하여 학습이 안정적으로 시작되도록 합니다.
        self.apply(xavier_normal_initialization)

    def forward(self, x, t):

        # 외부 함수 timestep_embedding_pi를 호출하여 시간 t를 사인/코사인 기반의 임베딩 벡터로 변환합니다. 변환된 벡터는 x와 동일한 장치(CPU/GPU)에 배치됩니다.
        time_emb = timestep_embedding_pi(t, self.time_emb_dim).to(x.device)

        # 시간 임베딩을 정의된 선형 레이어 self.emb_layer를 통과시킵니다.
        emb = self.emb_layer(time_emb)
        if self.norm:
            # 입력 사용자 벡터 x를 정규화(Normalization)합니다.
            x = F.normalize(x)

        # 입력 x에 초기 드롭아웃을 적용하여 과적합을 방지합니다.
        x = self.init_dropout(x)

        '''
        사용자 벡터 x와 처리된 시간 임베딩 emb를 마지막 차원(dim=-1)을 기준으로 결합(Concatenate)합니다. 
        이 벡터 h가 MLP의 실제 입력이 됩니다.
        '''
        h = torch.cat([x, emb], dim=-1)

        '''
        결합된 입력 h를 MLP 레이어에 통과시켜 최종 출력을 계산합니다. 
        이 출력은 최종 상호작용 벡터에 대한 예측
        '''
        output = self.mlp_layers(h)
        return output


class FlowCF(GeneralRecommender, AutoEncoderMixin):
    input_type = InputType.LISTWISE

    def __init__(self, config, dataset):
        super(FlowCF, self).__init__(config, dataset)
        super().build_histroy_items(dataset)

       # 1. n_steps 가져오기
        if config['n_steps'] is not None:
            self.n_steps = config['n_steps']
        elif hasattr(config, 'final_config_dict') and 'n_steps' in config.final_config_dict:
            self.n_steps = config.final_config_dict['n_steps']
        else:
            self.n_steps = 10  # YAML에도 없으면 기본값

        # 2. s_steps 가져오기
        if config['s_steps'] is not None:
            self.s_steps = config['s_steps']
        elif hasattr(config, 'final_config_dict') and 's_steps' in config.final_config_dict:
            self.s_steps = config.final_config_dict['s_steps']
        else:
            self.s_steps = 10   # YAML에도 없으면 기본값
        print(f">>> [FlowCF Config Check] YAML 설정값 로드 성공 -> n_steps: {self.n_steps}, s_steps: {self.s_steps}")
        self.time_steps = torch.linspace(0, 1, self.n_steps + 1)

        self.time_emb_size = config["time_embedding_size"]
        dims = [self.n_items] + config["dims_mlp"] + [self.n_items]

        self.flow_model = FlowModel(
            dims=dims,
            time_emb_size=self.time_emb_size,
            init_dropout=0
        )

        self.item_frequencies = self.get_item_frequencies()

    def get_item_frequencies(self):
        item_counts = torch.zeros(self.n_items, device=self.device)
        for user_id in range(self.n_users):
            user_item_ids = self.history_item_id[user_id]
            item_counts[user_item_ids] += 1

        item_frequencies = item_counts / self.n_users
        return item_frequencies

    def forward(self, x, t):
        return self.flow_model(x, t)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        x1 = self.get_rating_matrix(user)

        # Randomly sample steps
        steps = torch.randint(0, self.n_steps, (x1.size(0),), device=self.device)
        t = self.time_steps.to(x1.device)[steps].unsqueeze(1)

        # Sample x0 from behavior-guided prior
        x0 = torch.bernoulli(self.item_frequencies.expand(
            x1.size(0), -1)).to(x1.device)
        random_mask = torch.rand_like(x1, dtype=torch.float32) <= t
        # Interpolation between x0 and x1
        xt = torch.where(random_mask, x1, x0)

        model_output = self.forward(xt, t.squeeze(-1))
        loss = mean_flat((x1 - model_output) ** 2)
        loss = loss.mean()

        return loss

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        X_bar = self.get_rating_matrix(user)
        Xt = X_bar

        for i_t in range(self.n_steps-self.s_steps, self.n_steps):
            t = self.time_steps[i_t].repeat(Xt.shape[0], 1).to(X_bar.device)
            X1_hat = self.forward(Xt, t.squeeze(-1))
            if i_t == self.n_steps-1:
                break
            t_next = self.time_steps[i_t + 1].repeat(Xt.shape[0], 1).to(X_bar.device)
            v = (X1_hat - Xt) / (1 - t)
            Xt_pos_probs = Xt + v * (t_next - t)
            Xt_neg_probs = 1 - Xt_pos_probs
            Xt_probs = torch.stack([Xt_neg_probs, Xt_pos_probs], dim=-1)
            # Sample from Xt_probs, considering the unique nature of collaborative filtering, we only allow 0 -> 1
            Xt = Xt_probs.argmax(dim=-1)
            # Preverve the observed interactions
            Xt = torch.logical_or(X_bar.to(torch.bool), Xt.to(torch.bool))
            Xt = Xt.float()

        return X1_hat

    def predict(self, interaction):
        item = interaction[self.ITEM_ID]
        x_t = self.full_sort_predict(interaction)
        scores = x_t[:, item]
        return scores


def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def timestep_embedding_pi(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(
        timesteps.device
    ) * 2 * math.pi  # shape (dim//2,)
    args = timesteps[:, None].float() * freqs[None]  # (N, dim//2)
    embedding = torch.cat(
        [torch.cos(args), torch.sin(args)], dim=-1)  # (N, (dim//2)*2)
    if dim % 2:
        # zero pad in the last dimension to ensure shape (N, dim)
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

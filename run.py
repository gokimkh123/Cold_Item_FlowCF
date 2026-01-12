import sys
from logging import getLogger
import yaml
import argparse
from recbole.utils import init_logger, init_seed
from recbole.trainer import Trainer
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)
# 현 위치의 model 패키지를 불러오기 위한 import
from model import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the recommender system model')
    parser.add_argument('--config', type=str, default='flowcf.yaml', help='Path to the config file')
    args = parser.parse_args()

    # 1. Config 파일 로드
    with open(args.config, 'r', encoding='utf-8') as file:
        yaml_config = yaml.safe_load(file)
    model_config = yaml_config.get('model', None)
    
    # 2. RecBole 설정 초기화
    # locals()[model_config]를 통해 FlowCF 클래스를 동적으로 가져옵니다.
    config = Config(model=locals()[model_config], config_file_list=[args.config])
    init_seed(config['seed'], config['reproducibility'])
    
    # 3. 로거 초기화
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # 4. 데이터셋 로드 및 분할
    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # 5. 모델 로드 및 초기화
    model = locals()[config['model']](config, train_data.dataset).to(config['device'])
    logger.info(model)
    
    # [수정] FLOPs 계산 끄기
    # 커스텀 모델에서 이 함수는 자주 에러를 일으키므로 주석 처리합니다.
    # transform = construct_transform(config)
    # flops = get_flops(model, dataset, config["device"], logger, transform)
    # logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # 6. 트레이너 로드 및 학습 시작
    trainer = Trainer(config, model)

    # 학습 (Train)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, show_progress=config["show_progress"]
    )

    # 평가 (Test)
    test_result = trainer.evaluate(
        test_data, show_progress=config["show_progress"]
    )
    
    # 환경 정보 출력
    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")
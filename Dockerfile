# 1. Base Image
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# 2. 기본 설정
WORKDIR /app
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 3. 필수 도구 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. LightGBM, XGBoost (Conda로 설치 - 컴파일 에러 방지)
RUN conda install -y lightgbm xgboost && conda clean -ya

# 5. Python 라이브러리 설치 (최신 버전 사용)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "numpy<2.0" recbole pyyaml

# 6. 소스 코드 복사
COPY . /app

# 7. 우리가 만든(패치된) utils.py를 라이브러리에 덮어쓰기
RUN python -c "import recbole.data; import os; import shutil; \
    dest = os.path.join(os.path.dirname(recbole.data.__file__), 'utils.py'); \
    shutil.copyfile('/app/utils.py', dest); \
    print(f'Successfully injected custom utils.py to: {dest}')"

# 8. 실행
CMD ["python", "run.py", "--config", "flowcf.yaml"]
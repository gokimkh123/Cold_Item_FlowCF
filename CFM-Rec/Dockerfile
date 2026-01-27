# GPU 지원을 위해 공식 TensorFlow 이미지를 사용합니다.
FROM tensorflow/tensorflow:2.15.0-gpu

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 라이브러리 업데이트 및 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# 파이썬 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사 (데이터셋은 제외하고 볼륨으로 연결하는 것을 추천)
COPY . .

# 컨테이너 실행 시 기본 명령어
CMD ["python", "test_loader.py"]
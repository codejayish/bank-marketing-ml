# Base image: lightweight Python (works well with SageMaker)
FROM python:3.9-slim


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        curl \
        ca-certificates \
        python3-dev \
        build-essential && \
    rm -rf /var/lib/apt/lists/*


ENV SM_MODEL_DIR=/opt/ml/model \
    SM_OUTPUT_DIR=/opt/ml/output \
    SM_INPUT_DIR=/opt/ml/input \
    SM_CHANNEL_TRAINING=/opt/ml/input/data/training \
    SM_CHANNEL_TESTING=/opt/ml/input/data/testing


WORKDIR /opt/ml/code


COPY container-requirements.txt .
RUN pip install --no-cache-dir -r container-requirements.txt


COPY program/ ./program/


ENTRYPOINT ["python3"]


CMD ["program/train.py"]

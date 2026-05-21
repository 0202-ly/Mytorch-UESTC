# Mytorch-UESTC：训练任务容器镜像（可与 Kubernetes Job / Kubeflow 组件共用）
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends graphviz \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV PYTHONUNBUFFERED=1
ENV MYTORCH_HEADLESS=1

CMD ["python", "train_MNIST.py", "--no-viz", "--epochs", "5", "--data-root", "/data"]

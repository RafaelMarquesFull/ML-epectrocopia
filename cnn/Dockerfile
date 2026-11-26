# Construir a partir da raiz do projeto:
#   docker build -f cnn/Dockerfile -t diabetes-cnn .
# Os dados não entram na imagem; são montados em /data na hora de rodar.
FROM python:3.12-slim
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir numpy==2.1.3 scikit-learn==1.5.2
WORKDIR /app
COPY common/ common/
COPY cnn/train.py cnn/train.py
ENTRYPOINT ["python", "-u", "cnn/train.py"]

# Construir a partir da raiz do projeto:
#   docker build -f svm/Dockerfile -t diabetes-svm .
# Os dados não entram na imagem; são montados em /data na hora de rodar.
FROM python:3.12-slim
RUN pip install --no-cache-dir numpy==2.1.3 scikit-learn==1.5.2
WORKDIR /app
COPY common/ common/
COPY svm/train.py svm/train.py
ENTRYPOINT ["python", "-u", "svm/train.py"]

# Construir a partir da raiz do projeto:
#   docker build -f leaky_relu/Dockerfile -t diabetes-leaky-relu .
# Os dados não entram na imagem; são montados em /data na hora de rodar.
FROM python:3.12-slim
RUN pip install --no-cache-dir numpy==2.1.3
# 1 thread de BLAS: as matrizes são pequenas e várias threads só atrapalham
ENV OPENBLAS_NUM_THREADS=1
WORKDIR /app
COPY common/ common/
COPY leaky_relu/train.py leaky_relu/train.py
ENTRYPOINT ["python", "-u", "leaky_relu/train.py"]

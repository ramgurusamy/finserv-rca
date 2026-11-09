# Application image. Model serving is a separate deployment (vLLM on GPU nodes).
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY config ./config
COPY data ./data
COPY eval ./eval
RUN pip install --no-cache-dir .
USER 1000
ENTRYPOINT ["rca"]

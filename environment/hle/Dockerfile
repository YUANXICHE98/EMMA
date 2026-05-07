FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace:/workspace/environment:/workspace/exp

WORKDIR /workspace

COPY environment/hle/requirements.txt /tmp/hle-requirements.txt

RUN pip install --upgrade pip && \
    pip install -r /tmp/hle-requirements.txt

COPY . /workspace

ENTRYPOINT ["python", "environment/run_memrl_benchmark.py", "--benchmark", "hle"]

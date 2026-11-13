FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

RUN useradd -m -u 1000 cloudpilot

COPY allocation/ ./allocation/
COPY datasets/ ./datasets/
COPY ml/ ./ml/
COPY backend/ ./backend/
COPY main.py README.md ./

RUN chown -R cloudpilot:cloudpilot /app
USER cloudpilot

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=5s \
  CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/v1/health' % os.environ.get('PORT', '8000'), timeout=2)"]

CMD ["python", "-m", "backend.server", "--host", "0.0.0.0"]

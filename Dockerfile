FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir requests pillow numpy

COPY wbsync.py ./

ENV WB_DATA_DIR=/data
EXPOSE 8430
HEALTHCHECK --interval=60s --timeout=5s --start-period=15s \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8430/healthz')" || exit 1

CMD ["python3", "wbsync.py"]

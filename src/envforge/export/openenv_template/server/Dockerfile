FROM python:3.13-slim
WORKDIR /app
COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY . /app
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]

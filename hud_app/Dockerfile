FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir hud-python==0.6.7

COPY env.py /app/env.py

CMD ["hud", "serve", "env.py", "--host", "0.0.0.0"]

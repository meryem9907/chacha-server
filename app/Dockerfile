FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
RUN useradd -m appuser

COPY ./requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt 

COPY entrypoint.sh .
RUN chmod +x ./entrypoint.sh
COPY . .

USER appuser

EXPOSE 127.0.0.1:5000:5000

ENTRYPOINT ["./entrypoint.sh"]
FROM denoland/deno:bin-2.9.7 AS deno

FROM python:3.12-slim
# ffmpeg cuts let's play audio, deno solves YouTube's JS challenges for yt-dlp
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates gosu \
    && rm -rf /var/lib/apt/lists/*
COPY --from=deno /deno /usr/local/bin/deno

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /data && chown appuser:appuser /data \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

ENV PYTHONUNBUFFERED=1 \
    DATA_DIR=/data \
    DATABASE_URL=sqlite:////data/games.db \
    PORT=8080
EXPOSE 8080
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips '*'"]

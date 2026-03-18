# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app


FROM base AS test

COPY . /app

RUN python -m pip install --upgrade pip \
    && pip install ".[dev]"

RUN pytest -q

RUN python -m loopr rank \
    --matches examples/quickstart/matches.csv \
    --participants examples/quickstart/participants.csv \
    --appearances examples/quickstart/appearances.csv \
    --output /tmp/rankings.csv \
    && test -s /tmp/rankings.csv


FROM base AS dist

COPY . /app

RUN python -m pip install --upgrade pip build \
    && python -m build


FROM scratch AS artifacts

COPY --from=dist /app/dist/ /

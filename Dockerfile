FROM python:3.9-slim

WORKDIR /app

COPY pyproject.toml poetry.lock /app/

RUN pip install --upgrade pip && \
    pip install poetry==1.8.5 && \
    poetry install --no-dev

COPY . /app/

RUN poetry install

CMD ["python", "train.py"]
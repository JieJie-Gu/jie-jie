FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir .

RUN mkdir -p data

EXPOSE 8000

CMD ["uvicorn", "smart_cs.main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]

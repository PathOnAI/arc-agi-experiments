FROM python:3.10

WORKDIR /code

COPY ./pyproject.toml ./poetry.lock* /code/

RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-root --only main

COPY . /code/

CMD ["fastapi", "run", "src/app.py", "--port", "80", "--host", "0.0.0.0"]

# If running behind a proxy like Nginx or Traefik add --proxy-headers
# CMD ["fastapi", "run", "app/main.py", "--port", "80", "--proxy-headers"]
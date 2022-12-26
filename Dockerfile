#syntax = docker/dockerfile:experimental

FROM python:3.8

WORKDIR /app
COPY requirements.txt .	
RUN pip install -r requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

COPY . .

RUN mkdir /uploads
	
CMD ["./flask_run.sh"]

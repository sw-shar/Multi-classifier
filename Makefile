IMAGE = swshar/art
CONTAINER = art

all: run

build: requirements.txt
	DOCKER_BUILDKIT=1 docker build . -t $(IMAGE)

run:
	RUN_DEBUG=1 ./run.sh $(IMAGE) $(CONTAINER)

requirements.txt: requirements.in
	pip-compile --verbose

push:
	docker push $(IMAGE)

test:
	./test.sh

exec:
	docker exec -it $(CONTAINER) bash

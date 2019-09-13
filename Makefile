APP_NAME=amirassov/pneumothorax
CONTAINER_NAME=pneumothorax

build:  ## Build the container
	nvidia-docker build -t ${APP_NAME} -f Dockerfile .

run: ## Run container
	nvidia-docker run \
		-e DISPLAY=unix${DISPLAY} -v /tmp/.X11-unix:/tmp/.X11-unix --privileged \
		--ipc=host \
		-itd \
		--name=${CONTAINER_NAME} \
		-v /mnt/ssd/videoanalytics/pneumothorax_data:/data \
		-v /mnt/ssd/videoanalytics/pneumothorax_dumps:/dumps \
		-v $(shell pwd):/kaggle-pneumothorax $(APP_NAME) bash

exec: ## Run a bash in a running container
	nvidia-docker exec -it ${CONTAINER_NAME} bash

stop: ## Stop and remove a running container
	docker stop ${CONTAINER_NAME}; docker rm ${CONTAINER_NAME}
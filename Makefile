IMAGE_NAME := otus_rl_hw_04
CONTAINER_NAME := otus_rl_hw_04

.PHONY: build up down attach shell

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

attach:
	docker exec -it $(CONTAINER_NAME) bash

shell: build up attach

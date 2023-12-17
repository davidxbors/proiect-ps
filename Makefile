build:
	docker compose build

run:
	docker compose up

clean:
	rm ./src/data/*.html

super-clean:
	rm ./src/data/*
.PHONY: test lint train-smoke train-full clean

test:
	pytest tests/ -v --tb=short

lint:
	ruff check .

train-smoke:
	docker compose up train-smoke

train-full:
	docker compose up train-full

clean:
	rm -rf results/*.pt results/*.onnx mlruns/ mlflow.db

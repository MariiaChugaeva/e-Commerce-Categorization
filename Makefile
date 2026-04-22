.PHONY: venv activate run web

venv:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

activate:
	@echo "To activate the virtualenv run: source .venv/bin/activate (Unix) or .venv\Scripts\activate (Windows)"

run:
	.venv/bin/python service/main.py

web:
	.venv/bin/uvicorn service.web:app --host 0.0.0.0 --port 8000 --reload

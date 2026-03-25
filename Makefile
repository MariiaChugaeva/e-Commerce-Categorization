.PHONY: venv activate

venv:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

activate:
	@echo "To activate the virtualenv run: source .venv/bin/activate (Unix) or .venv\Scripts\activate (Windows)" 

run:
	.venv/bin/python service/main.py

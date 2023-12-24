all: venv/.update README.md

venv/.update: requirements.txt
	python3 -m venv venv
	venv/bin/pip install -r requirements.txt
	touch venv/.update

README.md: README.ipynb
	jupyter nbconvert --to markdown README.ipynb


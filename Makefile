all: venv/.update README.md

venv: venv/.update

venv/.update: requirements.txt
	python3 -m venv venv
	venv/bin/pip install -r requirements.txt
	venv/bin/pip install -e .
	touch venv/.update

README.md: README.ipynb
	jupyter nbconvert --to markdown README.ipynb

install:
	# install the package in the system
	/usr/bin/pip3 install .

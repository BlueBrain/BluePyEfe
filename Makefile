all: install
install:
	pip install -i https://bbpteam.epfl.ch/repository/devpi/bbprelman/dev/+simple --upgrade .
test: clean install_tox
	tox -v
test-gpfs: clean install_tox
	tox -v -e py27-gpfs
install_tox:
	pip install tox
clean:
	@find . -name "*.pyc" -exec rm -rf {} \;
	rm -rf BluePyEfe.egg-info
	rm -rf dist
	rm -rf testtype*
	rm -rf temptype*
doc: clean install_tox
	tox -v -e py3-docs
devpi:
	rm -rf dist
	python setup.py sdist
	upload2repo -t python -r dev -f `ls dist/BluePyEfe-*.tar.gz` 
	-upload2repo -t python -r release -f `ls dist/BluePyEfe-*.tar.gz`

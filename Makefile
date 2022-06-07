test: clean install_tox
	tox -v
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

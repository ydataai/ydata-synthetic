.PHONY: help lint test package clean install

help:	# The following lines will print the available commands when entering just 'make'
ifeq ($(UNAME), Linux)
	@grep -P '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
else
	@awk -F ':.*###' '$$0 ~ FS {printf "%15s%s\n", $$1 ":", $$2}' \
		$(MAKEFILE_LIST) | grep -v '@awk' | sort
endif

lint: ### Validates project with linting rules
	python -m pip install pylint
	python -m pylint src/

test: ### Runs all the project tests
	python -m pip install -r requirements-test.txt
	python -m pytest src/ydata_synthetic/tests

test_cov:
	python -m pip install -r requirements-test.txt
	python -m pytest --cov=. src/ydata_synthetic/tests

package: clean ### Runs the project setup
	echo "$(version)" > VERSION
	python -m setup.py sdist bdist_wheel

clean: ### Removes build binaries
	rm -rf build dist

install: ### Installs required dependencies
	python -m pip install dist/ydata-synthetic-$(version).tar.gz

install_test:
	echo "$(version)" > VERSION
	python -m pip install -e .

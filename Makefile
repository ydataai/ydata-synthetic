PYTHON = python3
PIP = pip3

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
	$(PIP) install pylint
	$(PYTHON) -m pylint src/

test: ### Runs all the project tests
	"Run tests"
	$(PIP) install pytest
	$(PYTHON) -m pytest tests/

package: clean ### Runs the project setup
	echo "$(version)" > VERSION
	$(PYTHON) setup.py sdist bdist_wheel

clean: ### Removes build binaries
	rm -rf build dist

install: ### Installs required dependencies
	$(PIP) install dist/ydata-synthetic-$(version).tar.gz




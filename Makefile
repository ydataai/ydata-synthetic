# The help target displays a list of available commands when the user enters 'make'
help:	# Print available commands
	@grep -P '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# The lint target checks the project for linting errors
lint: ### Validates project with linting rules
	$(PIP) install pylint
	$(PYTHON) -m pylint src/

# The test target runs all the project tests
test: ### Runs all the project tests
	"Run tests"
	$(PIP) install pytest
	$(PYTHON) -m pytest tests/

# The package target creates a source distribution and wheel distribution of the project
package: clean ### Runs the project setup
	echo "$(version)" > VERSION
	$(PYTHON) setup.py sdist bdist_wheel

# The clean target removes any build binaries
clean: ### Removes build binaries
	rm -rf build dist

# The install target installs the project from the source distribution
install: ### Installs required dependencies
	$(PIP) install dist/ydata-synthetic-$(version).tar.gz

# The publish-docs target publishes the project documentation
publish-docs: ### Publishes the documentation
	echo "$(version)" > VERSION
	$(PIP) install .
	mike deploy --push --update-aliases $(version) latest


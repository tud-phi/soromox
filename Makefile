#* Variables
PYTHON := python3
PYTHONPATH := `pwd`
#* Formatters
.PHONY: format
format:
	ruff --version
	ruff format --config pyproject.toml examples src tests

.PHONY: format-check
format-check:
	ruff --version
	ruff format --diff --check --config pyproject.toml examples src tests

.PHONY: flake8
flake8:
	flake8 --version
	flake8 src tests

.PHONY: pre-commit-install
pre-commit-install:
	pre-commit install

.PHONY: test
test:
	pytest

.PHONY: coverage
coverage:
	coverage run -m pytest
	coverage report

.PHONY: test_coverage
test_coverage: coverage

.PHONY: coverage_xml
coverage_xml:
	coverage run -m pytest
	coverage xml

.PHONY: test_coverage_xml
test_coverage_xml: coverage_xml

#* Cleaning
.PHONY: pycache-remove
pycache-remove:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf

.PHONY: dsstore-remove
dsstore-remove:
	find . | grep -E ".DS_Store" | xargs rm -rf

.PHONY: ipynbcheckpoints-remove
ipynbcheckpoints-remove:
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf

.PHONY: pytestcache-remove
pytestcache-remove:
	find . | grep -E ".pytest_cache" | xargs rm -rf

.PHONY: build-remove
build-remove:
	rm -rf build/

#* Documentation
.PHONY: docs-serve
docs-serve:
	uv run --extra docs zensical serve

.PHONY: docs-build
docs-build:
	uv run --extra docs zensical build --clean

.PHONY: docs-build-strict
docs-build-strict:
	@echo "Zensical does not currently support strict mode; running the normal docs build."
	uv run --extra docs zensical build --clean

.PHONY: cleanup
cleanup: pycache-remove dsstore-remove ipynbcheckpoints-remove pytestcache-remove

#* Version management
.PHONY: bump-patch
bump-patch:
	$(PYTHON) bump_version.py --patch --yes

.PHONY: bump-minor
bump-minor:
	$(PYTHON) bump_version.py --minor --yes

.PHONY: bump-major
bump-major:
	$(PYTHON) bump_version.py --major --yes

.PHONY: bump-version
bump-version:
	@echo "Usage: make bump-version VERSION=x.y.z"
	@if [ -z "$(VERSION)" ]; then \
		echo "Error: VERSION is required. Example: make bump-version VERSION=0.2.0"; \
		exit 1; \
	fi
	$(PYTHON) bump_version.py $(VERSION) --yes

.PHONY: release-patch
release-patch:
	$(PYTHON) bump_version.py --patch --yes --create-tag --push

.PHONY: release-minor
release-minor:
	$(PYTHON) bump_version.py --minor --yes --create-tag --push

.PHONY: release-major
release-major:
	$(PYTHON) bump_version.py --major --yes --create-tag --push

.PHONY: release
release:
	@echo "Usage: make release VERSION=x.y.z"
	@if [ -z "$(VERSION)" ]; then \
		echo "Error: VERSION is required. Example: make release VERSION=0.2.0"; \
		exit 1; \
	fi
	$(PYTHON) bump_version.py $(VERSION) --yes --create-tag --push

all: format-codestyle cleanup test

ci: check-codestyle

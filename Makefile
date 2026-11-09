.PHONY: install ingest test lint eval samples fetch full package clean
install:
	python -m pip install -e ".[dev]"
ingest:
	rca ingest -p offline
test:
	pytest --cov=finserv_rca
lint:
	ruff check src tests scripts
eval: ingest
	rca evaluate -p offline
samples: ingest
	python scripts/make_samples.py offline
fetch:
	python scripts/fetch_official.py
full:
	bash scripts/run_full.sh
package:
	python scripts/package.py
clean:
	rm -rf .rca

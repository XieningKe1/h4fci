PY ?= python3
BASIS ?= cc-pVTZ
THREADS ?= 20

.PHONY: run test bench clean

run:                       ## run the default calculation
	$(PY) -m h4fci --basis $(BASIS) --threads $(THREADS) --nroots 3 \
		--json results/results_$(BASIS).json

test:                      ## integral + SCF validation (a few seconds)
	$(PY) tests/test_integrals.py

bench:                     ## stage-by-stage timing -> docs/timing.md
	$(PY) tests/benchmark.py --threads $(THREADS)

clean:                     ## remove caches and build artefacts
	rm -rf cache __pycache__ h4fci/__pycache__ tests/__pycache__ \
	       build dist *.egg-info

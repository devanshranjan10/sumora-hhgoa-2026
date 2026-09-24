PYTHON ?= python3

.PHONY: prep verify answers answer-facts test ui-build

prep:
	$(PYTHON) scripts/prep_load_data.py

verify:
	bash gsql/connect_and_verify.sh verify

answers:
	$(PYTHON) -m bench.run --cases data/HHGOA_IEEE/case_pack.csv --out /tmp/sumora-answers --graph-agent --live

answer-facts:
	$(PYTHON) scripts/check_answer_facts.py

test:
	$(PYTHON) -m pytest bench/tests actions_api/tests -q

ui-build:
	cd ui && npm run build

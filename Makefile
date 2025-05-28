.PHONY: run_app run_advanced_app

poetry: poetry.lock install

poetry.lock: pyproject.toml
	(poetry lock)

install: poetry.lock
	(poetry install)

run_app: poetry
	streamlit run app.py --global.disableWidgetStateDuplicationWarning=True

run_advanced_app: poetry
	streamlit run app.py -- --app_version advanced_app

build_stlite: poetry
	python -m build
	cp dist/metapop-0.4.0-*-none-any.whl stlite
	cp stlite/*.whl .
	cp stlite/launch_stlite.py .
	zip -r stlite/metapop_stlite.zip launch_stlite.py *.whl scripts/app
	rm *.whl
	rm launch_stlite.py
	python3 stlite/make_measles_sim_html.py

rsconnect_requirements: poetry
	requirements
	manifest

requirements:
	pip freeze > tmp_requirements.txt
	grep -v "metapop" tmp_requirements.txt > requirements.txt
	metapop_version=$$(poetry version -s); \
	echo "metapop==$$metapop_version" >> requirements.txt
	rm tmp_requirements.txt

manifest:
	LANG=en_US.UTF-8 rsconnect write-manifest streamlit . --overwrite --exclude Makefile --exclude README.md

release: requirements manifest

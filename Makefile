.PHONY: run_app run_advanced_app

poetry: poetry.lock install

poetry.lock: pyproject.toml
	(poetry lock)

install: poetry.lock
	(poetry install)

run_app: poetry
	streamlit run metapop/launch_app.py --global.disableWidgetStateDuplicationWarning=True

run_advanced_app: poetry
	streamlit run metapop/launch_app.py -- --app_version advanced_app

build_stlite: poetry
	python -m build
	cp dist/metapop-0.4.0-*-none-any.whl stlite
	cp stlite/*.whl .
	cp stlite/launch_stlite.py .
	zip -r stlite/metapop_stlite.zip launch_stlite.py *.whl scripts/app
	rm *.whl
	rm launch_stlite.py
	python3 stlite/make_measles_sim_html.py

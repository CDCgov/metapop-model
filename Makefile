.PHONY: run_app run_advanced_app

poetry: poetry.lock install

poetry.lock: pyproject.toml
	(poetry lock --no-update)

install: poetry.lock
	(poetry install)

run_app: poetry
	streamlit run metapop/launch_app.py --global.disableWidgetStateDuplicationWarning=True

run_advanced_app: poetry
	streamlit run metapop/launch_app.py -- --app_version advanced_app

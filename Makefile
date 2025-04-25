.PHONY: run_app run_advanced_app

run_app:
	streamlit run metapop/launch_app.py --global.disableWidgetStateDuplicationWarning=True

run_advanced_app:
	streamlit run metapop/launch_app.py -- --app_version advanced_app

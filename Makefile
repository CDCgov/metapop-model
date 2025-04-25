.PHONY: run

run:
	streamlit run metapop/launch_app.py --global.disableWidgetStateDuplicationWarning=True

run_advanced:
	streamlit run metapop/launch_app.py -- --app_version advanced_app

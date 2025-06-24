"""
Do not edit the name of this file or change its location in the repository.
See below for more information.

This file defines and launches metapop apps. It serves as the entry point for the
outbreak simulator application and allows users to select different versions of
it based on their needs. This includes an advanced app, a one-population app,
and an app with a table for parameter input. Users running the app locally can
launch it with this file or use the instructions set up in the Makefile.

# ⚠️ Important Note about File Name and Location ⚠️

This file is named `app.py` and must be located in the root of the repository.

This file must live in this location in order for the python-rsconnect package
to find it as deployable content when deploying the app to Posit Connect
(aka RConnect). This file must also be named `app.py` in order to be recognized
by python-rsconnect for deployment. Other names or locatios will not work.
"""

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch the metapop app.")
    parser.add_argument(
        "--app_version",
        type=str,
        choices=["advanced_app", "one_pop_app"],
        default="one_pop_app",
        help="Specify the app version to launch. Defaults to 'one_pop_app'.",
    )
    args = parser.parse_args()
    app_version = args.app_version

    if app_version == "advanced_app":
        import metapop.advanced_app as app

    if app_version == "one_pop_app":
        import metapop.app as app

    # Launch the app
    app()

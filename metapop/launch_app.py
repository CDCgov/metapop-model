# Define and launch the metapop app

if __name__ == "__main__":

    # define the app version to launch for users
    app_version = "advanced_app"
    # app_version = "simple_app"
    # app_version = "app_with_parameter_table"

    if app_version == "advanced_app":
        from metapop import app as app

    if app_version == "simple_app":
        from metapop import app_simple as app

    if app_version == "app_with_parameter_table":
        from metapop import app_with_parameter_table as app

    # Launch the app
    app()

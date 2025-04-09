# Define and launch the metapop app

if __name__ == "__main__":

    # define the app version to launch for users
    # app_version = "advanced_app"
    # app_version = "one_pop_app"
    app_version = "app_with_parameter_table"

    if app_version == "advanced_app":
        import metapop.advanced_app as app

    if app_version == "one_pop_app":
        import metapop.app as app

    if app_version == "app_with_parameter_table":
        import metapop.app_with_table as app

    # Launch the app
    app()


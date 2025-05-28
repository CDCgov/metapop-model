# Define and launch the metapop app

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch the metapop app.")
    parser.add_argument(
        "--app_version",
        type=str,
        choices=["advanced_app", "one_pop_app", "app_with_table"],
        default="one_pop_app",
        help="Specify the app version to launch. Defaults to 'one_pop_app'.",
    )
    args = parser.parse_args()
    app_version = args.app_version

    if app_version == "advanced_app":
        import metapop.advanced_app as app

    if app_version == "one_pop_app":
        import metapop.app as app

    if app_version == "app_with_table":
        import metapop.app_with_table as app

    # Launch the app
    app()

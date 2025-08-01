import streamlit as st

from metapop import advanced_app, app


def test_app():
    """Test that the app runs without errors."""
    # assert that the app is callable
    assert callable(app), "app is not callable"

    # Assert that app() runs without raising an error
    try:
        # launch the app
        app()
    except Exception as e:
        print(f"Error occurred while running app: {e}")
        raise RuntimeError("An error occurred while running the app.") from e
    st.stop()  # stop the app


def test_app_accepts_config_file():
    """Test that the app accepts a config file as an argument."""
    # assert that the app is callable
    assert callable(app), "app is not callable"

    # Assert that app() runs without raising an error with a config file
    try:
        app(config_file="tests/test_one_pop_config.yaml")
    except Exception as e:
        print(f"Error occurred while running app with config file: {e}")
        raise RuntimeError(
            "An error occurred while running the app with a config file."
        ) from e
    st.stop()  # stop the app


def test_advanced_app():
    """Test that the advanced app with 3 populations runs without errors."""
    # assert that the advanced app is callable
    assert callable(advanced_app), "advanced_app is not callable"

    # Assert that advanced_app() runs without raising an error
    try:
        # launch the advanced app
        advanced_app()
    except Exception as e:
        print(f"Error occurred while running advanced_app: {e}")
        raise RuntimeError("An error occurred while running the advanced app.") from e
    st.stop()  # stop the app


def test_advanced_app_accepts_config_file():
    """Test that the advanced app accepts a config file as an argument."""
    # assert that the advanced app is callable
    assert callable(advanced_app), "advanced_app is not callable"

    # Assert that advanced_app() runs without raising an error with a config file
    try:
        advanced_app(config_file="tests/test_app_config.yaml")
    except Exception as e:
        print(f"Error occurred while running advanced_app with config file: {e}")
        raise RuntimeError(
            "An error occurred while running the advanced app with a config file."
        ) from e
    st.stop()  # stop the app


if __name__ == "__main__":
    # Run the tests
    test_app()
    test_advanced_app()

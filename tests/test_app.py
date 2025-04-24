import streamlit as st

from metapop import advanced_app, app, app_with_table


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


def test_app_with_table():
    """Test that the app with table runs without errors."""
    # assert that the app_with_table is callable
    assert callable(app_with_table), "app_with_table is not callable"

    # Assert that app_with_table() runs without raising an error
    try:
        # launch the app with table
        app_with_table()
    except Exception as e:
        print(f"Error occurred while running app_with_table: {e}")
        raise RuntimeError("An error occurred while running the app with table.") from e
    st.stop()  # stop the app


if __name__ == "__main__":
    # Run the tests
    test_app()
    test_advanced_app()
    test_app_with_table()

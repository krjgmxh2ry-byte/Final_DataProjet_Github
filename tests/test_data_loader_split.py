from src.data_loader import load_and_split


def test_load_and_split_returns_non_empty_sets():
    """
    Integration-style unit test for data loading and splitting.

    Goal:
    - Call load_and_split()
    - Verify that train/test sets and labels are non-empty and consistent.
    """

    X_train, X_test, y_train, y_test, df = load_and_split()

    # Basic non-emptiness checks
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0

    # Shapes must be compatible
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    # There should be at least one feature column
    assert X_train.shape[1] > 0
    assert X_test.shape[1] > 0
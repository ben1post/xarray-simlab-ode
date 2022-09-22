import xso

def test_variable():
    """Test variable function."""

    expected = xso.variable()
    actual = xso.variable()
    assert actual == expected, "xso.variable is weird"
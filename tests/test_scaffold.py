"""Smoke test: the package imports. Real tests arrive with each module."""


def test_import_pxr():
    import pxr

    assert pxr.__version__

import rllib_async


def test_package_is_importable_from_editable_install() -> None:
    assert rllib_async.__version__ == "0.0.1"

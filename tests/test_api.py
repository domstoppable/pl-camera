import pupil_labs.camera as plc


def test_package_metadata() -> None:
    assert hasattr(plc, "__version__")

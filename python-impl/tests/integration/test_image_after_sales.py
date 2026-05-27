import pytest

from smart_cs.infrastructure.assets import LocalAssetStorage


def test_image_is_stored_in_conversation_directory(tmp_path) -> None:
    storage = LocalAssetStorage(tmp_path / "assets")

    key = storage.save("conv-1", "damage.jpg", "image/jpeg", b"jpeg-data")

    assert key.startswith("conv-1/")
    assert key.endswith("-damage.jpg")
    assert (tmp_path / "assets" / key).read_bytes() == b"jpeg-data"


def test_non_image_evidence_is_rejected(tmp_path) -> None:
    storage = LocalAssetStorage(tmp_path / "assets")

    with pytest.raises(ValueError, match="JPEG and PNG"):
        storage.save("conv-1", "policy.md", "text/markdown", b"not-image")

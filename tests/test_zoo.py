import hashlib

import pytest

from septosympto import zoo


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("SEPTOSYMPTO_HOME", str(tmp_path))
    return tmp_path


def test_cache_dir_follows_the_env_override(home):
    assert zoo.cache_dir() == home


def test_a_published_name_resolves_to_a_verified_cached_file(home, monkeypatch):
    card = zoo.NECROSIS[zoo.DEFAULT_NECROSIS]
    payload = b"weights"
    monkeypatch.setattr(zoo, "NECROSIS", {**zoo.NECROSIS, card.name: zoo.ModelCard(
        **{**card.__dict__, "sha256": hashlib.sha256(payload).hexdigest()})})
    monkeypatch.setitem(zoo.CATALOGUE, "necrosis", zoo.NECROSIS)
    (home / "models").mkdir()
    (home / "models" / card.file).write_bytes(payload)

    resolved = zoo.resolve(card.name, "necrosis")
    assert resolved.path == home / "models" / card.file
    assert (resolved.kind, resolved.arch, resolved.threshold) == ("yolo", None, 0.3)


def test_a_corrupt_cached_file_is_refetched_and_a_bad_download_refused(home, monkeypatch):
    card = zoo.NECROSIS["unet-v1"]
    (home / "models").mkdir()
    (home / "models" / card.file).write_bytes(b"corrupt")

    def fake_download(url, target):
        assert url == card.url
        target.write_bytes(b"still wrong")

    monkeypatch.setattr(zoo.urllib.request, "urlretrieve",
                        lambda url, filename: fake_download(url, zoo.Path(filename)))
    with pytest.raises(RuntimeError, match="refusing to use it"):
        zoo.fetch(card, quiet=True)
    assert not (home / "models" / card.file).exists()
    assert not list((home / "models").glob("*.part"))


def test_paths_resolve_by_suffix_and_torch_needs_an_arch(tmp_path):
    pt = tmp_path / "best.pt"
    pt.write_bytes(b"x")
    st = tmp_path / "best.safetensors"
    st.write_bytes(b"x")

    assert zoo.resolve(str(pt), "necrosis").kind == "yolo"
    assert zoo.resolve(str(pt), "necrosis").threshold is None
    with pytest.raises(ValueError, match="--necrosis-arch"):
        zoo.resolve(str(st), "necrosis")
    assert zoo.resolve(str(st), "necrosis", arch="unet").arch == "unet"
    with pytest.raises(ValueError, match="unknown necrosis model"):
        zoo.resolve("nope", "necrosis")


def test_catalogue_entries_are_consistent():
    for task, table in zoo.CATALOGUE.items():
        for name, card in table.items():
            assert card.name == name and card.task == task
            assert len(card.sha256) == 64
            assert (card.kind == "yolo") == card.file.endswith(".pt")
            assert (card.kind == "torch") == (card.arch is not None)
    assert zoo.DEFAULT_NECROSIS in zoo.NECROSIS
    assert "necrosis" in zoo.describe() and zoo.DEFAULT_NECROSIS in zoo.describe()

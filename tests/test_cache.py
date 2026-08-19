from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from pxr.data.cache import (
    CACHE_VERSION,
    CacheError,
    build_cache,
    cohort_fingerprint,
    load_cache,
    read_images,
    verify_cache,
)

CHEXPERT_PATTERN = r"patient(\d+)[/_]study(\d+)[/_](view\d+_[a-zA-Z]+)"


def write_zip(path, names, size=224, mode="L", seed=0):
    rng = np.random.default_rng(seed)
    with zipfile.ZipFile(path, "w") as z:
        for name in names:
            shape = (size, size) if mode == "L" else (size, size, 3)
            pixels = rng.integers(0, 256, shape, dtype=np.uint8)
            buffer = io.BytesIO()
            Image.fromarray(pixels, mode=mode).save(buffer, format="PNG")
            z.writestr(name, buffer.getvalue())
    return path


class TestIdentifierMapping:
    """The mapping must be one-to-one, or the cache is scientifically useless."""

    def test_chexpert_identifiers_collide_on_the_filename_alone(self, tmp_path):
        """Every CheXpert image is named view1_frontal: depth 1 collapses the cohort."""
        cohort = [f"train/patient{i:05d}/study1/view1_frontal.jpg" for i in range(5)]
        write_zip(tmp_path / "a.zip", [f"train_patient{i:05d}_study1_view1_frontal.png"
                                       for i in range(5)])
        with pytest.raises(CacheError, match="match more than one cohort identifier"):
            build_cache(tmp_path / "a.zip", cohort, tmp_path / "cache", key_depth=1)

    def test_chexpert_pattern_resolves_the_collision(self, tmp_path):
        """With the site's own pattern the same cohort maps cleanly."""
        cohort = [f"train/patient{i:05d}/study1/view1_frontal.jpg" for i in range(5)]
        write_zip(tmp_path / "a.zip", [f"train_patient{i:05d}_study1_view1_frontal.png"
                                       for i in range(5)])
        index = build_cache(tmp_path / "a.zip", cohort, tmp_path / "cache",
                            key_pattern=CHEXPERT_PATTERN)
        assert len(index) == 5
        assert set(index.frame["image_id"]) == set(cohort)

    def test_duplicate_archive_entries_raise(self, tmp_path):
        """Two archive files matching one cohort image is ambiguous, not a preference."""
        write_zip(tmp_path / "a.zip", ["dir1/img1.png", "dir2/img1.png"])
        with pytest.raises(CacheError, match="more than one archive entry"):
            build_cache(tmp_path / "a.zip", ["img1"], tmp_path / "cache", key_depth=1)

    def test_unparseable_names_raise(self, tmp_path):
        write_zip(tmp_path / "a.zip", ["img1.png"])
        with pytest.raises(CacheError, match="canonical key"):
            build_cache(tmp_path / "a.zip", ["nonsense"], tmp_path / "cache",
                        key_pattern=CHEXPERT_PATTERN)

    def test_duplicate_cohort_identifiers_raise(self, tmp_path):
        write_zip(tmp_path / "a.zip", ["img1.png"])
        with pytest.raises(CacheError, match="duplicates"):
            build_cache(tmp_path / "a.zip", ["img1", "img1"], tmp_path / "cache")

    def test_extension_and_path_differences_are_absorbed(self, tmp_path):
        write_zip(tmp_path / "a.zip", ["sub/dir/img1.png"])
        index = build_cache(tmp_path / "a.zip", ["img1.jpg"], tmp_path / "cache")
        assert index.frame.iloc[0]["image_id"] == "img1.jpg"


class TestBuild:
    def test_packs_every_requested_image(self, tmp_path):
        names = [f"img{i}.png" for i in range(50)]
        write_zip(tmp_path / "a.zip", names)
        index = build_cache(tmp_path / "a.zip", [n[:-4] for n in names],
                            tmp_path / "cache", shard_size=20)
        assert len(index) == 50
        assert sorted(index.frame["shard"].unique()) == [0, 1, 2]

    def test_only_requested_images_are_packed(self, tmp_path):
        write_zip(tmp_path / "a.zip", [f"img{i}.png" for i in range(50)])
        index = build_cache(tmp_path / "a.zip", ["img1", "img7"], tmp_path / "cache")
        assert len(index) == 2

    def test_rebuild_is_deterministic(self, tmp_path):
        names = [f"img{i}.png" for i in range(30)]
        write_zip(tmp_path / "a.zip", names)
        ids = [n[:-4] for n in names]
        a = build_cache(tmp_path / "a.zip", ids, tmp_path / "c1", shard_size=10)
        b = build_cache(tmp_path / "a.zip", ids, tmp_path / "c2", shard_size=10)
        pd.testing.assert_frame_equal(a.frame, b.frame)

    def test_missing_images_raise(self, tmp_path):
        write_zip(tmp_path / "a.zip", ["img1.png"])
        with pytest.raises(CacheError, match="absent from"):
            build_cache(tmp_path / "a.zip", ["img1", "ghost"], tmp_path / "cache")

    def test_empty_request_raises(self, tmp_path):
        write_zip(tmp_path / "a.zip", ["img1.png"])
        with pytest.raises(CacheError, match="no images requested"):
            build_cache(tmp_path / "a.zip", [], tmp_path / "cache")


class TestRefusesToAlterImages:
    def test_wrong_size_raises_rather_than_resizing(self, tmp_path):
        write_zip(tmp_path / "a.zip", ["img1.png"], size=128)
        with pytest.raises(CacheError, match="expected 224x224"):
            build_cache(tmp_path / "a.zip", ["img1"], tmp_path / "cache")

    def test_colour_image_raises_rather_than_converting(self, tmp_path):
        write_zip(tmp_path / "a.zip", ["img1.png"], mode="RGB")
        with pytest.raises(CacheError, match="single-channel"):
            build_cache(tmp_path / "a.zip", ["img1"], tmp_path / "cache")


class TestAtomicity:
    """An interrupted build must leave no cache, not a partial one."""

    def test_failed_build_leaves_no_cache_directory(self, tmp_path):
        write_zip(tmp_path / "a.zip", ["img1.png", "img2.png"], size=128)
        out = tmp_path / "cache"
        with pytest.raises(CacheError):
            build_cache(tmp_path / "a.zip", ["img1", "img2"], out)
        assert not out.exists()

    def test_failed_build_leaves_no_staging_directory(self, tmp_path):
        write_zip(tmp_path / "a.zip", ["img1.png"], size=128)
        out = tmp_path / "cache"
        with pytest.raises(CacheError):
            build_cache(tmp_path / "a.zip", ["img1"], out)
        assert not list(tmp_path.glob(".*building"))

    def test_failed_rebuild_does_not_destroy_the_existing_cache(self, tmp_path):
        write_zip(tmp_path / "good.zip", ["img1.png"])
        out = tmp_path / "cache"
        build_cache(tmp_path / "good.zip", ["img1"], out)
        write_zip(tmp_path / "bad.zip", ["img1.png"], size=128)
        with pytest.raises(CacheError):
            build_cache(tmp_path / "bad.zip", ["img1"], out)
        assert load_cache(out) is not None


class TestManifest:
    def _built(self, tmp_path, ids=("img1", "img2")):
        write_zip(tmp_path / "a.zip", [f"{i}.png" for i in ids])
        return build_cache(tmp_path / "a.zip", list(ids), tmp_path / "cache",
                           site="mimic-cxr", config_hash="abc123def456")

    def test_manifest_records_build_properties(self, tmp_path):
        index = self._built(tmp_path)
        for key in ("cache_version", "site", "config_hash", "image_size", "dtype",
                    "n_images", "n_shards", "cohort_fingerprint", "built_utc"):
            assert key in index.manifest

    def test_cohort_fingerprint_is_order_independent(self):
        assert cohort_fingerprint(["a", "b", "c"]) == cohort_fingerprint(["c", "a", "b"])

    def test_different_cohorts_have_different_fingerprints(self):
        assert cohort_fingerprint(["a", "b"]) != cohort_fingerprint(["a", "c"])

    def test_loading_with_a_different_cohort_raises(self, tmp_path):
        """Stale pixels must not be paired with a rebuilt cohort."""
        self._built(tmp_path)
        with pytest.raises(CacheError, match="different cohort"):
            load_cache(tmp_path / "cache", cohort_ids=["img1", "img9"])

    def test_loading_with_a_different_config_raises(self, tmp_path):
        self._built(tmp_path)
        with pytest.raises(CacheError, match="different config|config"):
            load_cache(tmp_path / "cache", config_hash="999999999999")

    def test_loading_with_the_matching_cohort_succeeds(self, tmp_path):
        self._built(tmp_path)
        index = load_cache(tmp_path / "cache", cohort_ids=["img1", "img2"],
                           config_hash="abc123def456")
        assert len(index) == 2

    def test_stale_cache_version_raises(self, tmp_path):
        self._built(tmp_path)
        path = tmp_path / "cache" / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["cache_version"] = CACHE_VERSION + 1
        path.write_text(json.dumps(manifest))
        with pytest.raises(CacheError, match="cache version"):
            load_cache(tmp_path / "cache")

    def test_incomplete_cache_raises(self, tmp_path):
        self._built(tmp_path)
        (tmp_path / "cache" / "manifest.json").unlink()
        with pytest.raises(CacheError, match="not a complete cache"):
            load_cache(tmp_path / "cache")


class TestRead:
    def test_pixels_survive_the_round_trip(self, tmp_path):
        rng = np.random.default_rng(1)
        original = rng.integers(0, 256, (224, 224), dtype=np.uint8)
        with zipfile.ZipFile(tmp_path / "a.zip", "w") as z:
            buffer = io.BytesIO()
            Image.fromarray(original, mode="L").save(buffer, format="PNG")
            z.writestr("img1.png", buffer.getvalue())
        index = build_cache(tmp_path / "a.zip", ["img1"], tmp_path / "cache")
        assert np.array_equal(read_images(index, ["img1"])[0], original)

    def test_returns_images_in_the_requested_order(self, tmp_path):
        names = [f"img{i}.png" for i in range(20)]
        write_zip(tmp_path / "a.zip", names)
        index = build_cache(tmp_path / "a.zip", [n[:-4] for n in names],
                            tmp_path / "cache", shard_size=5)
        wanted = ["img7", "img2", "img19"]
        got = read_images(index, wanted)
        for i, name in enumerate(wanted):
            assert np.array_equal(got[i], read_images(index, [name])[0])

    def test_reads_across_shard_boundaries(self, tmp_path):
        names = [f"img{i}.png" for i in range(30)]
        write_zip(tmp_path / "a.zip", names)
        index = build_cache(tmp_path / "a.zip", [n[:-4] for n in names],
                            tmp_path / "cache", shard_size=10)
        assert read_images(index, ["img1", "img15", "img28"]).shape == (3, 224, 224)

    def test_unknown_identifier_raises(self, tmp_path):
        write_zip(tmp_path / "a.zip", ["img1.png"])
        index = build_cache(tmp_path / "a.zip", ["img1"], tmp_path / "cache")
        with pytest.raises(CacheError, match="not in the cache"):
            read_images(index, ["ghost"])


class TestVerify:
    def _cache(self, tmp_path, n=40):
        names = [f"img{i}.png" for i in range(n)]
        write_zip(tmp_path / "a.zip", names)
        return build_cache(tmp_path / "a.zip", [x[:-4] for x in names],
                           tmp_path / "cache", shard_size=15)

    def test_healthy_cache_passes(self, tmp_path):
        index = self._cache(tmp_path)
        report = verify_cache(index, sample=20,
                              cohort_ids=[f"img{i}" for i in range(40)])
        assert report.ok.all(), report.to_string(index=False)

    def test_corrupted_pixels_are_detected(self, tmp_path):
        index = self._cache(tmp_path)
        shard = np.load(index.shard_path(0))
        shard[0] = 0
        np.save(index.shard_path(0), shard)
        report = verify_cache(index, sample=40)
        assert not report[report.check == "pixels_match_checksums"].iloc[0].ok

    def test_missing_shard_is_detected(self, tmp_path):
        index = self._cache(tmp_path)
        index.shard_path(1).unlink()
        report = verify_cache(index, sample=10)
        assert not report[report.check == "shards_present"].iloc[0].ok

    def test_cohort_mismatch_is_detected(self, tmp_path):
        index = self._cache(tmp_path, n=20)
        report = verify_cache(index, sample=5,
                              cohort_ids=[f"img{i}" for i in range(25)])
        assert not report[report.check == "cohort_fingerprint"].iloc[0].ok
        assert not report[report.check == "covers_cohort"].iloc[0].ok

    def test_extra_images_beyond_the_cohort_are_flagged(self, tmp_path):
        index = self._cache(tmp_path, n=20)
        report = verify_cache(index, sample=5,
                              cohort_ids=[f"img{i}" for i in range(10)])
        assert not report[report.check == "no_extra_images"].iloc[0].ok

    def test_duplicate_slots_are_detected(self, tmp_path):
        index = self._cache(tmp_path, n=20)
        index.frame.loc[1, ["shard", "row"]] = index.frame.loc[0, ["shard", "row"]].values
        report = verify_cache(index, sample=5)
        assert not report[report.check == "positions_unique"].iloc[0].ok
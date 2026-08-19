from __future__ import annotations

import hashlib
import io
import json
import shutil
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pxr.data.cohort import canonical_image_key

__all__ = [
    "CacheError",
    "CacheIndex",
    "CACHE_VERSION",
    "DEFAULT_SHARD_SIZE",
    "build_cache",
    "cohort_fingerprint",
    "load_cache",
    "read_images",
    "verify_cache",
]

#: Bumped when the on-disk layout changes, so an old cache is rejected rather than
#: misread by newer code.
CACHE_VERSION = 1

#: Images per shard. Large enough that per-file overhead is negligible, small enough
#: that a partial failure loses one shard rather than the whole cache.
DEFAULT_SHARD_SIZE = 8192


class CacheError(RuntimeError):
    """Raised when a cache cannot be built or fails its integrity checks."""


def cohort_fingerprint(image_ids: Iterable[str]) -> str:
    """Order-independent digest of a cohort's image identifiers.

    Lets a cache state which population it was built for, so a cohort rebuilt under a
    different configuration cannot be silently paired with stale pixels.
    """
    joined = "\n".join(sorted(str(i) for i in image_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


@dataclass
class CacheIndex:
    """Where each image lives in the shards, and what it should contain.

    Attributes
    ----------
    frame
        One row per image: ``image_id``, ``shard``, ``row``, ``checksum``.
    directory
        Where the shards, index, and manifest sit.
    manifest
        Build properties: shape, dtype, counts, cohort fingerprint, cache version.
    """

    frame: pd.DataFrame
    directory: Path
    manifest: dict

    @property
    def image_size(self) -> int:
        return int(self.manifest["image_size"])

    def shard_path(self, shard: int) -> Path:
        return self.directory / f"shard_{shard:04d}.npy"

    def __len__(self) -> int:
        return len(self.frame)


def _checksum(array: np.ndarray) -> str:
    """Short digest of the pixel bytes, for detecting corruption after the build."""
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


def _keys(values: Iterable[str], depth: int, pattern: str | None) -> pd.Series:
    return canonical_image_key(pd.Series([str(v) for v in values], dtype="string"),
                               depth, pattern)


def _require_unique(keys: pd.Series, values: list[str], what: str) -> dict[str, str]:
    """Map canonical key to original identifier, refusing any collision."""
    frame = pd.DataFrame({"key": keys, "value": values})
    unresolved = frame[frame["key"].isna()]
    if len(unresolved):
        raise CacheError(
            f"{len(unresolved):,} {what} could not be reduced to a canonical key "
            f"(e.g. {unresolved['value'].head(3).tolist()}); the site's "
            "image_key_pattern does not fit these names"
        )
    counts = frame.groupby("key")["value"].nunique()
    collided = counts[counts > 1]
    if len(collided):
        examples = frame[frame["key"].isin(collided.index[:2])]
        raise CacheError(
            f"{len(collided):,} canonical key(s) match more than one {what}, so the "
            f"mapping is ambiguous (e.g. {examples['value'].head(4).tolist()}). "
            "Increase image_key_depth or set image_key_pattern for this site - the "
            "cohort-to-pixel mapping must be one-to-one."
        )
    return dict(zip(frame["key"], frame["value"], strict=True))


def build_cache(
    zip_path: str | Path,
    wanted: Iterable[str],
    out_dir: str | Path,
    *,
    image_size: int = 224,
    shard_size: int = DEFAULT_SHARD_SIZE,
    key_depth: int = 1,
    key_pattern: str | None = None,
    site: str | None = None,
    config_hash: str | None = None,
) -> CacheIndex:
    """Pack the wanted images from an archive into memmap shards.

    Parameters
    ----------
    zip_path
        Archive holding the site's images.
    wanted
        Image identifiers from the cohort. Only these are packed: the cache mirrors
        the analysis population, not the archive.
    key_depth, key_pattern
        Passed to :func:`pxr.data.cohort.canonical_image_key`, from the site's
        configuration. These must be the site's own values - CheXpert identifiers
        all share the stem ``view1_frontal``, so matching on the filename alone
        collapses the whole cohort onto one key.
    site, config_hash
        Recorded in the manifest so the cache can state what it belongs to.
    image_size
        Expected side length. Any other size raises rather than being resized: the
        cohort images are uniform, so a mismatch means the wrong archive or a corrupt
        file, and resizing would hide that.

    Returns
    -------
    CacheIndex

    Raises
    ------
    CacheError
        On ambiguous identifier mappings, images absent from the archive, or images
        of unexpected shape.

    Notes
    -----
    The build is atomic: shards are written to a sibling temporary directory and
    promoted only once every shard, the index, and the manifest are complete. An
    interrupted run therefore leaves no cache rather than a partial one that later
    looks finished.
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise CacheError("Pillow is required to build an image cache") from exc

    zip_path, out_dir = Path(zip_path), Path(out_dir)
    if not zip_path.exists():
        raise CacheError(f"archive not found: {zip_path}")

    target = [str(w) for w in wanted]
    if not target:
        raise CacheError("no images requested; the cache would be empty")
    if len(set(target)) != len(target):
        raise CacheError("the requested identifiers contain duplicates")

    lookup = _require_unique(_keys(target, key_depth, key_pattern), target,
                             "cohort identifier(s)")

    with zipfile.ZipFile(zip_path) as archive:
        names = [n for n in archive.namelist() if not n.endswith("/")]
        archive_keys = _keys(names, key_depth, key_pattern)
        relevant = pd.DataFrame({"key": archive_keys, "name": names})
        relevant = relevant[relevant["key"].isin(lookup)]

        # One archive entry per cohort image, or the mapping is not one-to-one.
        per_key = relevant.groupby("key")["name"].nunique()
        ambiguous = per_key[per_key > 1]
        if len(ambiguous):
            examples = relevant[relevant["key"].isin(ambiguous.index[:2])]
            raise CacheError(
                f"{len(ambiguous):,} cohort image(s) match more than one archive entry "
                f"(e.g. {examples['name'].head(4).tolist()}); the archive contains "
                "duplicates under this key scheme"
            )

        missing = set(lookup) - set(relevant["key"])
        if missing:
            absent = [lookup[k] for k in sorted(missing)[:3]]
            raise CacheError(
                f"{len(missing):,} of {len(lookup):,} requested images are absent from "
                f"{zip_path.name} (e.g. {absent}); the cache would not cover the cohort"
            )

        # Stable order so a rebuild produces an identical cache.
        entries = relevant.sort_values("key")[["key", "name"]].to_records(index=False)

        staging = out_dir.parent / f".{out_dir.name}.building"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

        try:
            rows: list[dict] = []
            n_shards = 0
            for shard, start in enumerate(range(0, len(entries), shard_size)):
                block = entries[start:start + shard_size]
                buffer = np.zeros((len(block), image_size, image_size), dtype=np.uint8)

                for row, (key, name) in enumerate(block):
                    with archive.open(str(name)) as handle:
                        pixels = np.array(Image.open(io.BytesIO(handle.read())))
                    if pixels.ndim != 2:
                        raise CacheError(
                            f"{name}: expected a single-channel image, got shape "
                            f"{pixels.shape}. Images are stored unchanged; convert "
                            "upstream if needed."
                        )
                    if pixels.shape != (image_size, image_size):
                        raise CacheError(
                            f"{name}: expected {image_size}x{image_size}, got "
                            f"{pixels.shape}. Images are stored as-is rather than "
                            "resized, so a mismatch means the wrong archive or a "
                            "corrupt file."
                        )
                    buffer[row] = pixels
                    rows.append({"image_id": lookup[str(key)], "shard": shard,
                                 "row": row, "checksum": _checksum(pixels)})

                np.save(staging / f"shard_{shard:04d}.npy", buffer)
                n_shards = shard + 1

            frame = pd.DataFrame(rows)
            manifest = {
                "cache_version": CACHE_VERSION,
                "site": site,
                "config_hash": config_hash,
                "image_size": image_size,
                "dtype": "uint8",
                "channels": 1,
                "n_images": len(frame),
                "n_shards": n_shards,
                "shard_size": shard_size,
                "key_depth": key_depth,
                "key_pattern": key_pattern,
                "cohort_fingerprint": cohort_fingerprint(target),
                "source_archive": zip_path.name,
                "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            frame.to_parquet(staging / "index.parquet", index=False)
            (staging / "manifest.json").write_text(json.dumps(manifest, indent=2))

            # Promote only once everything is on disk.
            if out_dir.exists():
                shutil.rmtree(out_dir)
            staging.replace(out_dir)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    return CacheIndex(frame=frame, directory=out_dir, manifest=manifest)


def load_cache(
    directory: str | Path,
    *,
    cohort_ids: Iterable[str] | None = None,
    config_hash: str | None = None,
) -> CacheIndex:
    """Load a cache, checking it belongs to the cohort and configuration in hand.

    Parameters
    ----------
    cohort_ids
        When given, the cohort fingerprint must match - so a cohort rebuilt under a
        different configuration cannot be paired with stale pixels.
    config_hash
        When given, must match the hash recorded at build time.
    """
    directory = Path(directory)
    index_path, manifest_path = directory / "index.parquet", directory / "manifest.json"
    if not index_path.exists() or not manifest_path.exists():
        raise CacheError(
            f"{directory} is not a complete cache (index and manifest required); "
            "build it first"
        )

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("cache_version") != CACHE_VERSION:
        raise CacheError(
            f"cache version {manifest.get('cache_version')} does not match "
            f"{CACHE_VERSION}; rebuild the cache"
        )
    if config_hash is not None and manifest.get("config_hash") not in (None, config_hash):
        raise CacheError(
            f"cache was built under config {manifest.get('config_hash')} but "
            f"{config_hash} is in use; rebuild the cache"
        )
    if cohort_ids is not None:
        expected = cohort_fingerprint(cohort_ids)
        if manifest.get("cohort_fingerprint") != expected:
            raise CacheError(
                "cache was built for a different cohort "
                f"(fingerprint {manifest.get('cohort_fingerprint')}, expected {expected}); "
                "rebuild the cache"
            )

    return CacheIndex(frame=pd.read_parquet(index_path), directory=directory,
                      manifest=manifest)


def read_images(index: CacheIndex, image_ids: Iterable[str]) -> np.ndarray:
    """Read images by identifier, in the order requested.

    Shards are opened memory-mapped and grouped, so a batch spanning several shards
    still touches each file once.
    """
    wanted = pd.DataFrame({"image_id": [str(i) for i in image_ids]})
    located = wanted.merge(index.frame, on="image_id", how="left")
    absent = located[located["shard"].isna()]
    if len(absent):
        raise CacheError(
            f"{len(absent)} image(s) are not in the cache "
            f"(e.g. {absent['image_id'].head(3).tolist()})"
        )

    size = index.image_size
    out = np.zeros((len(located), size, size), dtype=np.uint8)
    for shard, block in located.groupby("shard"):
        data = np.load(index.shard_path(int(shard)), mmap_mode="r")
        out[block.index.to_numpy()] = data[block["row"].to_numpy().astype(int)]
    return out


def verify_cache(
    index: CacheIndex,
    *,
    sample: int = 500,
    seed: int = 42,
    cohort_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Check the cache against its manifest and re-hash a sample of its pixels.

    The checksums establish that stored pixels have not been corrupted since the
    build. They cannot establish that those pixels came from the right source image -
    only the identifier mapping does that, which is why an ambiguous mapping raises
    at build time rather than being resolved here.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    manifest = index.manifest

    rows.append({"check": "manifest_counts",
                 "value": f"{manifest['n_images']:,} images, {manifest['n_shards']} shards",
                 "ok": manifest["n_images"] == len(index.frame),
                 "note": "" if manifest["n_images"] == len(index.frame)
                         else f"index holds {len(index.frame):,}"})

    shards = sorted(index.frame["shard"].unique())
    missing_shards = [s for s in shards if not index.shard_path(int(s)).exists()]
    rows.append({"check": "shards_present",
                 "value": f"{len(shards) - len(missing_shards)}/{len(shards)}",
                 "ok": not missing_shards,
                 "note": f"absent: {missing_shards[:3]}" if missing_shards else ""})

    duplicated = int(index.frame["image_id"].duplicated().sum())
    rows.append({"check": "identifiers_unique", "value": f"{len(index.frame):,} images",
                 "ok": duplicated == 0,
                 "note": f"{duplicated} duplicate identifier(s)" if duplicated else ""})

    positions = index.frame.duplicated(subset=["shard", "row"]).sum()
    rows.append({"check": "positions_unique", "value": f"{len(index.frame):,} slots",
                 "ok": positions == 0,
                 "note": f"{positions} images share a slot" if positions else ""})

    if cohort_ids is not None:
        cohort = {str(i) for i in cohort_ids}
        cached = set(index.frame["image_id"])
        rows.append({"check": "cohort_fingerprint",
                     "value": manifest.get("cohort_fingerprint", "?"),
                     "ok": manifest.get("cohort_fingerprint") == cohort_fingerprint(cohort),
                     "note": "cache was built for a different cohort"
                             if manifest.get("cohort_fingerprint") != cohort_fingerprint(cohort)
                             else ""})
        rows.append({"check": "covers_cohort", "value": f"{len(cached):,} cached",
                     "ok": cohort <= cached,
                     "note": f"{len(cohort - cached):,} cohort image(s) not cached"
                             if cohort - cached else ""})
        extra = cached - cohort
        rows.append({"check": "no_extra_images", "value": f"{len(extra):,} extra",
                     "ok": not extra,
                     "note": "cache holds images outside the cohort" if extra else ""})

    if not missing_shards and len(index.frame):
        take = min(sample, len(index.frame))
        chosen = index.frame.iloc[rng.choice(len(index.frame), take, replace=False)]
        pixels = read_images(index, chosen["image_id"])
        recomputed = np.array([_checksum(pixels[i]) for i in range(len(chosen))])
        mismatched = int((recomputed != chosen["checksum"].to_numpy()).sum())
        rows.append({"check": "pixels_match_checksums", "value": f"{take:,} sampled",
                     "ok": mismatched == 0,
                     "note": f"{mismatched} image(s) differ from their recorded checksum"
                             if mismatched else ""})

    return pd.DataFrame(rows)
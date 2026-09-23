"""Smoke tests for PatternAnalysis' on-disk caching of pipeline intermediates.

Covers the two cached steps added for issue #32: `build_data()` (wrapping
`reader.make_data()`) and `calculate_validation_encodings()`. Uses the real
"data" folder (like test_data_reader.py) with a small `max_samples` and
`embedder=None` so no deep model is built, and points `cache_folder` at
a fresh `tmp_path` per test so caches never leak between tests or pollute the
real `cache/` directory.
"""

import os
import time

import pytest

torch = pytest.importorskip("torch")

from src.analysis.pattern_analysis import PatternAnalysis  # noqa: E402

SEQUENCE_TYPE = "first_5_seconds"
MAX_SAMPLES = 10
CACHE_HIT_TIME_LIMIT = 2.0  # seconds; a cache hit should be near-instant


def make_pattern_analysis(
    cache_folder,
    max_samples=MAX_SAMPLES,
    sequence_type=SEQUENCE_TYPE,
    augmented_visualization=False,
):
    return PatternAnalysis(
        data_folder="data",
        cache_folder=str(cache_folder),
        embedder=None,
        feature_set="Pacman",
        sequence_type=sequence_type,
        max_samples=max_samples,
        augmented_visualization=augmented_visualization,
        verbose=False,
    )


class TestBuildDataCache:
    def test_cache_miss_then_hit(self, tmp_path):
        pa = make_pattern_analysis(tmp_path)

        pa.build_data()
        cache_path = pa._data_cache_path(pa._data_fingerprint)
        assert os.path.exists(cache_path)

        first_data = pa.raw_sequence_data
        first_fingerprint = pa._data_fingerprint

        # Clear in-memory state so the second call can only succeed by reading the cache file.
        pa.raw_sequence_data = None
        pa.processed_sequence_data = None
        pa.trajectory_list = None
        pa.metadata_dictionary = None

        t0 = time.time()
        pa.build_data()
        elapsed = time.time() - t0

        assert pa._data_fingerprint == first_fingerprint
        assert len(pa.raw_sequence_data) == len(first_data)
        assert elapsed < CACHE_HIT_TIME_LIMIT

    def test_cache_invalidated_by_config_change(self, tmp_path):
        pa_a = make_pattern_analysis(tmp_path, max_samples=10)
        pa_a.build_data()

        pa_b = make_pattern_analysis(tmp_path, max_samples=7)
        pa_b.build_data()

        assert pa_a._data_fingerprint != pa_b._data_fingerprint
        assert len(pa_b.raw_sequence_data) == 7
        # Both cache entries should coexist under the shared hpc_folder.
        assert os.path.exists(pa_a._data_cache_path(pa_a._data_fingerprint))
        assert os.path.exists(pa_b._data_cache_path(pa_b._data_fingerprint))

    def test_fingerprint_differs_by_sequence_type(self, tmp_path):
        pa_a = make_pattern_analysis(tmp_path, sequence_type="first_5_seconds")
        pa_b = make_pattern_analysis(tmp_path, sequence_type="pacman_attack")

        assert pa_a._compute_data_fingerprint() != pa_b._compute_data_fingerprint()


class TestValidationEncodingsCache:
    def test_cache_hit_returns_identical_result(self, tmp_path):
        pa = make_pattern_analysis(tmp_path)
        pa.build_data()

        first = pa.calculate_validation_encodings(pa.raw_sequence_data, "Behavlets")
        cache_path = pa._validation_cache_path(pa._data_fingerprint, "Behavlets")
        assert os.path.exists(cache_path)

        t0 = time.time()
        second = pa.calculate_validation_encodings(pa.raw_sequence_data, "Behavlets")
        elapsed = time.time() - t0

        assert second.equals(first)
        assert elapsed < CACHE_HIT_TIME_LIMIT

    def test_not_cached_for_data_outside_the_pipeline(self, tmp_path):
        pa = make_pattern_analysis(tmp_path)
        pa.build_data()

        # A different list object (not `is self.raw_sequence_data`) has no known cache key,
        # so it must be computed directly rather than reused from/written to the pipeline's cache.
        custom_raw = pa.raw_sequence_data[:3]
        pa.calculate_validation_encodings(custom_raw, "Behavlets")

        cache_path = pa._validation_cache_path(pa._data_fingerprint, "Behavlets")
        assert not os.path.exists(cache_path)


class TestGifBackfill:
    """augmented_visualization is excluded from the data-cache fingerprint (it doesn't affect
    raw/processed/trajectory/metadata), so an entry cached without gifs is reused across it.
    But gif paths aren't derivable from the cached data - the only way to get them is another
    reader.make_data(make_gif=True) call - so build_data() must backfill them into the cache
    on demand rather than silently returning an empty list. reader.make_data is mocked here so
    the test doesn't depend on ffmpeg or real video files, only on build_data()'s control flow."""

    def test_backfills_gifs_into_a_gifless_cache_entry(self, tmp_path, monkeypatch):
        pa = make_pattern_analysis(tmp_path, augmented_visualization=False)
        pa.build_data()
        assert pa.gif_path_list == []
        fingerprint = pa._data_fingerprint
        real_raw, real_processed, real_features, real_traj, real_meta = (
            pa.raw_sequence_data, pa.processed_sequence_data, pa.features_columns,
            pa.trajectory_list, pa.metadata_dictionary,
        )

        pa2 = make_pattern_analysis(tmp_path, augmented_visualization=True)
        assert pa2._compute_data_fingerprint() == fingerprint  # same cache entry is reused

        calls = []

        def fake_make_data(**kwargs):
            calls.append(kwargs.get("make_gif"))
            fake_gifs = [f"fake_{i}.gif" for i in range(len(real_raw))]
            return real_raw, real_processed, fake_gifs, real_features, real_traj, real_meta

        monkeypatch.setattr(pa2.reader, "make_data", fake_make_data)
        pa2.build_data()

        assert calls == [True]  # exactly one backfill call, requesting gifs
        assert pa2.gif_path_list == [f"fake_{i}.gif" for i in range(MAX_SAMPLES)]

        cache_path = pa2._data_cache_path(fingerprint)
        cached = pa2._load_cache(cache_path, fingerprint)
        assert cached[2] == pa2.gif_path_list  # the cache entry now carries the backfilled gifs

        # A further augmented_visualization=True run must hit the now-complete cache entry
        # without calling reader.make_data() again.
        def fail_make_data(**kwargs):
            raise AssertionError("should not recompute: cache already has gifs")

        pa3 = make_pattern_analysis(tmp_path, augmented_visualization=True)
        monkeypatch.setattr(pa3.reader, "make_data", fail_make_data)
        pa3.build_data()
        assert pa3.gif_path_list == pa2.gif_path_list


class TestIgnoreCache:
    """`ignore_cache=True` (build_data / calculate_validation_encodings / fit) must skip a
    cached entry rather than trust it, and overwrite it with the freshly computed result -
    verified here by planting a tampered entry under the real cache key and checking it
    only comes back when ignore_cache is left at its default of False."""

    def test_build_data_ignore_cache_bypasses_and_refreshes_stale_entry(self, tmp_path):
        pa = make_pattern_analysis(tmp_path)
        pa.build_data()
        fingerprint = pa._data_fingerprint
        cache_path = pa._data_cache_path(fingerprint)

        pa._save_cache(cache_path, fingerprint, ("tampered", None, None, None, None, None))

        # Sanity check: without ignore_cache, the tampered entry would be trusted as-is.
        pa_trusting = make_pattern_analysis(tmp_path)
        pa_trusting.build_data()
        assert pa_trusting.raw_sequence_data == "tampered"

        # ignore_cache=True must skip the tampered entry and recompute for real.
        pa_ignoring = make_pattern_analysis(tmp_path)
        pa_ignoring.build_data(ignore_cache=True)
        assert pa_ignoring.raw_sequence_data != "tampered"
        assert len(pa_ignoring.raw_sequence_data) == MAX_SAMPLES

        # ...and it must have overwritten the cache with that fresh result.
        pa_after = make_pattern_analysis(tmp_path)
        pa_after.build_data()
        assert len(pa_after.raw_sequence_data) == MAX_SAMPLES

    def test_validation_encodings_ignore_cache_bypasses_stale_entry(self, tmp_path):
        pa = make_pattern_analysis(tmp_path)
        pa.build_data()
        cache_path = pa._validation_cache_path(pa._data_fingerprint, "Behavlets")
        pa._save_cache(cache_path, pa._data_fingerprint, "tampered")

        trusted = pa.calculate_validation_encodings(pa.raw_sequence_data, "Behavlets")
        assert trusted == "tampered"

        fresh = pa.calculate_validation_encodings(
            pa.raw_sequence_data, "Behavlets", ignore_cache=True
        )
        assert not isinstance(fresh, str)
        assert len(fresh) == MAX_SAMPLES

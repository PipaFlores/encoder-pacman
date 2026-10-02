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


class TestGifPaths:
    """GIF paths are derived from the sequences on every build_data(), never read from the
    cache, so they follow gifs_folder and survive cache entries written by older code."""

    def test_same_paths_on_cache_miss_and_hit_under_gifs_folder(self, tmp_path):
        gifs = tmp_path / "gifs"
        pa = make_pattern_analysis(tmp_path, augmented_visualization=True)
        pa.gifs_folder = str(gifs)
        pa.build_data()

        assert len(pa.gif_path_list) == MAX_SAMPLES
        assert all(os.path.dirname(p) == str(gifs) for p in pa.gif_path_list)

        # A stale third slot (as written before GIFs were named from the final sequences)
        # must not leak through on a cache hit.
        fingerprint = pa._data_fingerprint
        cache_path = pa._data_cache_path(fingerprint)
        cached = list(pa._load_cache(cache_path, fingerprint))
        cached[2] = ["./Results/subsequences/stale.gif"] * MAX_SAMPLES
        pa._save_cache(cache_path, fingerprint, tuple(cached))

        pa_hit = make_pattern_analysis(tmp_path, augmented_visualization=True)
        pa_hit.gifs_folder = str(gifs)
        pa_hit.build_data()
        assert pa_hit.gif_path_list == pa.gif_path_list


class TestGifUrlsRelativeToHtml:
    """`<img src>` must resolve from the folder the HTML is saved in, wherever that is
    relative to the GIFs (the old rewrite only worked for one particular save folder)."""

    @pytest.mark.parametrize("html", ["out.html", "figs/out.html", "a/b/out.html", "gifs/out.html"])
    def test_url_resolves_from_html_folder(self, tmp_path, html):
        gif = tmp_path / "hpc" / "gifs" / "level_1_000000_000099.gif"
        html_path = tmp_path / "notebooks" / html

        (url,) = PatternAnalysis._gif_urls_relative_to(str(html_path), [str(gif)])

        assert "\\" not in url
        assert (html_path.parent / url).resolve() == gif.resolve()


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

        # Tampered = well-formed but recognisably wrong (3 sequences instead of MAX_SAMPLES), since
        # build_data() derives GIF names from whatever sequences it loads.
        n_tampered = 3
        pa._save_cache(cache_path, fingerprint, (
            pa.raw_sequence_data[:n_tampered], pa.processed_sequence_data[:n_tampered], None,
            pa.features_columns, pa.trajectory_list[:n_tampered], pa.metadata_dictionary,
        ))

        # Sanity check: without ignore_cache, the tampered entry would be trusted as-is.
        pa_trusting = make_pattern_analysis(tmp_path)
        pa_trusting.build_data()
        assert len(pa_trusting.raw_sequence_data) == n_tampered

        # ignore_cache=True must skip the tampered entry and recompute for real.
        pa_ignoring = make_pattern_analysis(tmp_path)
        pa_ignoring.build_data(ignore_cache=True)
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

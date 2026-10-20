"""Function-by-function agreement with the R reference (see compare/README.md)."""

import pytest
from compare._rbridge import REFERENCE_SHA, find_rscript, r_package_sha

pytestmark = pytest.mark.r


@pytest.fixture(scope="module")
def r_available():
    if find_rscript() is None:
        pytest.skip("Rscript not found")
    sha = r_package_sha()
    if sha is None:
        pytest.skip("R package shrinkGPR not installed")
    if not sha.startswith(REFERENCE_SHA):
        pytest.skip(f"R shrinkGPR is at {sha[:7]}, not the reference {REFERENCE_SHA}")


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_matches_r(r_available, dtype):
    from compare.numerics import run_comparison, summarize

    rows = run_comparison(dtype)
    assert rows
    assert all(r.ok for r in rows), summarize(rows, dtype)

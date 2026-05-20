"""Tests for the distance metric registry."""

import pytest

from confetti.distances._registry import get_cdist_function
from confetti.distances._dtw import cdist_dtw
from confetti.distances._soft_dtw import cdist_soft_dtw
from confetti.distances._gak import cdist_gak
from confetti.distances._ctw import cdist_ctw
from confetti.distances._manhattan import cdist_manhattan
from confetti.errors import CONFETTIConfigurationError


@pytest.mark.parametrize("name,expected_fn", [
    ("dtw", cdist_dtw),
    ("softdtw", cdist_soft_dtw),
    ("gak", cdist_gak),
    ("ctw", cdist_ctw),
    ("manhattan", cdist_manhattan),
])
def test_known_metrics(name, expected_fn):
    assert get_cdist_function(name) is expected_fn


@pytest.mark.parametrize("name", ["DTW", "Dtw", "SOFTDTW", "Manhattan"])
def test_case_insensitive(name):
    get_cdist_function(name)


def test_unknown_metric_raises():
    with pytest.raises(CONFETTIConfigurationError):
        get_cdist_function("not-a-real-metric")

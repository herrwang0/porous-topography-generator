import numpy as np
import pytest

from ptopo.regrid.tile_utils import decompose_domain

def test_decompose_domain_covers_domain():
    N = 17
    nd = 4

    out = decompose_domain(N, nd)

    # correct number of subdomains
    assert len(out) == nd

    # starts at 0
    assert out[0][0] == 0

    # ends at N
    assert out[-1][1] == N

    # contiguous and non-overlapping
    for (s0, e0), (s1, e1) in zip(out[:-1], out[1:]):
        assert e0 == s1
        assert s0 < e0

    # total length equals N
    lengths = [e - s for s, e in out]
    assert sum(lengths) == N

def test_decompose_domain_even_split():
    out = decompose_domain(8, 4)
    expected = np.array([(0,2), (2,4), (4,6), (6,8)], dtype='i,i')
    assert np.array_equal(out, expected)

def test_decompose_domain_remainder_nonsymmetric():
    out = decompose_domain(10, 3, symmetric=False)
    lengths = [e - s for s, e in out]

    # remainder goes to first blocks
    assert lengths == [4, 3, 3]

def test_decompose_domain_symmetric_distribution():
    out = decompose_domain(10, 4, symmetric=True)
    lengths = np.array([e - s for s, e in out])

    # symmetry around center
    assert lengths[0] == lengths[-1]
    assert lengths[1] == lengths[-2]
    assert lengths.sum() == 10

def test_warning_nd_gt_N():
    with pytest.warns(UserWarning):
        decompose_domain(3, 10)

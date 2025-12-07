"""
tile_utils.py

Pure helper functions for creating tiles
"""

import numpy
from dataclasses import dataclass
from typing import Tuple, Union

GLOBAL_POS = object()

@dataclass(frozen=True)
class BoundaryBox:
    """Rectangular subdomain boundary defined in global coordinates.

    Parameters
    ----------
    j_start, j_end, i_start, i_end : int
        Global computation domain j-index and i-index range.
    halo : int or (int, int)
        Halo width. If a single integer is given, it is applied to both
        j and i directions. If a tuple, interpreted as (halo_j, halo_i).
    position : tuple[int, int]
        The tile's index in the global layout grid. (iy, ix) = (row_index, column_index).
    """

    j_start: int
    j_end: int
    i_start: int
    i_end: int
    halo: Union[int, Tuple[int, int]] = 0
    position: Union[Tuple[int, int], object] = GLOBAL_POS

    def __post_init__(self):
        # Convert single halo to a (halo_j, halo_i) tuple
        if isinstance(self.halo, int):
            object.__setattr__(self, "halo", (self.halo, self.halo))
        elif (
            isinstance(self.halo, tuple)
            and len(self.halo) == 2
            and all(isinstance(h, int) for h in self.halo)
        ):
            pass
        else:
            raise ValueError("halo must be an int or a tuple of two ints.")

        if not (self.j_end > self.j_start and self.i_end > self.i_start):
            raise ValueError("End indices must be greater than start indices.")

    @property
    def is_global(self) -> bool:
        return self.position is GLOBAL_POS

    @property
    def global_compute_j_slice(self) -> slice:
        """Full global j-slice excluding halo."""
        return slice(self.j_start, self.j_end)

    @property
    def global_compute_i_slice(self) -> slice:
        """Full global i-slice excluding halo."""
        return slice(self.i_start, self.i_end)

    @property
    def global_compute_box(self) -> Tuple[slice, slice]:
        """Full global box including halo."""
        return (self.global_compute_j_slice, self.global_compute_i_slice)

    @property
    def halo_j(self) -> int:
        return self.halo[0]

    @property
    def halo_i(self) -> int:
        return self.halo[1]

    @property
    def compute_nj(self) -> int:
        """nj of the compute region."""
        return self.j_start.stop - self.global_j_slice.start

    @property
    def compute_ni(self) -> int:
        """ni of the compute region."""
        return self.global_i_slice.stop - self.global_i_slice.start

    @property
    def compute_shape(self) -> int:
        """Shape of the compute region."""
        return ( self.compute_nj, self.compute_ni )

    @property
    def nj(self) -> int:
        """Data nj of the region."""
        return self.compute_nj + 2 * self.halo_j

    @property
    def ni(self) -> int:
        """Data ni of the region."""
        return self.compute_ni + 2 * self.halo_i

    @property
    def shape(self) -> Tuple[int, int]:
        """Data shape of the region."""
        return ( self.nj, self.ni )

    @property
    def compute_j_slice(self) -> slice:
        """Compute local j-slice (excludes halo)."""
        return slice(self.halo_j, self.nj - self.halo_j)

    @property
    def compute_i_slice(self) -> slice:
        """Compute local i-slice (excludes halo)."""
        return slice(self.halo_i, self.ni - self.halo_i)

    @property
    def compute_box(self) -> Tuple[slice, slice]:
        """Compute local box."""
        return self.compute_j_slice, self.compute_i_slice

def slice_array(arr, box, position='corner', fold_north=True, cyclic_zonal=True, fold_south=False):
    """Slice a 2D field with extend indices that cover halos
    Treatment of halos beyond the boundaries:
    If not specified, all boundaries are assumed to be reflective.
        For example, to extend 2 point from | 0, 1, 2 ... => 1, 0, | 0, 1, 2...
    By default, cyclic boundary is assumed at the western and eastern boundaries,
    and folding boundary is assumed at the northern boundary. The southern boundary is ignored.
    This method is operated over the "symmetric" grid strutures, i.e. corner fields are
    one-point larger in each direction.

    Parameters
    ----------
    arr : 2D ndarray
        Input field
    box : tuple
        Four-element tuple (jst, jed, ist, ied). The indices are for cell centers.
    position : string, optional
        Grid position of the input field. Can be either 'center', 'corner', 'u', and 'v'. Default is 'corner'
    fold_north : bool, optional
        If true, folding (in the middle) boundary is assumed at the northern end (bi-polar cap).
        Default is True.
    cyclic_zonal : bool, optional
        If true, cyclic boundary is assumed in the zonal direction.
        Default is True
    fold_north : bool, optional
        Placeholder for a folding boundary at the southern end. Does not work if True.
        Default is False.

    Returns
    ----------
    Out : ndarray
        sliced field
    """
    Nj, Ni = arr.shape
    jst, jed, ist, ied = box

    # Additional points for staggered locations (symmetric)
    if position == 'center':
        oy, ox = 0, 0
    elif position == 'corner':
        oy, ox = 1, 1
    elif position == 'u':
        oy, ox = 0, 1
    elif position == 'v':
        oy, ox = 1, 0
    else: raise Exception('Unknown grid position.')

    # The center piece reach the natural boundaries of the domain
    #   This is equivalent to slice(max(jst,0), min(jed+oy, Ni))
    yc = slice(jst-Nj, jed+oy)
    xc = slice(ist-Ni, ied+ox)

    if cyclic_zonal:
        # Cyclic boundary condition at the western and eastern end
        #   For boundary grid, we need to move 1 point "outward" to exclude shared western and eastern boundaries.
        xw = slice(Ni+ist-ox, Ni-ox)  # [] if ist>=0
        xe = slice(-Ni+ox, -Ni+ox+(ied+ox-Ni))  # [] if ied+ox<=Ni
    else:
        # reflective
        xw = slice(-Ni-ist, -Ni, -1)
        xe = slice(Ni-2, Ni-2-(ied+ox-Ni), -1)
        # No extention
        # xw = slice(0, 0)
        # xe = slice(0, 0)

    if fold_north:
        # Fold northern boundary
        #   For boundary grid, we need to move 1 point southward to exclude shared northern boundary.
        yn = slice(Nj-1-oy, Nj-1-oy-(jed+oy-Nj), -1)
        xn = slice(-1-(ist-Ni), -1-(ied+ox), -1)  # ii=0 <-> ii=Ni-1 (-1), ii=1 <-> ii=Ni-2 (-2) ...
    else:
        # reflective
        yn = slice(Nj-2, Nj-2-(jed+oy-Nj), -1)
        # No extention
        # yn = slice(0, 0)
        xn = xc

    if cyclic_zonal and fold_north:
        xnw = slice(-1-(Ni+ist-ox), -1-(Ni-ox), -1)
        xne = slice(-1-(-Ni+ox), -1-(-Ni+ox+(ied+ox-Ni)), -1) # slice(Ni-1-ox, Ni-1-ox-(ied+ox-Ni), -1)
    else:
        xnw = xw
        xne = xe

    if fold_south:
        raise Exception('fold_south is not supported.')
    else:
        # reflective
        ys = slice(-Nj-jst, -Nj, -1)
        # No extention over southern boundary, a place holder
        # ys = slice(0, 0)
        xs = xc
        xsw = xw
        xse = xe

    NoWe, North, NoEa = arr[yn, xnw], arr[yn, xn], arr[yn, xne]
    West, Center, East = arr[yc, xw], arr[yc, xc], arr[yc, xe]
    SoWe, South, SoEa = arr[ys, xsw], arr[ys, xs], arr[ys, xse]
    # print(NoWe.shape, North.shape, NoEa.shape)
    # print(West.shape, Center.shape, East.shape)
    # print(SoWe.shape, South.shape, SoEa.shape)
    return numpy.r_[numpy.c_[SoWe, South, SoEa], numpy.c_[West, Center, East], numpy.c_[NoWe, North, NoEa]]

def decompose_domain(N, nd, symmetric=False):
    """Decompose 1-D domain

    Parameters
    ----------
    N : integer
        Number of grid points in the 1-D domain
    nd : integer
        Number of sub-domains
    symmetric : bool, optional
        When true, try to generate a symmetric decomposition

    Returns
    ----------
    Out : ndarray
        A list of starting and ending indices of the sub-domains
    """

    if nd>N:
        print("Warning: number of sub-domains > number of grid points. Actual sub-domain number is reduced to %i."%N)
        return numpy.array([(ist,ist+1) for ist in range(N)], dtype='i,i')

    dn = numpy.ones((nd,), dtype=numpy.int16)*(N//nd)
    res = N%nd

    if ((nd%2==0) and (res%2==1)) or (not symmetric):
        dn[:res] += 1
    else:
        if (nd%2==0) and (res%2==0):
            dn[nd//2-res//2:nd//2+res//2] += 1
        if (nd%2==1):
            dn[nd//2-res//2:nd//2+res//2+1] += 1
            if (res%2==0): dn[nd//2] -= 1
    return numpy.array([(ist, ied) for ist, ied in zip(numpy.r_[0, numpy.cumsum(dn)[:-1]],numpy.cumsum(dn))], dtype='i,i')

def box_halo(box, halo):
    """Extend box with halo"""
    jst, jed, ist, ied = box
    if isinstance(halo, (tuple, list)):
        halo_j, halo_i = halo
    else:
        halo_j, halo_i = halo, halo
    return (jst - halo_j, jed + halo_j, ist - halo_i, ied + halo_j)

def normlize_longitude(lon, lat):
    """Shift longitude by 360*n so that it is monotonic (when it is possible)"""
    # To avoid jumps, the reference longitude should be outside of the domain.
    # reflon is a best-guess of a reference longitude.
    reflon = lon[lon.shape[0]//2, lon.shape[1]//2] - 180.0
    lon_shift = numpy.mod(lon-reflon, 360.0) + reflon

    # North Pole longitude should be within the range of the rest of the domain
    Jp, Ip = numpy.nonzero(lat==90)
    for jj, ii in zip(Jp, Ip):
        if lon_shift[jj, ii]==lon_shift.max() or lon_shift[jj, ii]==lon_shift.min():
            lon_shift[jj, ii] = numpy.nan
            lon_shift[jj, ii] = numpy.nanmean(lon_shift)
    return lon_shift
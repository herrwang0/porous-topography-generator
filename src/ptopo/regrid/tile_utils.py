"""
tile_utils.py

Pure helper functions for creating tiles
"""

import numpy as np
import warnings
from dataclasses import dataclass
from typing import Tuple, Union

GLOBAL_POS = object()

@dataclass(frozen=True)
class BoundaryBox:
    """Rectangular subdomain boundary defined in global coordinates.

    Parameters
    ----------
    j0, j1, i0, i1 : int
        Global computation domain j-index and i-index range.
    halo : int or (int, int)
        Halo width. If a single integer is given, it is applied to both
        j and i directions. If a tuple, interpreted as (halo_j, halo_i).
    position : tuple[int, int]
        The tile's index in the global layout grid. (iy, ix) = (row_index, column_index).
    """

    j0: int
    j1: int
    i0: int
    i1: int
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

        if not (self.j1 > self.j0 and self.i1 > self.i0):
            raise ValueError("End indices must be greater than start indices.")

    def __repr__(self):
        return (
            f"BoundaryBox("
            f"j0={self.j0}, j1={self.j1}, i0={self.i0}, i1={self.i1}, halo={self.halo}, "
            f"position={self.position}"
            f")"
        )

    def __str__(self):
        # alias
        gj0, gj1 = self.jdg_slice.start, self.jdg_slice.stop
        gi0, gi1 = self.idg_slice.start, self.idg_slice.stop

        # Position (tile indices); (-1, -1) means global box
        pos = "global" if self.position == GLOBAL_POS else f"{self.position}"

        # Construct lines
        disp = [
            str(type(self)),
            f"  position = {pos}",
            f"  global computer domain: ",
            f"     (nj, ni) = ({self.nj}, {self.ni}), (j0, j1, i0, i1) = ({self.j0}, {self.j1}, {self.i0}, {self.i1})",
            f"  global data domain: ",
            f"     (nj, ni) = ({self.data_nj}, {self.data_ni}), (j0, j1, i0, i1) = ({gj0}, {gj1}, {gi0}, {gi1})"
        ]
        return '\n'.join(disp)

    @property
    def is_global(self) -> bool:
        return self.position is GLOBAL_POS

    # ============================================================
    # Size and shape
    # ============================================================
    @property
    def halo_j(self) -> int:
        return self.halo[0]

    @property
    def halo_i(self) -> int:
        return self.halo[1]

    @property
    def nj(self) -> int:
        """nj of the computation region."""
        return self.j1 - self.j0

    @property
    def ni(self) -> int:
        """ni of the computation region."""
        return self.i1 - self.i0

    @property
    def shape(self) -> int:
        """Shape of the computation region."""
        return ( self.nj, self.ni )

    @property
    def data_nj(self) -> int:
        """nj of the data region."""
        return self.nj + 2 * self.halo_j

    @property
    def data_ni(self) -> int:
        """ni of the data region."""
        return self.ni + 2 * self.halo_i

    @property
    def data_shape(self) -> Tuple[int, int]:
        """Shape of the data region."""
        return ( self.data_nj, self.data_ni )

    # ============================================================
    # Global computation (excluding halo) domain
    # ============================================================
    @property
    def jcg_slice(self) -> slice:
        """Computation global j-slice."""
        return slice(self.j0, self.j1)

    @property
    def icg_slice(self) -> slice:
        """Computation global i-slice."""
        return slice(self.i0, self.i1)

    @property
    def J0cg(self) -> int:
        """Computation global southern edge."""
        return self.j0

    @property
    def J1cg(self) -> int:
        """Computation global northern edge."""
        return self.j1

    @property
    def I0cg(self) -> int:
        """Computation global western edge."""
        return self.i0

    @property
    def I1cg(self) -> int:
        """Computation global eastern edge."""
        return self.i1

    @property
    def Jcg_inner_slice(self) -> slice:
        """Computation global J-slice for inner edges."""
        return slice(self.j0 + 1, self.j1)

    @property
    def Icg_inner_slice(self) -> slice:
        """Computation global I-slice for inner edges."""
        return slice(self.i0 + 1, self.i1)

    @property
    def Jcg_outer_slice(self) -> slice:
        """Computation global j-slice for outer edges."""
        return slice(self.j0, self.j1 + 1)

    @property
    def Icg_outer_slice(self) -> slice:
        """Computation global i-slice for outer edges."""
        return slice(self.i0, self.i1 + 1)

    # ============================================================
    # Global data (including halo) domain
    # ============================================================
    @property
    def jdg_slice(self) -> slice:
        """Data global j-slice."""
        return slice(self.j0 - self.halo_j, self.j1 + self.halo_j)

    @property
    def idg_slice(self) -> slice:
        """Data global i-slice."""
        return slice(self.i0 - self.halo_i, self.i1 + self.halo_i)

    # ============================================================
    # Local computation (excluding halo) domain
    # ============================================================
    @property
    def jcl_slice(self) -> slice:
        """Computation local j-slice."""
        return slice(self.halo_j, self.data_nj - self.halo_j)

    @property
    def icl_slice(self) -> slice:
        """Computation local i-slice."""
        return slice(self.halo_i, self.data_ni - self.halo_i)

    @property
    def J0cl(self) -> int:
        """Computation local southern edge."""
        return self.halo_j

    @property
    def J1cl(self) -> int:
        """Computation local northern edge."""
        return self.data_nj - self.halo_j

    @property
    def I0cl(self) -> int:
        """Computation local western edge."""
        return self.halo_i

    @property
    def I1cl(self) -> int:
        """Computation local eastern edge."""
        return self.data_ni - self.halo_i

    @property
    def Jcl_inner_slice(self) -> slice:
        """Computation local J-slice for inner edges."""
        return slice(self.halo_j + 1, self.data_nj - self.halo_j)

    @property
    def Icl_inner_slice(self) -> slice:
        """Computation local I-slice for inner edges."""
        return slice(self.halo_i + 1, self.data_ni - self.halo_i)

    @property
    def Jcl_outer_slice(self) -> slice:
        """Computation local J-slice for outer edges."""
        return slice(self.halo_j, self.data_nj - self.halo_j + 1)

    @property
    def Icl_outer_slice(self) -> slice:
        """Computation local I-slice for outer edges."""
        return slice(self.halo_i, self.data_ni - self.halo_i + 1)

    # ============================================================
    # Local data (including halo) domain
    # ============================================================
    @property
    def jdl_slice(self) -> slice:
        """Data local j-slice."""
        return slice(0, self.data_nj)

    @property
    def idl_slice(self) -> slice:
        """Data local i-slice."""
        return slice(0, self.data_ni)

    def local_masks(self, mask_boxes):
        """Finds where the mask rectangles overlap"""
        masks = []

        J0, J1, I0, I1 = self.jdg_slice.start, self.jdg_slice.stop, self.idg_slice.start, self.idg_slice.stop

        for box in mask_boxes:
            j0m, j1m, i0m, i1m = box.jdg_slice.start, box.jdg_slice.stop, box.idg_slice.start, box.idg_slice.stop

            # Relative indices
            j0 = max(J0, j0m) - J0
            j1 = min(J1, j1m) - J0
            i0 = max(I0, i0m) - I0
            i1 = min(I1, i1m) - I0

            # if mask boundary is beyond subdomain boundary but within halo, ignore halo
            if j0m <= self.j0: j0 = 0
            if j1m >= self.j1: j1 = self.data_nj
            if i0m <= self.i0: i0 = 0
            if i1m >= self.i1: i1 = self.data_ni
            # # The following addresses a very trivial case when the mask reaches
            # # the southern bounndary, which may only happen in tests.
            # if j0m == 0 and self.j0 < 0: j0 = 0

            if j1 > j0 and i1 > i0:
                masks.append(
                    BoundaryBox(j0=j0, j1=j1, i0=i0, i1=i1, halo=0, position=GLOBAL_POS)
                )
        return masks

    def with_halo(self, halo: Union[int, tuple[int, int]]):
        """
        Return a new BoundaryBox with the same global computation bounds
        but a different halo.
        """
        return BoundaryBox(
            j0=self.j0,
            j1=self.j1,
            i0=self.i0,
            i1=self.i1,
            halo=halo,
            position=self.position
        )

def reverse_slice(s: slice) -> slice:
    start, stop = s.start, s.stop
    step = s.step or 1

    if step <= 0:
        raise ValueError("reverse_slice only supports positive-step slices")

    # For a forward slice(start, stop), Python includes start and excludes stop.
    # The reversed slice needs:
    #   new_start = stop - 1
    #   new_stop  = start - 1
    return slice(stop - 1, start - 1, -step)

def slice_array(arr, bbox, position='corner', fold_north=True, cyclic_zonal=True, fold_south=False):
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
    bbox : BoundaryBox
        BoundaryBox that has four-element tuple (jst, jed, ist, ied). The indices are for cell centers.
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
    # jst, jed, ist, ied = box
    jst, jed, ist, ied = bbox.jdg_slice.start, bbox.jdg_slice.stop, bbox.idg_slice.start, bbox.idg_slice.stop

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
    return np.r_[np.c_[SoWe, South, SoEa], np.c_[West, Center, East], np.c_[NoWe, North, NoEa]]

def decompose_domain(N, nd, symmetric=False):
    """
    Decompose a 1-D domain into contiguous sub-domains.

    Parameters
    ----------
    N : int
        Number of grid points in the 1-D domain
    nd : int
        Number of requested sub-domains
    symmetric : bool, optional
        When True, attempt to distribute remainder points symmetrically

    Returns
    -------
    ndarray of shape (k, 2)
        Each row is (start, end) for a sub-domain
    """

    if N < 0:
        raise ValueError("N must be non-negative")
    if nd <= 0:
        raise ValueError("nd must be positive")

    # Degenerate case: more subdomains than points
    if nd > N:
        warnings.warn(
            f"number of sub-domains ({nd}) > number of grid points ({N}); "
            f"reducing to {N}",
            UserWarning,
        )
        return np.array([(i, i + 1) for i in range(N)], dtype="i,i")

    # Base size and remainder
    base = N // nd
    rem = N % nd

    sizes = np.full(nd, base, dtype=np.int16)

    if rem > 0:
        if not symmetric or (nd % 2 == 0 and rem % 2 == 1):
            # Left-biased distribution
            sizes[:rem] += 1
        else:
            # Symmetric distribution around center
            mid = nd // 2
            half = rem // 2

            if nd % 2 == 0: # rem % 2 == 0
                sizes[mid - half : mid + half] += 1
            else:
                sizes[mid - half : mid + half + 1] += 1
                if rem % 2 == 0:
                    sizes[mid] -= 1

    starts = np.concatenate(([0], np.cumsum(sizes[:-1])))
    ends = np.cumsum(sizes)

    return np.array(list(zip(starts, ends)), dtype="i,i")

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
    lon_shift = np.mod(lon-reflon, 360.0) + reflon

    # North Pole longitude should be within the range of the rest of the domain
    Jp, Ip = np.nonzero(lat==90)
    for jj, ii in zip(Jp, Ip):
        if lon_shift[jj, ii]==lon_shift.max() or lon_shift[jj, ii]==lon_shift.min():
            lon_shift[jj, ii] = np.nan
            lon_shift[jj, ii] = np.nanmean(lon_shift)
    return lon_shift
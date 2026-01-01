import numpy as np
import warnings

from ptopo.external.thinwall.python import ThinWalls
from .tile_utils import slice_array, decompose_domain, normalize_longitude, box_halo, BoundaryBox, reverse_slice
from .configs import CalcConfig, TileConfig

class Domain(ThinWalls.ThinWalls):
    """
    A wrapper of ThinWalls object for tiling

    1) Attributes Idx, Idy : inverse of v, u point spacing
    2) Attributes [cuv]_rfl : refine level arrays for center, u, v points
    3) Methods for creating a subdomain, generating/stitching back tiles and sporing a mask domain
    """
    def __init__(self, lon=None, lat=None, Idx=None, Idy=None, is_geo_coord=True,
                 reentrant_x=False, fold_n=False, bbox=None, mask_res=[],
                 eds=None, subset_eds=False, src_halo=0):
        """
        Parameters
        ----------
        lon, lat : float
            Cell corner coordinates to construct ThinWalls
        Idx, Idy : float, optional
            The inverse of v and u point grid spacing, used for calculating gradient.
        is_geo_coord : bool, optional
            If False, lon/lat are not geographic coordinates. Default is True.
        reentrant_x : bool, optional
            If true, the domain is reentrant in x-direction. Used for halos and assigning depth to the westernmost and easternmost u-edges.
            Default is False.
        fold_n : bool, optional
            If true, the domain is folded at the northern boundary. Used for halos and assigning depth to the northernmost v-edges.
            Default is False.
        bbox : BoundaryBox, optional
            BoundaryBox that includes indexing information
        mask_res : list, optional
            List of rectangles to exclude regions to calculate resolution limit
        eds : GMesh.UniformEDS objects, optional
            Contains source coordinates and data.
        subset_eds : bool, optional
            If true, source data eds to encompass target grid, with a halo decided by src_halo. Default is False.
        src_halo : integer, optional
            Halo size of the source grid in either direction.
        """
        super().__init__(lon=lon, lat=lat, is_geo_coord=is_geo_coord)
        self.Idx, self.Idy = Idx, Idy
        self.c_rfl = np.zeros( self.shape, dtype=np.int32 )
        self.u_rfl = np.zeros( (self.nj, self.ni + 1), dtype=np.int32 )
        self.v_rfl = np.zeros( (self.nj + 1, self.ni), dtype=np.int32 )

        if bbox:
            assert (self.nj == bbox.data_nj) and (self.ni == bbox.data_ni), f'bbox incorrect {self.nj}!= {bbox.nj}, {self.ni}!= {bbox.ni}'
            self.bbox = bbox
        else:
            self.bbox = BoundaryBox(j0=0, j1=self.nj, i0=0, i1=self.ni, halo=0)

        self.reentrant_x = reentrant_x
        self.fold_n = fold_n
        if self.fold_n:
            assert self.ni % 2 == 0, 'An odd number ni does not work with bi-polar cap.'

        self.mask_res = mask_res

        if eds: self._fit_src_coords(eds, subset_eds=subset_eds, halo=src_halo)

    def __str__(self):
        return self.format(indent=0)

    def format(self, indent=0):
        pad = " " * indent

        if self.is_geo_coord:
            coord_name = 'lat lon'
            lonmin, lonmax = np.mod(self.lon.min(), 360), np.mod(self.lon.max(), 360)
            src_lonmin, src_lonmax = np.mod(self.eds.lon_coord.bounds[0], 360), np.mod(self.eds.lon_coord.bounds[-1], 360)
        else:
            coord_name = 'y x'
            lonmin, lonmax = self.lon.min(), self.lon.max()
            src_lonmin, src_lonmax = self.eds.lon_coord.bounds[0], self.eds.lon_coord.bounds[-1]

        disp = [
            f'{pad}' + str(type(self)),
            self.bbox.format(indent=indent + 2),
            f'{pad}  Geographic coordinate? {self.is_geo_coord}',
            f'{pad}  Domain size (nj ni): ({self.nj}, {self.ni})',
            f'{pad}  Domain range ({coord_name}): [{self.lat.min():10.6f}, {self.lat.max():10.6f}], [{lonmin:10.6f}, {lonmax:10.6f}]',
            f'{pad}  Source grid size (nj ni): ({self.eds.nj:9d}, {self.eds.ni:9d}) ' + \
            f'{pad}  indices: ({self.eds.lat_coord.start}, {self.eds.lat_coord.stop}, {self.eds.lon_coord.start}, {self.eds.lon_coord.stop})',
            f'{pad}  Source grid range ({coord_name}): [{self.eds.lat_coord.bounds[0]:10.6f}, {self.eds.lat_coord.bounds[-1]:10.6f}], [{src_lonmin:10.6f}, {src_lonmax:10.6f}]'
        ]

        if len(self.mask_res)>0:
            disp.append(f'{pad}  Mask rectangles')
            for box in self.mask_res:
                disp.append(f'{pad}    js,je,is,ie: {box[0]}, {box[1]}, {box[2]}, {box[3]}, shape: ({box[1]-box[0]}, {box[3]-box[2]})')

        return '\n'.join(disp)

    def _fit_src_coords(self, eds, subset_eds=True, halo=0):
        """Returns the four-element indices of source grid that covers the current domain."""
        if subset_eds:
            Is, Ie, Js, Je = eds.bb_slices( self.lon, self.lat, halo_lon=halo, halo_lat=halo )
            # If the North Pole is encompassed in the domain, we will need the full longitude circle,
            # regardless what bb_slices thinks.
            for jj, ii in self.np_index:
                if (jj>0 and jj<self.shape[0]+1) or (ii>0 and ii<self.shape[1]+1): # North pole is NOT on the boundaries.
                    Is, Ie = 0, eds.ni
                    break
            self.eds = eds.subset(Is, Ie, Js, Je)
        else:
            self.eds = eds

    def update_thinwalls_arrays(self, tw):
        """
        Replace the Domain's ThinWalls arrays with the ones from `tw` without copying
        """
        self.c_simple = tw.c_simple
        self.u_simple = tw.u_simple
        self.v_simple = tw.v_simple
        self.c_effective = tw.c_effective
        self.u_effective = tw.u_effective
        self.v_effective = tw.v_effective

    def make_subdomain(self, bbox, norm_lon=False, reentrant_x=None, fold_n=None, global_masks=[], subset_eds=False, src_halo=0):
        """
        Makes a sub-domains from a BoundaryBox.

        Parameters
        ----------
        bbox : BoundaryBox object
        norm_lon : bool, optional
            If True, make sure there is no longitude "jumps" (except for poles) in the subdomain.
        reentrant_x : bool | None, optional
            If None, use self.reentrant_x
        fold_n : bool | None, optional
            If None, use self.fold_n
        global_masks : list
            List of resolution mask BoundaryBox
        subset_eds : bool, optional
            If True, subset source data
        src_halo : integer, optional
            Halo size of the source grid in either direction.
        verbose : bool, optional

        Returns
        ----------
        Out : Domain object
        """

        if reentrant_x is None: reentrant_x = self.reentrant_x
        if fold_n is None: fold_n = self.fold_n

        lon = slice_array(self.lon, bbox=bbox, cyclic_zonal=reentrant_x, fold_north=fold_n)
        lat = slice_array(self.lat, bbox=bbox, cyclic_zonal=reentrant_x, fold_north=fold_n)
        if norm_lon: lon = normalize_longitude(lon, lat)

        if self.Idx is None: Idx = None
        else: Idx = slice_array(self.Idx, bbox=bbox, position='center', cyclic_zonal=reentrant_x, fold_north=fold_n)
        if self.Idy is None: Idy = None
        else: Idy = slice_array(self.Idy, bbox=bbox, position='center', cyclic_zonal=reentrant_x, fold_north=fold_n)

        masks = []
        for mask_in in global_masks:
            if bbox.local_mask(mask_in):
                masks.append( bbox.local_mask(mask_in).to_box() )

        return Domain(
            lon=lon, lat=lat, Idx=Idx, Idy=Idy, is_geo_coord=self.is_geo_coord,
            reentrant_x=False, fold_n=False, bbox=bbox, mask_res=masks,
            eds=self.eds, subset_eds=subset_eds, src_halo=src_halo
        )

    def make_tiles(self, config=TileConfig(), verbose=False):
        """Creates a list of sub-domains with corresponding source lon, lat and elev sections.

        Parameters
        ----------
        pelayout : tuple of integers, (nj, ni)
            Number of sub-domains in each direction
        tgt_halo : int, optional
            Halo size
        symmetry : tuple of bool, optional
            (y_sym, x_sym). Whether try to use symmetric domain decomposition in x or y.
            Default is (False, True).
        norm_lon : bool, optional
            If True, make sure there is no longitude "jumps" (except for poles) in the subdomain.
        subset_eds : bool, optional
            If True, subset source data
        src_halo : integer, optional
            Halo size of the source grid in either direction.
        verbose : bool, optional

        Returns
        ----------
        self.pelayout : a 2D array documenting indices of each subdomain
        Out : ndarray
            A 2D array of RefineWrapper objects
        """

        norm_lon = self.is_geo_coord if config.norm_lon is None else config.norm_lon

        if config.pelayout[1]==1 and config.tgt_halo>0:
            warnings.warn(
                f"only 1 subdomain in i-direction, which may not work with bi-polar cap.",
                UserWarning
            )
        if (not config.symmetry[1]) and self.fold_n:
            warnings.warn(
                "domain decomposition is not guaranteed to be symmetric in x direction, which may not work with bi-polar cap.",
                UserWarning
            )

        j_domain = decompose_domain(self.nj, config.pelayout[0], symmetric=config.symmetry[0])
        i_domain = decompose_domain(self.ni, config.pelayout[1], symmetric=config.symmetry[1])
        if verbose:
            print(f'Domain is decomposed to {config.pelayout}. Halo size = {config.tgt_halo:d}.')
            print('  i: ', i_domain)
            print('  j: ', j_domain)
            print('\n')

        chunks = np.empty( (j_domain.size, i_domain.size), dtype=object )
        self.pelayout = np.empty( (j_domain.size, i_domain.size), dtype=object )
        for pe_j, (jst, jed) in enumerate(j_domain):
            for pe_i, (ist, ied) in enumerate(i_domain):
                self.pelayout[pe_j, pe_i] = ((jst, jed, ist, ied), config.tgt_halo) # indices for cell centers

                bbox = BoundaryBox(jst, jed, ist, ied, config.tgt_halo, (pe_j, pe_i))
                chunks[pe_j, pe_i] = self.make_subdomain(
                    bbox, norm_lon=norm_lon, global_masks=self.mask_res, subset_eds=config.subset_eds, src_halo=config.src_halo
                )

                if verbose:
                    print(chunks[pe_j, pe_i], '\n')
        return chunks

    def _stitch_tile(self, tile, config):
        """
        Stitch all cell-center and all edges properties
        """
        jg_slice, ig_slice = tile.bbox.jcg_slice, tile.bbox.icg_slice
        jl_slice, il_slice = tile.bbox.jcl_slice, tile.bbox.icl_slice

        Jg_slice, Ig_slice = tile.bbox.Jcg_outer_slice, tile.bbox.Icg_outer_slice
        Jl_slice, Il_slice = tile.bbox.Jcl_outer_slice, tile.bbox.Icl_outer_slice

        self.c_simple[jg_slice, ig_slice] = tile.c_simple[jl_slice, il_slice]
        self.c_rfl[jg_slice, ig_slice] = tile.c_rfl[jl_slice, il_slice]

        # inner edges
        if config.calc_thinwalls:
            self.u_simple[jg_slice, Ig_slice] = tile.u_simple[jl_slice, Il_slice]
            self.v_simple[Jg_slice, ig_slice] = tile.v_simple[Jl_slice, il_slice]
            self.u_rfl[jg_slice, Ig_slice] = tile.u_rfl[jl_slice, Il_slice]
            self.v_rfl[Jg_slice, ig_slice] = tile.v_rfl[Jl_slice, il_slice]

            if config.calc_effective_tw:
                self.c_effective[jg_slice, ig_slice] = tile.c_effective[jl_slice, il_slice]
                self.u_effective[jg_slice, Ig_slice] = tile.u_effective[jl_slice, Il_slice]
                self.v_effective[Jg_slice, ig_slice] = tile.v_effective[Jl_slice, il_slice]

        if config.calc_roughness: self.roughness[jg_slice, ig_slice] = tile.roughness[jl_slice, il_slice]
        if config.calc_gradient: self.gradient[jg_slice, ig_slice] = tile.gradient[jl_slice, il_slice]

    def _stitch_i(self, left, right, tolerance, calc_effective=False, verbose=False):
        """
        Reconcile shared edges from two providers (tile or domain) along i.
        """
        lbox, rbox = left.bbox, right.bbox
        if left is self and right is self:
            raise Exception('Both "left" and "right" are the parent Domain.')

        elif left == self: # "left" is the parent Domain and "right" is a tile.
            edgeloc = f'Tile western edge [{rbox.position}]'
            global_j_slice, global_I = rbox.jcg_slice, rbox.i0
            local_j_slice_l, local_I_l = global_j_slice, global_I
            local_j_slice_r, local_I_r = rbox.jcl_slice, rbox.I0cl

        elif right is self: # "right" is the parent Domain and "left" is a tile.
            edgeloc = f'Tile eastern edge [{lbox.position}]'
            global_j_slice, global_I = lbox.jcg_slice, lbox.i1
            local_j_slice_l, local_I_l = lbox.jcl_slice, lbox.I1cl
            local_j_slice_r, local_I_r = global_j_slice, global_I

        else: # Both "left" and "right" are tiles
            edgeloc = f'{lbox.position} and {rbox.position}'
            global_j_slice_l, global_I_l = lbox.jcg_slice, lbox.i1
            global_j_slice_r, global_I_r = rbox.jcg_slice, rbox.i0

            i_match = (global_I_l == global_I_r) or (self.reentrant_x and global_I_l==self.ni and global_I_r==0)
            if  (not i_match) or global_j_slice_l != global_j_slice_r:
                raise ValueError( f"""{edgeloc} do not match. global_I_l={global_I_l}, global_I_r={global_I_r},
                                 global_j_slice_l={global_j_slice_l}, global_j_slice_r={global_j_slice_r}.""")

            global_j_slice, global_I = global_j_slice_l, global_I_l
            local_j_slice_l, local_I_l = lbox.jcl_slice, lbox.I1cl
            local_j_slice_r, local_I_r = rbox.jcl_slice, rbox.I0cl

        rfl_l = left.u_rfl[local_j_slice_l, local_I_l]
        rfl_r = right.u_rfl[local_j_slice_r, local_I_r]

        self.u_simple[global_j_slice, global_I] = match_edges(
            left.u_simple[local_j_slice_l, local_I_l], right.u_simple[local_j_slice_r, local_I_r], rfl_l, rfl_r,
            tolerance=tolerance, verbose=verbose, message=edgeloc+' (simple)'
        )
        self.u_rfl[global_j_slice, global_I] = np.maximum(rfl_l, rfl_r)

        if calc_effective:
            self.u_effective[global_j_slice, global_I] = match_edges(
                left.u_effective[local_j_slice_l, local_I_l], right.u_effective[local_j_slice_r, local_I_r], rfl_l, rfl_r,
                tolerance=tolerance, verbose=verbose, message=edgeloc+' (effective)'
            )

    def _stitch_j(self, bottom, top, tolerance, calc_effective=False, verbose=False):
        """
        Reconcile shared edges from two providers (tile or domain) along j.
        """
        bbox, tbox = bottom.bbox, top.bbox
        if bottom is self and top is self:
            raise Exception('Both "bottom" and "top" are the parent Domain.')

        elif bottom is self: # "bottom" is the parent Domain and "top" is a tile.
            edgeloc = f'Tile southern edge [{tbox.position}]'
            global_J, global_i_slice = tbox.j0, tbox.icg_slice
            local_J_b, local_i_slice_b = global_J, global_i_slice
            local_J_t, local_i_slice_t = tbox.J0cl, tbox.icl_slice

        elif top is self: # "top" is the parent Domain and "bottom" is a tile.
            edgeloc = f'Tile northern edge [{bbox.position}]'
            global_J, global_i_slice = bbox.j1, bbox.icg_slice
            local_J_b, local_i_slice_b = bbox.J1cl, bbox.icl_slice
            local_J_t, local_i_slice_t = global_J, global_i_slice

        else: # Both "bottom" and "top" are tiles
            edgeloc = f'{bbox.position} and {tbox.position}'
            global_J_b, global_i_slice_b = bbox.j1, bbox.icg_slice
            global_J_t, global_i_slice_t = tbox.j0, tbox.icg_slice

            if global_J_b != global_J_t or global_i_slice_b != global_i_slice_t:
                raise ValueError( f"""{edgeloc} do not match. global_J_b={global_J_b}, global_J_t={global_J_t},
                                 global_i_slice_b={global_i_slice_b}, global_i_slice_t={global_i_slice_t}.""")

            global_J, global_i_slice = global_J_b, global_i_slice_b
            local_J_b, local_i_slice_b = bbox.J1cl, bbox.icl_slice
            local_J_t, local_i_slice_t = tbox.J0cl, tbox.icl_slice

        rfl_b = bottom.v_rfl[local_J_b, local_i_slice_b]
        rfl_t = top.v_rfl[local_J_t, local_i_slice_t]

        self.v_simple[global_J, global_i_slice] = match_edges(
            bottom.v_simple[local_J_b, local_i_slice_b], top.v_simple[local_J_t, local_i_slice_t], rfl_b, rfl_t,
            tolerance=tolerance, verbose=verbose, message=edgeloc+' (simple)'
        )
        self.v_rfl[global_J, global_i_slice] = np.maximum(rfl_b, rfl_t)

        if calc_effective:
            self.v_effective[global_J, global_i_slice] = match_edges(
                bottom.v_effective[local_J_b, local_i_slice_b], top.v_effective[local_J_t, local_i_slice_t], rfl_b, rfl_t,
                tolerance=tolerance, verbose=verbose, message=edgeloc+' (effective)'
            )

    def stitch_reentrant_x(self, tolerance, calc_effective=False, verbose=False):
        """
        Verify and reconcile the cyclic western/eastern edges
        """

        msg = 'zonal boundaries (simple)'
        self.u_simple[:, 0] = match_edges(
            self.u_simple[:, 0], self.u_simple[:, -1], self.u_rfl[:, 0], self.u_rfl[:, -1],
            tolerance=tolerance, verbose=verbose, message=msg
        )
        self.u_simple[:, -1] = self.u_simple[:, 0]

        if calc_effective:
            msg = 'zonal boundaries (effective)'
            self.u_effective[:, 0] = match_edges(
                self.u_effective[:, 0], self.u_effective[:, -1], self.u_rfl[:, 0], self.u_rfl[:, -1],
                tolerance=tolerance, verbose=verbose, message=msg
            )
            self.u_effective[:, -1] = self.u_effective[:, 0]

    def stitch_fold_n(self, tolerance, calc_effective=False, verbose=False):
        """
        Verify and reconcile the folding northern edge
        """

        assert self.ni % 2 == 0, "Doing folding northern boundary with an odd nj."
        ni_mid = self.ni // 2

        msg = 'northern boundary (simple)'
        self.v_simple[-1, :ni_mid] = match_edges(
            self.v_simple[-1, :ni_mid], self.v_simple[-1, ni_mid:][::-1], self.v_rfl[-1, :ni_mid], self.v_rfl[-1, ni_mid:][::-1],
            tolerance=tolerance, verbose=verbose, message=msg
        )
        self.v_simple[-1, ni_mid:] = self.v_simple[-1, :ni_mid][::-1]

        if calc_effective:
            msg = 'northern boundary (effective)'
            self.v_effective[-1, :ni_mid] = match_edges(
                self.v_effective[-1, :ni_mid], self.v_effective[-1, ni_mid:][::-1], self.v_rfl[-1, :ni_mid], self.v_rfl[-1, ni_mid:][::-1],
                tolerance=tolerance, verbose=verbose, message=msg
            )
            self.v_effective[-1, ni_mid:] = self.v_effective[-1, :ni_mid][::-1]

    def stitch_tiles(self, thinwalls_list, tolerance=0, config=CalcConfig(), verbose=True):
        """"
        Stitch all tiles
        """
        npj, npi = self.pelayout.shape

        # Put the list of ThinWalls on a 2D array to utilize np array's slicing
        tiles = np.empty( (npj, npi), dtype=object )
        for tw in thinwalls_list:
            tiles[tw.bbox.position] = tw

        if config.calc_roughness: self.roughness = np.zeros( self.shape )
        if config.calc_gradient: self.gradient = np.zeros( self.shape )

        # Insert all tiles
        for iy in range(npj):
            for ix in range(npi):
                self._stitch_tile( tiles[iy,ix], config=config )

        # Reconcile outer edges
        if config.calc_thinwalls:
            for iy in range(npj):
                for ix in range(npi-1):
                    self._stitch_i( tiles[iy, ix], tiles[iy, ix+1], tolerance, calc_effective=config.calc_effective_tw, verbose=verbose )
            for iy in range(npj-1):
                for ix in range(npi):
                    self._stitch_j( tiles[iy, ix], tiles[iy+1, ix], tolerance, calc_effective=config.calc_effective_tw, verbose=verbose )
            if self.reentrant_x:
                self.stitch_reentrant_x( tolerance=tolerance, calc_effective=config.calc_effective_tw, verbose=verbose )
            if self.fold_n:
                self.stitch_fold_n( tolerance=tolerance, calc_effective=config.calc_effective_tw, verbose=verbose )

    def stitch_mask(self, mask, config=CalcConfig(), tolerance=2, verbose=False):
        """
        Insert mask domain.
        The assumption is the mask domain has the higher refine level and always replace the old values.
        """
        self._stitch_tile(mask, config)

        if config.calc_thinwalls:
            # Western edge
            self._stitch_i( self, mask, tolerance=tolerance, calc_effective=False, verbose=verbose )
            # Eastern edge
            self._stitch_i( mask, self, tolerance=tolerance, calc_effective=False, verbose=verbose )
            # Southern edge
            self._stitch_j( self, mask, tolerance=tolerance, calc_effective=False, verbose=verbose )
            # Northern edge
            self._stitch_j( mask, self, tolerance=tolerance, calc_effective=False, verbose=verbose )

def match_edges(edge1, edge2, rfl1, rfl2, tolerance=0, verbose=True, message=''):
    """
    Check if two edges are identical and if not, return the proper one.

    Parameters
    ----------
    edge1, edge2 : ThinWalls.Stats object
        Edges at the same grid location from two sources (e.g. different tiles).
    rfl1, rfl2 : int
        Corresponding maximum refinement levels of the two edges.
    tolerance : int, optional
        Controls whether stopping the script or reconciling edges with different heights.
        0 : raise an exception and stop the script if there is difference.
        1 : differences between edges with different RFLs are tolerated. Use the heights
            of the edge with higher refinement level.
        2 : all differences are tolerated. If the two edges have the same RFL, max() is
            used to determine the final low, ave and hgh. Should be used for debug only.
    verbose : bool, optional
        If true, screen display the differences and how they are treated.
    message : str, optional
        Information (tile ID, measure) added to screen display.

    Output
        edge : ThinWalls.Stats object
    ----------
Source grid size
    """
    ndiff_hgh = (edge1.hgh!=edge2.hgh).sum()
    ndiff_ave = (edge1.ave!=edge2.ave).sum()
    ndiff_low = (edge1.hgh!=edge2.hgh).sum()

    str_diff = '[hgh: {:4d}, ave: {:4d}, low: {:4d}]'.format(ndiff_hgh, ndiff_ave, ndiff_low)
    if np.array(rfl1).size==1 and np.array(rfl2).size==1:
        str_rfls = '[rfl={:2d} vs rfl={:2d}]'.format(rfl1, rfl2)
    else:
        str_rfls = ''
    msg = ' '.join(['Edges differ', str_diff, ':', message, str_rfls])+'. '

    if np.array(rfl1).size==1:
        rfl1 = np.ones_like(edge1.hgh) * rfl1
    if np.array(rfl2).size==1:
        rfl2 = np.ones_like(edge2.hgh) * rfl2

    if ndiff_hgh+ndiff_ave+ndiff_low!=0:
        if tolerance==0:
            raise Exception(msg)
        if np.any(rfl1!=rfl2):
            if verbose:
                print(msg+'Use higher rfl')
            edge = ThinWalls.StatsBase(edge1.shape)
            edge.low = np.where(rfl1>rfl2, edge1.low, edge2.low)
            edge.ave = np.where(rfl1>rfl2, edge1.ave, edge2.ave)
            edge.hgh = np.where(rfl1>rfl2, edge1.hgh, edge2.hgh)
            return edge
        else:
            if tolerance==1:
                raise Exception(msg)
            if verbose:
                print(msg+'Use shallower depth')
            edge = ThinWalls.StatsBase(edge1.shape)
            edge.low = np.maximum(edge1.low, edge2.low)
            edge.ave = np.maximum(np.maximum(edge1.ave, edge2.ave), edge.low)
            edge.hgh = np.maximum(np.maximum(edge1.hgh, edge2.hgh), edge.ave)
            return edge
    else:
        if np.any(rfl1!=rfl2) and verbose: # This should hardly happen.
            print(message+' have the same edge but different refinement levels '+str_rfls+'.')
        return edge1
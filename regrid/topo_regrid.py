import sys
import numpy

from .external.thinwall.python import GMesh
from .external.thinwall.python import ThinWalls
from .tile_utils import slice_array, decompose_domain, normlize_longitude, box_halo, BoundaryBox, reverse_slice
from .kernel import CalcConfig
# from .north_pole import NorthPoleMask

class HitMap(GMesh.GMesh):
    """A container for hits on the source grid
    """
    def __init__(self, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)
        self._hits = numpy.zeros( self.shape )
        self._box = (0, self.shape[0], 0, self.shape[1])
    def __getitem__(self, key):
        return self._hits[key]
    def __setitem__(self, key, value):
        self._hits[key] = value
    @property
    def box(self):
        return self._box
    @box.setter
    def box(self, value):
        assert isinstance(value, tuple) and len(value)==4, "Box needs to be a 4-element tuple."
        # jst, jed, ist, ied = value
        # assert (jed-jst, ied-ist)==self.shape or , "Wrong box size."
        self._box = value
    def stitch_hits(self, hits_list):
        """Puts together hits from a list of subdomains"""
        for hits in hits_list:
            jst, jed, ist, ied = hits.box
            if ist<ied:
                self[jst:jed, ist:ied] += hits[:]
            else:
                self[jst:jed, :ied] += hits[:,-ied:]
                self[jst:jed, ist:] += hits[:,:-ied]
    def check_lat(self):
        """Checks if all points at each latitude are hit."""
        return numpy.all(self[:,:], axis=1)
    def pcolormesh(self, axis, **kwargs):
        """Plots hit map"""
        return axis.pcolormesh( self.lon, self.lat, self[:,:], **kwargs )

class NorthPoleMask:
    def __init__(self, domain, counts=0, radius=0.25):
        """
        Parameters
        ----------
        domain : Domain object
            The domain where the mask is to be found.
        counts : integer, optional
            Number of north poles in the target grid. E.g. there are two north poles in the bi-polar cap.
        radius : float
            The radius of the north pole region used to a) decide the mask of north pole in the target grid;
            b) ignore the hits in source grid
        verbose : bool
        """
        self.domain = domain
        self.counts = counts
        self.radius = radius
        self._masks = self._find_north_pole_rectangles()

    def __str__(self):
        disp = [
            f'North Pole rectangles (radius = {self.radius:5.2f}{chr(176):1s}) in {repr(self.domain)}'
        ]
        for box in self:
            idx_str = ','.join([f"{idx:d}" for idx in box])
            disp.append( f'  js,je,is,ie: {idx_str}, shape: ({box[1]-box[0]}, {box[3]-box[2]})' )
        return '\n'.join(disp)

    def __getitem__(self, index):
        return self._masks[index]

    def _find_north_pole_rectangles(self):
        """Returns the extented rectangles of the grids enclosed by a latitudinal circle
        Output:
        ----------
        recs : list of tuples
            Indices of rectangle boxes. Number of boxes depends on num_north_pole.
        """
        domain = self.domain
        jj, ii = numpy.where(domain.lat > (90.0 - self.radius))

        if jj.size==0 or ii.size==0 or self.counts==0:
            recs = []
        elif self.counts==1:
            recs = BoundaryBox( j0=jj.min(), j1=jj.max(), i0=ii.min(), i1=ii.max() )
        elif self.counts==2:
            jjw = jj[ii<domain.ni//2]; iiw = ii[ii<domain.ni//2]
            jje = jj[ii>domain.ni//2]; iie = ii[ii>domain.ni//2]
            assert numpy.all(jjw==jje), 'nj in the two mask domains mismatch.'
            jj = jjw
            assert (jjw.max()==domain.nj), 'mask domains do not reach the north boundary.'
            assert (iiw.min()+iie.max()==domain.ni) and ((iiw.max()+iie.min()==domain.ni)), \
                'ni in the two mask domains mismatch.'
            recs = [
                  BoundaryBox( j0=jj.min(), j1=jj.max(), i0=iiw.min(), i1=iiw.max() ),
                  BoundaryBox( j0=jj.min(), j1=jj.max(), i0=iie.min(), i1=iie.max() )
            ]
            # self.masks = [(jj.min(), 2*domain.nj-jj.min(), iiw.min(), iiw.max()),
            #                    (jj.min(), 2*domain.nj-jj.min(), iie.min(), iie.max())] # extend passing the northern boundary for halos
        else:
            raise Exception('Currently only two north pole rectangles are supported.')
        return recs

    def find_local_masks(self, bbox):
        """Finds where the north pole rectangles overlap with the subdomain"""
        masks = []
        jst, jed, ist, ied = bbox.jcg_slice.start, bbox.jcg_slice.stop, bbox.icg_slice.start, bbox.icg_slice.stop
        jsth, jedh, isth, iedh = bbox.jdg_slice.start, bbox.jdg_slice.stop, bbox.idg_slice.start, bbox.idg_slice.stop
        for mask_bbox in self:
            jstm, jedm, istm, iedm = mask_bbox.jcg_slice.start, mask_bbox.jcg_slice.stop, mask_bbox.icg_slice.start, mask_bbox.icg_slice.stop
            if (not ((istm>=iedh) or iedm<=isth)) and (not ((jstm>=jedh) or (jedm<=jsth))):
                # Relative indices
                j0, j1, i0, i1 = max(jstm,jsth)-jsth, min(jedm,jedh)-jsth, max(istm,isth)-isth, min(iedm,iedh)-isth
                # if mask boundary is beyond subdomain boundary but within halo, ignore halo
                if jstm<=jst:
                    j0 = 0
                if jedm>=jed:
                    j1 = jedh - jsth
                if istm<=ist:
                    i0 = 0
                if iedm>=ied:
                    i1 = iedh - isth
                # if jedm==self.nj and jed>self.nj: j1 = jed-jst
                # # The following addresses a very trivial case when the mask reaches
                # # the southern bounndary, which may only happen in tests.
                # if jstm==0 and jst<0: j0 = 0
                masks.append( BoundaryBox(j0, j1, i0, i1) )
        return masks

    def create_mask_domain(self, tgt_halo=0, norm_lon=True, pole_radius=0.25):
        """Creates a domain for the masked north pole region

        Parameters
        ----------
        tgt_halo : int, optional
            Halo size
        pole_radius : float, optional
            Polar radius in the new mask domain

        Output
        ----------
        mds : A list of Domain object
        """

        domain = self.domain
        mds = []
        for mask in self:
            mask_box = box_halo(mask, tgt_halo)

            lon = slice_array(domain.lon, box=mask_box, cyclic_zonal=False, fold_north=True)
            lat = slice_array(domain.lat, box=mask_box, cyclic_zonal=False, fold_north=True)
            if norm_lon: lon = normlize_longitude(lon, lat)

            Idx, Idy = None, None
            if domain.Idx is not None:
                Idx = slice_array(domain.Idx, box=mask_box, position='center', cyclic_zonal=False, fold_north=True)
            if domain.Idy is not None:
                Idy = slice_array(domain.Idy, box=mask_box, position='center', cyclic_zonal=False, fold_north=True)

            mds.append( Domain(lon=lon, lat=lat, Idx=Idx, Idy=Idy, reentrant_x=False, num_north_pole=1, pole_radius=pole_radius) )
        return mds

class Domain(ThinWalls.ThinWalls):
    """A container for regrided topography
    """
    def __init__(self, lon=None, lat=None, Idx=None, Idy=None, is_geo_coord=True,
                 reentrant_x=False, fold_n=False, bbox=None, num_north_pole=0, pole_radius=0.25, np_masks=[],
                 eds=None, subset_eds=False, src_halo=0):
        """
        Parameters
        ----------
        lon, lat : float
            Cell corner coordinates to construct ThinWalls
        Idx, Idy : float, optional
            v and u point grid spacing, used for calculating gradient.
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
        num_north_pole : integer, optional
            Number of north poles in the target grid. E.g. there are two north poles in the bi-polar cap.
        pole_radius : float
            The radius of the north pole region used to a) decide the mask of north pole in the target grid;
            b) ignore the hits in source grid
        eds : GMesh.UniformEDS objects, optional
            Contains source coordinates and data.
        subset_eds : bool, optional
            If true, source data eds to encompass target grid, with a halo decided by src_halo. Default is False.
        src_halo : integer, optional
            Halo size of the source grid in either direction.
        """
        super().__init__(lon=lon, lat=lat, is_geo_coord=is_geo_coord)
        self.Idx, self.Idy = Idx, Idy

        if bbox:
            assert (self.nj == bbox.data_nj) and (self.ni == bbox.data_ni), f'bbox incorrect {self.nj}!= {bbox.nj}, {self.ni}!= {bbox.ni}'
            self.bbox = bbox
        else:
            self.bbox = BoundaryBox(j0=0, j1=self.nj, i0=0, i1=self.ni, halo=0)

        self.reentrant_x = reentrant_x
        self.fold_n = fold_n
        if self.fold_n:
            assert self.ni%2==0, 'An odd number ni does not work with bi-polar cap.'

        if num_north_pole > 0:
            self.north_mask = NorthPoleMask(self, counts=num_north_pole, radius=pole_radius)
        else:
            self.north_mask = np_masks

        # self.pole_radius = pole_radius
        # self.north_mask = self.find_north_pole_rectangles(num_north_pole=num_north_pole)

        if eds: self._fit_src_coords(eds, subset_eds=subset_eds, halo=src_halo)

    def __str__(self):
        if self.is_geo_coord:
            coord_name = 'lat lon'
            lonmin, lonmax = numpy.mod(self.lon.min(), 360), numpy.mod(self.lon.max(), 360)
            src_lonmin, src_lonmax = numpy.mod(self.eds.lon_coord.bounds[0], 360), numpy.mod(self.eds.lon_coord.bounds[-1], 360)
        else:
            coord_name = 'y x'
            lonmin, lonmax = self.lon.min(), self.lon.max()
            src_lonmin, src_lonmax = self.eds.lon_coord.bounds[0], self.eds.lon_coord.bounds[-1]
        disp = [
            str(type(self)),
            str(self.bbox),
            f'Geographic coordinate? {self.is_geo_coord}',
            f'Domain size (nj ni): ({self.nj}, {self.ni})',
            f'Domain range ({coord_name}): [{self.lat.min():10.6f}, {self.lat.max():10.6f}], [{lonmin:10.6f}, {lonmax:10.6f}]'
            f'Source grid size (nj ni): ({self.eds.nj:9d}, {self.eds.ni:9d}) ' + \
            f'indices: {(self.eds.lat_coord.start, self.eds.lat_coord.stop, self.eds.lon_coord.start, self.eds.lon_coord.stop)}',
            f'Source grid range ({coord_name}): [{self.eds.lat_coord.bounds[0]:10.6f}, {self.eds.lat_coord.bounds[-1]:10.6f}], [{src_lonmin:10.6f}, {src_lonmax:10.6f}]'
        ]
        if hasattr(self, 'north_mask'): disp.append(str(self.north_mask))
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

    def create_subdomains(self, pelayout, tgt_halo=0, x_sym=True, y_sym=False, norm_lon=None, eds=None, subset_eds=True, src_halo=0,
                          verbose=False):
        """Creates a list of sub-domains with corresponding source lon, lat and elev sections.

        Parameters
        ----------
        pelayout : tuple of integers, (nj, ni)
            Number of sub-domains in each direction
        x_sym, y_sym : boo, optional
            Whether try to use symmetric domain decomposition in x or y.
            Default is x_sym=True and y_sym=False.
        eds : GMesh.UniformEDS object, optional
            Source coordinates and topography
        src_halo : integer, optional
            Halo size of the source grid in either direction.
        verbose : bool, optional

        Returns
        ----------
        self.pelayout : a 2D array documenting indices of each subdomain
        Out : ndarray
            A 2D array of RefineWrapper objects
        """
        if norm_lon is None: norm_lon = self.is_geo_coord

        if pelayout[1]==1 and tgt_halo>0:
            print('WARNING: only 1 subdomain in i-direction, which may not work with bi-polar cap.')
        if (not x_sym) and self.fold_n:
            print(('WARNING: domain decomposition is not guaranteed to be symmetric in x direction, ',
                   'which may not work with bi-polar cap.'))

        j_domain = decompose_domain(self.nj, pelayout[0], symmetric=y_sym)
        i_domain = decompose_domain(self.ni, pelayout[1], symmetric=x_sym)
        if verbose:
            print('Domain is decomposed to {:}. Halo size = {:d}.'.format(pelayout, tgt_halo))
            print('  i: ', i_domain)
            print('  j: ', j_domain)
            print('\n')

        chunks = numpy.empty((j_domain.size, i_domain.size), dtype=object)
        self.pelayout = numpy.empty((j_domain.size, i_domain.size), dtype=object)
        for pe_j, (jst, jed) in enumerate(j_domain):
            for pe_i, (ist, ied) in enumerate(i_domain):
                self.pelayout[pe_j, pe_i] = ((jst, jed, ist, ied), tgt_halo) # indices for cell centers

                bbox = BoundaryBox(jst, jed, ist, ied, tgt_halo, (pe_j, pe_i))
                lon = slice_array(self.lon, bbox=bbox, cyclic_zonal=self.reentrant_x, fold_north=self.fold_n)
                lat = slice_array(self.lat, bbox=bbox, cyclic_zonal=self.reentrant_x, fold_north=self.fold_n)
                if norm_lon: lon = normlize_longitude(lon, lat)

                masks = self.north_mask.find_local_masks(bbox)
                if self.Idx is None: Idx = None
                else: Idx = slice_array(self.Idx, bbox=bbox, position='center', cyclic_zonal=self.reentrant_x, fold_north=self.fold_n)
                if self.Idy is None: Idy = None
                else: Idy = slice_array(self.Idy, bbox=bbox, position='center', cyclic_zonal=self.reentrant_x, fold_north=self.fold_n)

                chunks[pe_j, pe_i] = Domain(
                    lon=lon, lat=lat, Idx=Idx, Idy=Idy, is_geo_coord=self.is_geo_coord,
                    reentrant_x=False, fold_n=False, bbox=bbox, np_masks=masks, eds=eds, subset_eds=subset_eds, src_halo=src_halo
                )

                if verbose:
                    print(chunks[pe_j, pe_i], '\n')
        return chunks

    def create_mask_domain(self, mask, tgt_halo=0, norm_lon=True, pole_radius=0.25):
        """Creates a domain for the masked north pole region

        Parameters
        ----------
        mask : tuple
            Tuple of mask rectangle indices
        tgt_halo : int, optional
            Halo size
        pole_radius : float, optional
            Polar radius in the new mask domain
        Returns
        ----------
        Output : Domain object
        """
        jst, jed, ist, ied = mask
        mask_halo = (jst-tgt_halo, jed+tgt_halo, ist-tgt_halo, ied+tgt_halo)
        lon = slice_array(self.lon, box=mask_halo, cyclic_zonal=False, fold_north=True)
        lat = slice_array(self.lat, box=mask_halo, cyclic_zonal=False, fold_north=True)
        if norm_lon: lon = normlize_longitude(lon, lat)

        Idx, Idy = None, None
        if self.Idx is not None:
            Idx = slice_array(self.Idx, box=mask_halo, position='center',
                              cyclic_zonal=False, fold_north=True)
        if self.Idy is not None:
            Idy = slice_array(self.Idy, box=mask_halo, position='center',
                              cyclic_zonal=False, fold_north=True)

        return Domain(lon=lon, lat=lat, Idx=Idx, Idy=Idy, reentrant_x=False, num_north_pole=1, pole_radius=pole_radius)

    def _stitch_center(self, tiles, config):
        """Stitch all cell-center and inner edges properties
        """
        npj, npi = tiles.shape

        if config.calc_roughness: self.roughness = numpy.zeros( self.shape )
        if config.calc_gradient: self.gradient = numpy.zeros( self.shape )

        for iy in range(npj):
            for ix in range(npi):

                this = tiles[iy,ix]
                jg_slice, ig_slice = this.bbox.jcg_slice, this.bbox.icg_slice
                jl_slice, il_slice = this.bbox.jcl_slice, this.bbox.icl_slice

                Jg_slice, Ig_slice = this.bbox.Jcg_inner_slice, this.bbox.Icg_inner_slice
                Jl_slice, Il_slice = this.bbox.Jcl_inner_slice, this.bbox.Icl_inner_slice

                self.c_simple[jg_slice, ig_slice] = this.c_simple[jl_slice, il_slice]
                self.c_rfl[jg_slice, ig_slice] = this.mrfl

                # inner edges
                if config.calc_thinwalls:
                    self.u_simple[jg_slice, Ig_slice] = this.u_simple[jl_slice, Il_slice]
                    self.v_simple[Jg_slice, ig_slice] = this.v_simple[Jl_slice, il_slice]
                    self.u_rfl[jg_slice, Ig_slice] = this.mrfl
                    self.v_rfl[Jg_slice, ig_slice] = this.mrfl

                    if config.calc_effective_tw:
                        self.c_effective[jg_slice, ig_slice] = this.c_effective[jl_slice, il_slice]
                        self.u_effective[jg_slice, Ig_slice] = this.u_effective[jl_slice, Il_slice]
                        self.v_effective[Jg_slice, ig_slice] = this.v_effective[Jl_slice, il_slice]

                if config.calc_roughness: self.roughness[jg_slice, ig_slice] = tiles[iy,ix].roughness[jl_slice, il_slice]
                if config.calc_gradient: self.gradient[jg_slice, ig_slice] = tiles[iy,ix].gradient[jl_slice, il_slice]

    def _stitch_i(self, tile_l, tile_r, tolerance, calc_effective=False, verbose=False):
        """Stitch edges properties in-between tiles along i
        """
        jg_slice, I1g, I0g = tile_l.bbox.jcg_slice, tile_l.bbox.I1cg, tile_r.bbox.I0cg
        jl_slice, I1l, I0l = tile_l.bbox.jcl_slice, tile_l.bbox.I1cl, tile_r.bbox.I0cl
        if I1g != I0g: raise ValueError(f"Left tile and right tile edges do not match {I1g}!={I0g}.")

        edgeloc = '{} and {}'.format(tile_l.id, tile_r.id)
        self.u_simple[jg_slice, I1g] = match_edges(tile_l.u_simple[jl_slice, I1l],
            tile_r.u_simple[jl_slice, I0l], tile_l.mrfl, tile_r.mrfl,
            tolerance=tolerance, verbose=verbose, message=edgeloc+' (simple)')
        self.u_rfl[jg_slice, I1g] = max(tile_l.mrfl, tile_r.mrfl)

        if calc_effective:
            self.u_effective[jg_slice, I1g] = match_edges(tile_l.u_effective[jl_slice, I1l],
                tile_r.u_effective[jl_slice, I0l], tile_l.mrfl, tile_r.mrfl,
                tolerance=tolerance, verbose=verbose, message=edgeloc+' (effective)')

    def _stitch_j(self, tile_d, tile_u, tolerance, calc_effective=False, verbose=False):
        """Stitch edges properties in-between tiles along j
        """
        ig_slice, J1g, J0g = tile_d.bbox.icg_slice, tile_d.bbox.J1cg, tile_u.bbox.J0cg
        il_slice, J1l, J0l = tile_d.bbox.icl_slice, tile_d.bbox.J1cl, tile_u.bbox.J0cl
        if J1g != J0g: raise ValueError(f"Down tile and up tile edges do not match {J1g}!={J0g}.")

        edgeloc = '{} and {}'.format(tile_d.id, tile_u.id)
        self.v_simple[J1g, ig_slice] = match_edges(tile_d.v_simple[J1l, il_slice],
            tile_u.v_simple[J0l, il_slice], tile_d.mrfl, tile_u.mrfl,
            tolerance=tolerance, verbose=verbose, message=edgeloc+' (simple)')
        self.v_rfl[J1g, ig_slice] = max(tile_d.mrfl, tile_u.mrfl)

        if calc_effective:
            self.v_effective[J1g, ig_slice] = match_edges(tile_d.v_effective[J1l, il_slice],
                tile_u.v_effective[J0l, il_slice], tile_d.mrfl, tile_u.mrfl,
                tolerance=tolerance, verbose=verbose, message=edgeloc+' (effective)')

    def _stitch_edges(self, tiles, calc_effective, tolerance):
        """A wrapper of _stitch_[ij]
        """
        npj, npi = tiles.shape

        for iy in range(npj):
            for ix in range(npi-1):
                self._stitch_i( tiles[iy, ix], tiles[iy, ix+1], tolerance, calc_effective=calc_effective )
        for iy in range(npj-1):
            for ix in range(npi):
                self._stitch_j( tiles[iy, ix], tiles[iy+1, ix], tolerance, calc_effective=calc_effective )

    def _stitch_reentrant_x(self, tiles, tolerance, calc_effective=False):
        """Stitch cyclic zonal edges
        """
        npj, _ = tiles.shape
        for iy in range(npj):
            self._stitch_i( tiles[iy, -1], tiles[iy, 0], tolerance, calc_effective=calc_effective )

    def _stitch_fold_n(self, tiles, tolerance, calc_effective=False, verbose=False):
        """Stitch the folding north edge
        """
        _, npi = tiles.shape
        for ix in range(npi//2):
            tile_d, tile_u = tiles[-1,ix], tiles[-1,npi-ix-1]

            ig_slice_d, ig_slice_u = tile_d.bbox.icg_slice, tile_u.bbox.icg_slice
            J1l_d, il_slice_d = tile_d.J1cl, tile_d.bbox.icl_slice
            J1l_u, il_slice_u = tile_u.J1cl, tile_u.bbox.icl_slice

            edgeloc = '{} and {}'.format(tile_d.id, tile_u.id)
            self.v_simple[-1, ig_slice_d] = match_edges(tile_d.v_simple[J1l_d, il_slice_d],
                tile_u.v_simple[J1l_u, il_slice_u][::-1], tile_d.mrfl, tile_u.mrfl,
                tolerance=tolerance, verbose=verbose, message=edgeloc+' (simple)')
            self.v_rfl[-1, ig_slice_d] = max(tile_d.mrfl, tile_u.mrfl)
            self.v_simple[-1, ig_slice_u] = self.v_simple[-1, ig_slice_d][::-1]
            self.v_rfl[-1, ig_slice_u] = self.v_rfl[-1, ig_slice_d][::-1]

            if calc_effective:
                self.v_effective[-1, ig_slice_d] = match_edges(tile_d.v_effective[J1l_d, il_slice_d],
                    tile_u.v_effective[J1l_u, il_slice_u][::-1], tile_d.mrfl, tile_u.mrfl,
                    tolerance=tolerance, verbose=verbose, message=edgeloc+' (effective)')
                self.v_effective[-1, ig_slice_u] = self.v_effective[-1, ig_slice_d][::-1]

        # probably don't need this ...
        if npi%2!=0:
            tile = tiles[-1, npi//2]

            I0cg, I1cg = tile.I0cg, tile.I1cg
            J1cl, I0cl, I1cl = tile.J1cl, tile.I0cl, tile.I1cl
            nhf = tile.ni // 2

            edgeloc = '{}'.format(tile.id)
            self.v_simple[-1, I0cg:I0cg+nhf] = match_edges(tile.v_simple[J1cl, I0cl:I0cl+nhf],
                tile.v_simple[J1cl, I0cl+nhf:I1cl][::-1], tile.mrfl, tile.mrfl,
                tolerance=tolerance, verbose=verbose, message=edgeloc+' (simple)')
            self.v_simple[-1, I0cg+nhf:I1cg] = self.v_simple[-1, I0cg:I0cg+nhf][::-1]
            self.v_rfl[-1, I0cg+nhf:I1cg] = tile.mrfl
            if calc_effective:
                self.v_effective[-1, I0cg:I0cg+nhf] = match_edges(tile.v_effective[J1cl, I0cl:I0cl+nhf],
                    tile.v_effective[J1cl, I0cl+nhf:I1cl][::-1], tile.mrfl, tile.mrfl,
                    tolerance=tolerance, verbose=verbose, message=edgeloc+' (effective)')
                self.v_effective[-1, I0cg+nhf:I1cg] = self.v_effective[-1, I0cg:I0cg+nhf][::-1]

    def stitch_subdomains(self, thinwalls_list, tolerance=0, config=CalcConfig(), verbose=True):
        """"Stitch subdomains
        """
        npj, npi = self.pelayout.shape

        # Put the list of ThinWalls on a 2D array to utilize numpy array's slicing
        tiles = numpy.empty( (npj, npi), dtype=object )
        for tw in thinwalls_list:
            tiles[tw.bbox.position] = tw

        self.c_rfl = numpy.zeros( self.shape, dtype=numpy.int32 )
        if config.calc_thinwalls:
            self.u_rfl = numpy.zeros( (self.shape[0],self.shape[1]+1), dtype=numpy.int32 )
            self.v_rfl = numpy.zeros( (self.shape[0]+1,self.shape[1]), dtype=numpy.int32 )

        self._stitch_center(tiles, config=config)
        if config.calc_thinwalls:
            self._stitch_edges(tiles, tolerance=tolerance, calc_effective=config.calc_effective_tw)
            if self.reentrant_x:
                self._stitch_reentrant_x(tiles, tolerance=tolerance, calc_effective=config.calc_effective_tw)
            if self.fold_n:
                self._stitch_fold_n(tiles, tolerance=tolerance, calc_effective=config.calc_effective_tw)

    def stitch_mask_domain(self, mask, rec, halo, config=CalcConfig(), tolerance=2, verbose=False):
        """
        The assumption is the masked domain has the higher refine level
        """
        jsg, jeg, isg, ieg = rec  # global indices
        # Cell center sizes
        nj, ni = jeg-jsg+2*halo, ieg-isg+2*halo
        jst, jet, ist, iet = halo, nj-halo, halo, ni-halo  # tile indices

        # aliasing
        dCs, dUs, dVs = self.c_simple, self.u_simple, self.v_simple
        mCs, mUs, mVs = mask.c_simple, mask.u_simple, mask.v_simple
        dCr, dUr, dVr = self.c_rfl, self.u_rfl, self.v_rfl
        mCr, mUr, mVr = mask.c_rfl, mask.u_rfl, mask.v_rfl
        if config.calc_effective_tw:
            dCe, dUe, dVe = self.c_effective, self.u_effective, self.v_effective
            mCe, mUe, mVe = mask.c_effective, mask.u_effective, mask.v_effective

        # assert numpy.all(dCr[jsg:jeg,isg:ieg]<=mCr[jst:jet,ist:iet]), \
        #     'Mask refinement level lower than parent domain.'

        # middle part
        dCs[jsg:jeg,isg:ieg] = mCs[jst:jet,ist:iet]
        dCr[jsg:jeg,isg:ieg] = mCr[jst:jet,ist:iet]

        if config.calc_thinwalls:
            dUs[jsg:jeg,isg+1:ieg] = mUs[jst:jet,ist+1:iet]
            dUr[jsg:jeg,isg+1:ieg] = mUr[jst:jet,ist+1:iet]
            dVs[jsg+1:jeg,isg:ieg] = mVs[jst+1:jet,ist:iet]
            dVr[jsg+1:jeg,isg:ieg] = mVr[jst+1:jet,ist:iet]

            msg = 'NP mask western edge (simple)'
            dUs[jsg:jeg,isg] = match_edges(dUs[jsg:jeg,isg], mUs[jst:jet,ist],
                                           dUr[jsg:jeg,isg], mUr[jst:jet,ist],
                                           tolerance=tolerance, verbose=verbose, message=msg)
            msg = 'NP mask eastern edge (simple)'
            dUs[jsg:jeg,ieg] = match_edges(dUs[jsg:jeg,ieg], mUs[jst:jet,iet],
                                           dUr[jsg:jeg,ieg], mUr[jst:jet,iet],
                                           tolerance=tolerance, verbose=verbose, message=msg)
            dUr[jsg:jeg,isg] = numpy.maximum(dUr[jsg:jeg,isg], mUr[jst:jet,ist])
            dUr[jsg:jeg,ieg] = numpy.maximum(dUr[jsg:jeg,ieg], mUr[jst:jet,iet])

            msg = 'NP mask northern edge (simple)'
            dVs[jsg,isg:ieg] = match_edges(dVs[jsg,isg:ieg], mVs[jst,ist:iet],
                                           dVr[jsg,isg:ieg], mVr[jst,ist:iet],
                                           tolerance=tolerance, verbose=verbose, message=msg)
            msg = 'NP mask southern edge (simple)'
            dVs[jeg,isg:ieg] = match_edges(dVs[jeg,isg:ieg], mVs[jet,ist:iet],
                                           dVr[jeg,isg:ieg], mVr[jet,ist:iet],
                                           tolerance=tolerance, verbose=verbose, message=msg)
            dVr[jsg,isg:ieg] = numpy.maximum(dVr[jsg,isg:ieg], mVr[jst,ist:iet])
            dVr[jeg,isg:ieg] = numpy.maximum(dVr[jeg,isg:ieg], mVr[jet,ist:iet])

            if config.calc_effective_tw:
                dCe[jsg:jeg,isg:ieg] = mCe[jst:jet,ist:iet]
                dUe[jsg:jeg,isg+1:ieg] = mUe[jst:jet,ist+1:iet]
                dVe[jsg+1:jeg,isg:ieg] = mVe[jst+1:jet,ist:iet]

                msg = 'NP mask W edge (effective)'
                dUe[jsg:jeg,isg] = match_edges(dUe[jsg:jeg,isg], mUe[jst:jet,ist],
                                                dUr[jsg:jeg,isg], mUr[jst:jet,ist],
                                               tolerance=tolerance, verbose=verbose, message=msg)
                msg = 'NP mask E edge (effective)'
                dUe[jsg:jeg,ieg] = match_edges(dUe[jsg:jeg,ieg], mUe[jst:jet,iet],
                                               dUr[jsg:jeg,ieg], mUr[jst:jet,iet],
                                               tolerance=tolerance, verbose=verbose, message=msg)
                msg = 'NP mask N edge (effective)'
                dVe[jsg,isg:ieg] = match_edges(dVe[jsg,isg:ieg], mVe[jst,ist:iet],
                                               dVr[jsg,isg:ieg], mVr[jst,ist:iet],
                                               tolerance=tolerance, verbose=verbose, message=msg)
                msg = 'NP mask S edge (effective)'
                dVe[jeg,isg:ieg] = match_edges(dVe[jeg,isg:ieg], mVe[jet,ist:iet],
                                               dVr[jeg,isg:ieg], mVr[jet,ist:iet],
                                               tolerance=tolerance, verbose=verbose, message=msg)

        if config.calc_roughness:
            self.roughness[jsg:jeg,isg:ieg] = mask.roughness[jst:jet,ist:iet]
        if config.calc_gradient:
            self.gradient[jsg:jeg,isg:ieg] = mask.gradient[jst:jet,ist:iet]

    def stitch_mask_fold_north(self, do_effective=False, tolerance=2, verbose=False):
        if not self.fold_n:
            return
        _, _, isw, iew = self.north_mask[0]
        _, _, ise, iee = self.north_mask[1]

        Vs, Vr = self.v_simple, self.v_rfl
        if do_effective: Ve = self.v_effective

        msg = 'northern boundary (simple)'
        Vs[-1,isw:iew] = match_edges(Vs[-1,isw:iew], Vs[-1,iee-1:ise-1:-1],
                                     Vr[-1,isw:iew], Vr[-1,iee-1:ise-1:-1],
                                     tolerance=tolerance, verbose=verbose, message=msg)
        Vr[-1,isw:iew] = numpy.maximum(Vr[-1,isw:iew], Vr[-1,iee-1:ise-1:-1])
        Vs[-1,iee-1:ise-1:-1], Vr[-1,iee-1:ise-1:-1] = Vs[-1,isw:iew], Vr[-1,isw:iew]
        if do_effective:
            msg = 'northern boundary (effective)'
            Ve[-1,isw:iew] = match_edges(Ve[-1,isw:iew], Ve[-1,iee-1:ise-1:-1],
                                         Vr[-1,isw:iew], Vr[-1,iee-1:ise-1:-1],
                                         tolerance=tolerance, verbose=verbose, message=msg)
            Ve[-1,iee-1:ise-1:-1] = Ve[-1,isw:iew]

    def regrid_topography(self, pelayout=None, tgt_halo=0, nprocs=1, eds=None, src_halo=0,
                          refine_loop_args={}, calc_args={}, hitmap=None, bnd_tol_level=1, verbose=False):
        """"A wrapper for getting elevation from a domain"""
        subdomains = self.create_subdomains(pelayout, tgt_halo=tgt_halo, eds=eds, src_halo=src_halo,
                                            refine_loop_args=refine_loop_args, verbose=verbose)

        topo_gen_args = calc_args.copy()
        topo_gen_args.update({'save_hits': not (hitmap is None), 'verbose': verbose})

        if nprocs>1:
            twlist = topo_gen_mp(subdomains.flatten(), nprocs=nprocs, topo_gen_args=topo_gen_args)
        else:
            twlist = [topo_gen(dm, **topo_gen_args) for dm in subdomains.flatten()]

        if topo_gen_args['save_hits']:
            twlist, hitlist = zip(*twlist)
            hitmap.stitch_hits(hitlist)

        self.stitch_subdomains(twlist, tolerance=bnd_tol_level, verbose=verbose, **calc_args)

    def regrid_topography_masked(self, lat_start=None, lat_end=89.75, lat_step=0.5,
                                 pelayout=None, tgt_halo=0, nprocs=1, eds=None, src_halo=0,
                                 refine_loop_args={}, calc_args={}, hitmap=None, bnd_tol_level=1,
                                 verbose=True):

        if lat_start is None: lat_start = 90.0 - self.pole_radius
        latc = lat_start + lat_step
        north_masks = self.north_mask
        while latc<=lat_end:
            print(latc)
            mask_domains = north_masks.create_mask_domain(tgt_halo=tgt_halo, pole_radius=90.0-latc)
            for ii, mask_domain in enumerate(mask_domains):
                refine_loop_args['singularity_radius'] = mask_domain.pole_radius
                mask_domain.regrid_topography(pelayout=pelayout, tgt_halo=tgt_halo, nprocs=nprocs, eds=eds, src_halo=src_halo,
                                              refine_loop_args=refine_loop_args, calc_args=calc_args, hitmap=hitmap,
                                              bnd_tol_level=bnd_tol_level)
                self.stitch_mask_domain(mask_domain, north_masks[ii], tgt_halo, tolerance=bnd_tol_level, **calc_args)
            self.stitch_mask_fold_north(tolerance=bnd_tol_level, do_effective=calc_args['do_effective'])
            north_masks = NorthPoleMask(radius=latc, counts=2)
            latc += lat_step

def match_edges(edge1, edge2, rfl1, rfl2, tolerance=0, verbose=True, message=''):
    """Check if two edges are identical and if not, return the proper one.

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

    """
    ndiff_hgh = (edge1.hgh!=edge2.hgh).sum()
    ndiff_ave = (edge1.ave!=edge2.ave).sum()
    ndiff_low = (edge1.hgh!=edge2.hgh).sum()

    str_diff = '[hgh: {:4d}, ave: {:4d}, low: {:4d}]'.format(ndiff_hgh, ndiff_ave, ndiff_low)
    if numpy.array(rfl1).size==1 and numpy.array(rfl2).size==1:
        str_rfls = '[rfl={:2d} vs rfl={:2d}]'.format(rfl1, rfl2)
    else:
        str_rfls = ''
    msg = ' '.join(['Edges differ', str_diff, ':', message, str_rfls])+'. '

    if numpy.array(rfl1).size==1:
        rfl1 = numpy.ones_like(edge1.hgh) * rfl1
    if numpy.array(rfl2).size==1:
        rfl2 = numpy.ones_like(edge2.hgh) * rfl2

    if ndiff_hgh+ndiff_ave+ndiff_low!=0:
        if tolerance==0:
            raise Exception(msg)
        if numpy.any(rfl1!=rfl2):
            if verbose:
                print(msg+'Use higher rfl')
            edge = ThinWalls.StatsBase(edge1.shape)
            edge.low = numpy.where(rfl1>rfl2, edge1.low, edge2.low)
            edge.ave = numpy.where(rfl1>rfl2, edge1.ave, edge2.ave)
            edge.hgh = numpy.where(rfl1>rfl2, edge1.hgh, edge2.hgh)
            return edge
        else:
            if tolerance==1:
                raise Exception(msg)
            if verbose:
                print(msg+'Use shallower depth')
            edge = ThinWalls.StatsBase(edge1.shape)
            edge.low = numpy.maximum(edge1.low, edge2.low)
            edge.ave = numpy.maximum(numpy.maximum(edge1.ave, edge2.ave), edge.low)
            edge.hgh = numpy.maximum(numpy.maximum(edge1.hgh, edge2.hgh), edge.ave)
            return edge
    else:
        if numpy.any(rfl1!=rfl2) and verbose: # This should hardly happen.
            print(message+' have the same edge but different refinement levels '+str_rfls+'.')
        return edge1
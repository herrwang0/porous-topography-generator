import sys
import numpy
import multiprocessing
import functools
from .external.thinwall.python import GMesh
from .external.thinwall.python import ThinWalls
from .roughness import subgrid_roughness_gradient
from .tile_utils import slice_array, decompose_domain, normlize_longitude, box_halo
from .output_utils import TimeLog
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

class RefineWrapper(GMesh.GMesh):
    """A wrapper for grid refinement (of a subdomain)

    Object from this class encapsules source grid, elevation and arguments for refine_loop method.
    It can be generated as a subdomain of a full domain grid.

    print(self) offers a detailed overlook of the target and source grid information.
    """
    def __init__(self, lon=None, lat=None, id=(0,0), is_geo_coord=True,
                 eds=None, subset_eds=False, src_halo=0, mask_recs=[], refine_loop_args={}):
        """
        Parameters
        ----------
        lon, lat : array of float
            Cell corner coordinates
        id : tuple, optional
            An identifier for the current subdomain. Used for easily linking subdomain to the parent domain.
        eds : GMesh.UniformEDS objects, optional
            Contains source coordinates and data.
        subset_eds : bool, optional
            If true, source data eds to encompass target grid, with a halo decided by src_halo. Default is False.
        src_halo : integer, optional
            Halo size of the source grid in either direction.
        mask_res : list, optional
            List of rectangle indices of masks. Used to mask out maximum resolution near the north poles.
        refine_loop_args : dict, optional
            A dictionary storing arguments for refine_loop methods, possible items include:
            fixed_refine_level : integer, optional
                If non-negative, the refine loop will only exit after reaching a certain level.
            max_mb : float, optional
                Memory limit for both parent and subdomains. Can be superceded when creating subdomains.
            refine_in_3d : bool, optional
                If true, interpolate coordinates with great circle distance when refine.
            use_resolution_limit : bool
                If true, exit the refine loop based on the coarsest resolution of the target grid
            pole_radius : float
                The radius of the north pole region used to a) decide the mask of north pole in the target grid;
                b) ignore the hits in source grid
            verbose : bool, optional
                Verbose opition for GMesh.refine_loop()
        """
        super().__init__(lon=lon, lat=lat, is_geo_coord=is_geo_coord)
        self.id = id
        self.north_masks = mask_recs
        self.refine_loop_args = refine_loop_args
        self._fit_src_coords(eds, subset_eds=subset_eds, halo=src_halo)

    def __str__(self):
        if self.is_geo_coord:
            coord_name = 'lat lon'
            tgt_lonmin, tgt_lonmax = numpy.mod(self.lon.min(), 360), numpy.mod(self.lon.max(), 360)
            src_lonmin, src_lonmax = numpy.mod(self.eds.lon_coord.bounds[0], 360), numpy.mod(self.eds.lon_coord.bounds[-1], 360)
        else:
            coord_name = 'y x'
            tgt_lonmin, tgt_lonmax = self.lon.min(), self.lon.max()
            src_lonmin, src_lonmax = self.eds.lon_coord.bounds[0], self.eds.lon_coord.bounds[-1]
        disp = [
            str(type(self)),
            f'Sub-domain identifier: {self.id}',
            f'Geographic coordinate? {self.is_geo_coord}',
            f'Target grid size (nj ni): ({self.nj}, {self.ni})',
            f'Source grid size (nj ni): ({self.eds.nj:9d}, {self.eds.ni:9d}) ' + \
            f'indices: {(self.eds.lat_coord.start, self.eds.lat_coord.stop, self.eds.lon_coord.start, self.eds.lon_coord.stop)}',
            f'Target grid range ({coord_name}): [{self.lat.min():10.6f}, {self.lat.max():10.6f}], [{tgt_lonmin:10.6f}, {tgt_lonmax:10.6f}]',
            f'Source grid range ({coord_name}): [{self.eds.lat_coord.bounds[0]:10.6f}, {self.eds.lat_coord.bounds[-1]:10.6f}], [{src_lonmin:10.6f}, {src_lonmax:10.6f}]'
        ]
        if len(self.north_masks)>0:
            disp.append('North Pole rectangles: ')
            for box in self.north_masks:
                disp.append('  js,je,is,ie: %s, shape: (%i,%i)'%(box, box[1]-box[0], box[3]-box[2]))
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

    def refine_loop(self, verbose=False, timers=False):
        """A self-contained version of GMesh.refine_loop()"""
        return super().refine_loop(self.eds, mask_res=self.north_masks, verbose=verbose, timers=timers, **self.refine_loop_args)

class Domain(ThinWalls.ThinWalls):
    """A container for regrided topography
    """
    def __init__(self, lon=None, lat=None, Idx=None, Idy=None, is_geo_coord=True, reentrant_x=False, fold_n=False, num_north_pole=0, pole_radius=0.25):
        """
        Parameters
        ----------
        lon, lat : float
            Cell corner coordinates to construct ThinWalls
        reentrant_x : bool, optional
            If true, the domain is reentrant in x-direction. Used for halos and assigning depth to the westernmost and easternmost u-edges.
            Default is False.
        fold_n : bool, optional
            If true, the domain is folded at the northern boundary. Used for halos and assigning depth to the northernmost v-edges.
            Default is False.
        num_north_pole : integer, optional
            Number of north poles in the target grid. E.g. there are two north poles in the bi-polar cap.
        pole_radius : float
            The radius of the north pole region used to a) decide the mask of north pole in the target grid;
            b) ignore the hits in source grid
        verbose : bool
        """
        super().__init__(lon=lon, lat=lat, is_geo_coord=is_geo_coord)
        self.Idx, self.Idy = Idx, Idy

        self.reentrant_x = reentrant_x
        self.fold_n = fold_n
        if self.fold_n:
            assert self.ni%2==0, 'An odd number ni does not work with bi-polar cap.'
            num_north_pole=2

        if num_north_pole > 0:
            self.north_mask = NorthPoleMask(self, counts=num_north_pole, radius=pole_radius)

        # self.pole_radius = pole_radius
        # self.north_mask = self.find_north_pole_rectangles(num_north_pole=num_north_pole)

    def __str__(self):
        if self.is_geo_coord:
            coord_name = 'lat lon'
            lonmin, lonmax = numpy.mod(self.lon.min(), 360), numpy.mod(self.lon.max(), 360)
        else:
            coord_name = 'y x'
            lonmin, lonmax = self.lon.min(), self.lon.max()
        disp = [
            str(type(self)),
            f'Geographic coordinate? {self.is_geo_coord}',
            f'Domain size (nj ni): ({self.nj}, {self.ni})',
            f'Domain range ({coord_name}): [{self.lat.min():10.6f}, {self.lat.max():10.6f}], [{lonmin:10.6f}, {lonmax:10.6f}]'
        ]
        if hasattr(self, 'north_mask'): disp.append(str(self.north_mask))
        return '\n'.join(disp)

    def create_subdomains(self, pelayout, tgt_halo=0, x_sym=True, y_sym=False, norm_lon=None, eds=None, subset_eds=True, src_halo=0,
                          refine_loop_args={}, verbose=False):
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
        refine_loop_args : dict, optional
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

                box_data = (jst-tgt_halo, jed+tgt_halo, ist-tgt_halo, ied+tgt_halo)
                lon = slice_array(self.lon, box=box_data, cyclic_zonal=self.reentrant_x, fold_north=self.fold_n)
                lat = slice_array(self.lat, box=box_data, cyclic_zonal=self.reentrant_x, fold_north=self.fold_n)
                if norm_lon: lon = normlize_longitude(lon, lat)

                masks = self.north_mask.find_local_masks((jst, jed, ist, ied), tgt_halo)
                chunks[pe_j, pe_i] = RefineWrapper(lon=lon, lat=lat, id=(pe_j, pe_i), is_geo_coord=self.is_geo_coord,
                                                   eds=eds, subset_eds=subset_eds, src_halo=src_halo,
                                                   mask_recs=masks, refine_loop_args=refine_loop_args)
                if self.Idx is not None:
                    Idx = slice_array(self.Idx, box=box_data, position='center',
                                      cyclic_zonal=self.reentrant_x, fold_north=self.fold_n)
                else:
                    Idx = None
                if self.Idy is not None:
                    Idy = slice_array(self.Idy, box=box_data, position='center',
                                      cyclic_zonal=self.reentrant_x, fold_north=self.fold_n)
                else:
                    Idy = None
                chunks[pe_j, pe_i].Idx, chunks[pe_j, pe_i].Idy = Idx, Idy

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

    def stitch_subdomains(self, thinwalls_list, tolerance=0, do_thinwalls=True, do_effective=True,
                          do_roughness=False, do_gradient=False, verbose=True):
        """"Stitch subdomains
        """
        do_effective = do_effective and do_thinwalls
        npj, npi = self.pelayout.shape

        # Put the list of ThinWalls on a 2D array to utilize numpy array's slicing
        tiles = numpy.empty( (npj, npi), dtype=object )
        for tw in thinwalls_list:
            tiles[tw.id] = tw

        self.c_rfl = numpy.zeros( self.shape, dtype=numpy.int32 )
        if do_thinwalls:
            self.u_rfl = numpy.zeros( (self.shape[0],self.shape[1]+1), dtype=numpy.int32 )
            self.v_rfl = numpy.zeros( (self.shape[0]+1,self.shape[1]), dtype=numpy.int32 )

        for iy in range(npj):
            for ix in range(npi):
                this = tiles[iy,ix]
                (jsg, jeg, isg, ieg), halo = self.pelayout[iy, ix] # global indices
                nj, ni = tiles[iy, ix].shape
                jst, jet, ist, iet = halo, nj-halo, halo, ni-halo # tile indices

                self.c_simple[jsg:jeg,isg:ieg] = this.c_simple[jst:jet,ist:iet]
                self.c_rfl[jsg:jeg,isg:ieg] = this.mrfl

                if not do_thinwalls:
                    continue

                self.u_simple[jsg:jeg,isg:ieg+1] = this.u_simple[jst:jet,ist:iet+1]
                self.u_rfl[jsg:jeg,isg:ieg+1] = this.mrfl
                self.v_simple[jsg:jeg+1,isg:ieg] = this.v_simple[jst:jet+1,ist:iet]
                self.v_rfl[jsg:jeg+1,isg:ieg] = this.mrfl

                if do_effective:
                    self.c_effective[jsg:jeg,isg:ieg] = this.c_effective[jst:jet,ist:iet]
                    self.u_effective[jsg:jeg,isg:ieg+1] = this.u_effective[jst:jet,ist:iet+1]
                    self.v_effective[jsg:jeg+1,isg:ieg] = this.v_effective[jst:jet+1,ist:iet]

                if (ix<npi-1) or (ix==npi-1 and self.reentrant_x):
                    if ix<npi-1:
                        TR = tiles[iy,ix+1]
                    else:
                        TR = tiles[iy,0]
                    edgeloc = '{} and {}'.format(this.id, TR.id)
                    self.u_simple[jsg:jeg, ieg] = match_edges(this.u_simple[jst:jet,iet],
                        TR.u_simple[jst:jet,ist], this.mrfl, TR.mrfl,
                        tolerance=tolerance, verbose=verbose, message=edgeloc+' (simple)')
                    self.u_rfl[jsg:jeg, ieg] = max(this.mrfl, TR.mrfl)
                    if do_effective:
                        self.u_effective[jsg:jeg, ieg] = match_edges(this.u_effective[jst:jet,iet],
                            TR.u_effective[jst:jet,ist], this.mrfl, TR.mrfl,
                            tolerance=tolerance, verbose=verbose, message=edgeloc+' (effective)')
                if iy<npj-1:
                    TU = tiles[iy+1,ix]
                    edgeloc = '{} and {}'.format(this.id, TU.id)
                    self.v_simple[jeg,isg:ieg] = match_edges(this.v_simple[jet,ist:iet],
                        TU.v_simple[jst,ist:iet], this.mrfl, TU.mrfl,
                        tolerance=tolerance, verbose=verbose, message=edgeloc+' (simple)')
                    self.v_rfl[jeg,isg:ieg] = max(this.mrfl, TU.mrfl)
                    if do_effective:
                        self.v_effective[jeg,isg:ieg] = match_edges(this.v_effective[jet,ist:iet],
                            TU.v_effective[jst,ist:iet], this.mrfl, TU.mrfl,
                            tolerance=tolerance, verbose=verbose, message=edgeloc+' (effective)')
        # Cyclic/folding boundaries
        if do_thinwalls:
            if self.reentrant_x:
                self.u_simple[:,0], self.u_rfl[:,0] = self.u_simple[:,-1], self.u_rfl[:,-1]
                if do_effective:
                    self.u_effective[:,0] = self.u_effective[:,-1]

            if self.fold_n:
                for ix in range(npi//2):
                    (_, _, isg, ieg), halo = self.pelayout[-1,ix]
                    (_, _, isgf, iegf), _ = self.pelayout[-1,npi-ix-1]

                    this, TU = tiles[-1,ix], tiles[-1,npi-ix-1]
                    edgeloc = '{} and {}'.format(this.id, TU.id)
                    (nj1, ni1), (nj2, ni2) = this.shape, TU.shape
                    jet1, ist1, iet1 = nj1-halo, halo, ni1-halo
                    jet2, ist2, iet2 = nj2-halo, -ni2+halo-1, ni2-halo-1

                    self.v_simple[-1,isg:ieg] = match_edges(this.v_simple[jet1,ist1:iet1],
                        TU.v_simple[jet2,iet2:ist2:-1], this.mrfl, TU.mrfl,
                        tolerance=tolerance, verbose=verbose, message=edgeloc+' (simple)')
                    self.v_rfl[-1,isg:ieg] = max(this.mrfl, TU.mrfl)
                    self.v_simple[-1,isgf:iegf] = self.v_simple[-1,isg:ieg][::-1]
                    self.v_rfl[-1,isgf:iegf] = self.v_rfl[-1,isg:ieg][::-1]

                    if do_effective:
                        self.v_effective[-1,isg:ieg] = match_edges(this.v_effective[jet1,ist1:iet1],
                            TU.v_effective[jet2,iet2:ist2:-1], this.mrfl, TU.mrfl,
                            tolerance=tolerance, verbose=verbose, message=edgeloc+' (effective)')
                        self.v_effective[-1,isgf:iegf] = self.v_effective[-1,isg:ieg][::-1]

                if npi%2!=0:
                    (_, _, ist, ied), halo = self.pelayout[-1,npi//2]
                    this = tiles[-1,npi//2]
                    nj, ni = this.shape
                    nhf = (ied-ist)//2
                    edgeloc = '{}'.format(this.id)
                    self.v_simple[-1,ist:ist+nhf] = match_edges(this.v_simple[nj-halo,halo:halo+nhf],
                        this.v_simple[nj-halo,halo+nhf:ni-halo:][::-1], this.mrfl, this.mrfl,
                        tolerance=tolerance, verbose=verbose, message=edgeloc+' (simple)')
                    self.v_simple[-1,ist+nhf:ied] = self.v_simple[-1,ist:ist+nhf][::-1]
                    self.v_rfl[-1,ist:ied] = this.mrfl
                    if do_effective:
                        self.v_effective[-1,ist:ist+nhf] = match_edges(this.v_effective[nj-halo,halo:halo+nhf],
                            this.v_effective[nj-halo,halo+nhf:ni-halo:][::-1], this.mrfl, this.mrfl,
                            tolerance=tolerance, verbose=verbose, message=edgeloc+' (effective)')
                        self.v_effective[-1,ist+nhf:ied] = self.v_effective[-1, ist:ist+nhf][::-1]

        if do_roughness:
            self.roughness = numpy.zeros( self.shape )
            for iy in range(npj):
                for ix in range(npi):
                    (jst, jed, ist, ied), halo = self.pelayout[iy, ix]
                    nj, ni = tiles[iy,ix].shape
                    self.roughness[jst:jed,ist:ied] = tiles[iy,ix].roughness[halo:nj-halo,halo:ni-halo]
        if do_gradient:
            self.gradient = numpy.zeros( self.shape )
            for iy in range(npj):
                for ix in range(npi):
                    (jst, jed, ist, ied), halo = self.pelayout[iy, ix]
                    nj, ni = tiles[iy,ix].shape
                    self.gradient[jst:jed,ist:ied] = tiles[iy,ix].gradient[halo:nj-halo,halo:ni-halo]

    def stitch_mask_domain(self, mask, rec, halo, do_thinwalls=True, do_effective=True,
                           do_roughness=False, do_gradient=False, tolerance=2, verbose=False):
        """
        The assumption is the masked domain has the higher refine level
        """
        do_effective = do_effective and do_thinwalls
        jsg, jeg, isg, ieg = rec  # global indices
        # Cell center sizes
        nj, ni = jeg-jsg+2*halo, ieg-isg+2*halo
        jst, jet, ist, iet = halo, nj-halo, halo, ni-halo  # tile indices

        # aliasing
        dCs, dUs, dVs = self.c_simple, self.u_simple, self.v_simple
        mCs, mUs, mVs = mask.c_simple, mask.u_simple, mask.v_simple
        dCr, dUr, dVr = self.c_rfl, self.u_rfl, self.v_rfl
        mCr, mUr, mVr = mask.c_rfl, mask.u_rfl, mask.v_rfl
        if do_effective:
            dCe, dUe, dVe = self.c_effective, self.u_effective, self.v_effective
            mCe, mUe, mVe = mask.c_effective, mask.u_effective, mask.v_effective

        # assert numpy.all(dCr[jsg:jeg,isg:ieg]<=mCr[jst:jet,ist:iet]), \
        #     'Mask refinement level lower than parent domain.'

        # middle part
        dCs[jsg:jeg,isg:ieg] = mCs[jst:jet,ist:iet]
        dCr[jsg:jeg,isg:ieg] = mCr[jst:jet,ist:iet]

        if do_thinwalls:
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

            if do_effective:
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

        if do_roughness:
            self.roughness[jsg:jeg,isg:ieg] = mask.roughness[jst:jet,ist:iet]
        if do_gradient:
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
            recs = [(jj.min(), jj.max(), ii.min(), ii.max())]
        elif self.counts==2:
            jjw = jj[ii<domain.ni//2]; iiw = ii[ii<domain.ni//2]
            jje = jj[ii>domain.ni//2]; iie = ii[ii>domain.ni//2]
            assert numpy.all(jjw==jje), 'nj in the two mask domains mismatch.'
            jj = jjw
            assert (jjw.max()==domain.nj), 'mask domains do not reach the north boundary.'
            assert (iiw.min()+iie.max()==domain.ni) and ((iiw.max()+iie.min()==domain.ni)), \
                'ni in the two mask domains mismatch.'
            recs = [(jj.min(), jj.max(), iiw.min(), iiw.max()),
                    (jj.min(), jj.max(), iie.min(), iie.max())]
            # self.masks = [(jj.min(), 2*domain.nj-jj.min(), iiw.min(), iiw.max()),
            #                    (jj.min(), 2*domain.nj-jj.min(), iie.min(), iie.max())] # extend passing the northern boundary for halos
        else:
            raise Exception('Currently only two north pole rectangles are supported.')
        return recs

    def find_local_masks(self, box, halo):
        """Finds where the north pole rectangles overlap with the subdomain"""
        masks = []
        jst, jed, ist, ied = box
        jsth, jedh, isth, iedh = jst-halo, jed+halo, ist-halo, ied+halo
        for jstm, jedm, istm, iedm in self:
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
                masks.append((j0, j1, i0, i1))
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

def topo_gen(grid, do_roughness=False, do_gradient=False, do_thinwalls=False, tw_interp='max', do_effective=True, save_hits=True, verbose=True, timers=False):
    """The main function for generating topography

    Parameters
    ----------
    grid : RefineWrapper object
        Contains all setting parameters needed for GMesh.refine_loop()

    Returns
    ----------
    tw : ThinWalls.ThinWalls object
    """

    if timers: clock = TimeLog(['setup topo_gen', 'refine grid', 'assign data', 'init thinwalls', 'roughness/gradient', 'total',
                                'refine loop (total)', 'refine loop (effective thinwalls)', 'refine loop (coarsen grid)'])
    if verbose:
        print('topo_gen() for domain {:}'.format(grid.id))

    use_center = grid.refine_loop_args['use_center']
    if (not use_center) and (do_roughness or do_gradient):
        raise Exception('"use_center" needs to be used for roughness or gradient')
    do_effective = do_effective and do_thinwalls
    if timers: clock.delta('setup topo_gen')

    # Step 1: Refine grid
    levels = grid.refine_loop(verbose=False, timers=False)
    nrfl = levels[-1].rfl
    if timers: clock.delta('refine grid')

    # Step 2: Project elevation to the finest grid
    levels[-1].project_source_data_onto_target_mesh(grid.eds, use_center=use_center, work_in_3d=grid.refine_loop_args['work_in_3d'])
    if save_hits:
        lon_src, lat_src = grid.eds.lon_coord, grid.eds.lat_coord
        hits = HitMap(shape=(lat_src.size, lon_src.size))
        hits[:] = levels[-1].source_hits(grid.eds, use_center=use_center, singularity_radius=0.0)
        hits.box = (lat_src.start, lat_src.stop, lon_src.start, lon_src.stop)
    if timers: clock.delta('assign data')

    # Step 2: Create a ThinWalls object on the finest grid and coarsen back
    tw = ThinWalls.ThinWalls(lon=levels[-1].lon, lat=levels[-1].lat, rfl=levels[-1].rfl)
    if use_center:
        tw.set_cell_mean(levels[-1].height)
        if do_thinwalls:
            tw.set_edge_to_step(tw_interp)
    else:
        tw.set_center_from_corner(levels[-1].height)
        if do_thinwalls:
            tw.set_edge_from_corner(levels[-1].height)
    if do_effective:
        tw.init_effective_values()
    if timers: clock.delta('init thinwalls')
    if verbose: print('Refine level {:} {:}'.format(tw.rfl, tw))

    if timers: loop_start = clock.now
    for _ in range(nrfl):
        if timers: clock.update_prev()
        if do_effective:
            # # old methods
            # patho_ew = tw.diagnose_EW_pathway()
            # patho_ns = tw.diagnose_NS_pathway()
            # patho_sw, patho_se, patho_ne, patho_nw = tw.diagnose_corner_pathways()
            # tw.push_corners(verbose=False)
            # tw.lower_tallest_buttress(verbose=False)
            # # tw.fold_out_central_ridges(er=True, verbose=False)
            # tw.fold_out_central_ridges(verbose=False)
            # tw.invert_exterior_corners(verbose=False)
            # tw.limit_NS_EW_connections(patho_ns, patho_ew, verbose=False)
            # tw.limit_corner_connections(patho_sw, patho_se, patho_ne, patho_nw, verbose=False)

            # new methods
            pathn_s = tw.diagnose_pathways_straight()
            pathn_c = tw.diagnose_pathways_corner()
            tw.push_interior_corners(adjust_centers=True, verbose=False)
            tw.lower_interior_buttresses(do_ave=True, adjust_mean=False, verbose=False)
            tw.fold_interior_ridges(adjust_centers=True, adjust_low_only=True, verbose=False)
            tw.expand_interior_corners(adjust_centers=True, verbose=False)
            tw.limit_connections(connections=pathn_s, verbose=False)
            tw.limit_connections(connections=pathn_c, verbose=False)
            tw.lift_ave_max()
        if timers: clock.delta('refine loop (effective thinwalls)')
        tw = tw.coarsen(do_thinwalls=do_thinwalls, do_effective=do_effective)
        if verbose: print('Refine level {:} {:}'.format(tw.rfl, tw))
        if timers: clock.delta('refine loop (coarsen grid)')
    if timers: clock.delta('refine loop (total)', ref_time=loop_start)

    out = subgrid_roughness_gradient(
        levels, tw.c_simple.ave, do_roughness=do_roughness, do_gradient=do_gradient, Idx=grid.Idx, Idy=grid.Idy)
    tw.roughness, tw.gradient = out['h2'], out['gh']
    if timers: clock.delta("roughness/gradient")

    # Step 3: Decorate the coarsened ThinWalls object
    tw.id = grid.id
    tw.mrfl = nrfl

    if timers:
        clock.delta('total')
        clock.print()

    if save_hits: return tw, hits
    else: return tw

def topo_gen_mp(domain_list, nprocs=None, topo_gen_args={}):
    """A wrapper for multiprocessing topo_gen"""
    if nprocs is None:
        nprocs = len(domain_list)
    pool = multiprocessing.Pool(processes=nprocs)
    tw_list = pool.map(functools.partial(topo_gen, **topo_gen_args), domain_list)
    pool.close()
    pool.join()

    return tw_list
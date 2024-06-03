import argparse
import sys
import numpy
import multiprocessing
import functools
import netCDF4
import time
# sys.path.insert(0,'./thin-wall-topography/python')
sys.path.insert(0,'/Users/hewang/lib/thin-wall-topography/python')
import GMesh
import ThinWalls

class TimeLog(object):
    """An object logging times"""
    def __init__(self, keys):
        self.processes = dict()
        for key in keys:
            self.processes[key] = 0.0
        self.update_prev()
    def update_prev(self):
        """Set current time as a reference previous time"""
        self.ref_time = self.now
    @property
    def now(self):
        """"Current time"""
        return time.time_ns()
    def delta(self, key, ref_time=None):
        """Accumulates time elapsed since reference time in processes[key]"""
        if ref_time is None: ref_time = self.ref_time
        dt = self.now - ref_time
        self.update_prev()
        self.processes[key] += dt
    def print(self):
        for label, dt in self.processes.items():
            dt //= 1000000
            if dt<9000: print( '{:>10}ms : {}'.format( dt, label) )
            else: print( '{:>10}secs : {}'.format( dt / 1000, label) )

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
    def __init__(self, lon=None, lat=None, id=(0,0),
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
        super().__init__(lon=lon, lat=lat)
        self.id = id
        self.north_masks = mask_recs
        self.refine_loop_args = refine_loop_args
        self._fit_src_coords(eds, subset_eds=subset_eds, halo=src_halo)

    def __str__(self):
        disp = [str(type(self)),
                "Sub-domain identifier: {}".format(self.id),
                "Target grid size (nj ni): ({:9d}, {:9d})".format( self.nj, self.ni ),
                "Source grid size (nj ni): ({:9d}, {:9d}), indices: {}".format( self.eds.nj, self.eds.ni,
                                                                               (self.eds.lat_coord.start, self.eds.lat_coord.stop,
                                                                                self.eds.lon_coord.start, self.eds.lon_coord.stop) ),
                ("Target grid range (lat lon): "+
                 "({:10.6f}, {:10.6f})  ({:10.6f}, {:10.6f})").format( self.lat.min(), self.lat.max(),
                                                                       numpy.mod(self.lon.min(), 360),
                                                                       numpy.mod(self.lon.max(), 360) ),
                ("Source grid range (lat lon): "+
                 "({:10.6f}, {:10.6f})  ({:10.6f}, {:10.6f})").format( self.eds.lat_coord.bounds[0], self.eds.lat_coord.bounds[-1],
                                                                       numpy.mod(self.eds.lon_coord.bounds[0], 360),
                                                                       numpy.mod(self.eds.lon_coord.bounds[-1], 360) )
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
    def __init__(self, lon=None, lat=None, reentrant_x=False, fold_n=False, num_north_pole=0, pole_radius=0.25):
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
        super().__init__(lon=lon, lat=lat)

        self.reentrant_x = reentrant_x
        self.fold_n = fold_n
        if self.fold_n:
            assert self.ni%2==0, 'An odd number ni does not work with bi-polar cap.'
            num_north_pole=2

        self.pole_radius = pole_radius
        self.north_mask = self.find_north_pole_rectangles(num_north_pole=num_north_pole)

    def __str__(self):
        disp = [str(type(self)),
                "Domain size (nj ni): (%i, %i)"%(self.nj, self.ni),
                "Domain range (lat lon): (%10.6f, %10.6f)  (%10.6f, %10.6f)"%(self.lat.min(), self.lat.max(), numpy.mod(self.lon.min(), 360), numpy.mod(self.lon.max(), 360))
               ]
        if len(self.north_mask)>0:
            disp.append('North Pole rectangles (radius =%5.2f%1s)'%(self.pole_radius, chr(176)))
            for box in self.north_mask:
                disp.append('  js,je,is,ie: %s, shape: (%i,%i)'%(box, box[1]-box[0], box[3]-box[2]))
        return '\n'.join(disp)

    def find_north_pole_rectangles(self, north_pole_cutoff_lat=None, num_north_pole=0):
        """Returns the extented rectangles of the grids enclosed by a latitudinal circle
        Parameters
        ----------
        north_pole_cutoff_lat : float, optional
            Cutoff latitude (not included) which the rectangle boxes enclose.
        num_north_pole : int, optional
            Number of North Poles.

        Output:
        ----------
        recs : list of tuples
            Indices of rectangle boxes. Number of boxes depends on num_north_pole.
        """
        if north_pole_cutoff_lat is None:
            north_pole_cutoff_lat = 90.0 - self.pole_radius
        jj, ii = numpy.where(self.lat>north_pole_cutoff_lat)

        if jj.size==0 or ii.size==0 or num_north_pole==0:
            recs = []
        elif num_north_pole==1:
            recs = [(jj.min(), jj.max(), ii.min(), ii.max())]
        elif num_north_pole==2:
            jjw = jj[ii<self.ni//2]; iiw = ii[ii<self.ni//2]
            jje = jj[ii>self.ni//2]; iie = ii[ii>self.ni//2]
            assert numpy.all(jjw==jje), 'nj in the two mask domains mismatch.'
            jj = jjw
            assert (jjw.max()==self.nj), 'mask domains do not reach the north boundary.'
            assert (iiw.min()+iie.max()==self.ni) and ((iiw.max()+iie.min()==self.ni)), \
                'ni in the two mask domains mismatch.'
            recs = [(jj.min(), jj.max(), iiw.min(), iiw.max()),
                    (jj.min(), jj.max(), iie.min(), iie.max())]
            # self.north_mask = [(jj.min(), 2*self.nj-jj.min(), iiw.min(), iiw.max()),
            #                    (jj.min(), 2*self.nj-jj.min(), iie.min(), iie.max())] # extend passing the northern boundary for halos
        else:
            raise Exception('Currently only two north pole rectangles are supported.')
        return recs

    def find_local_masks(self, box, halo):
        """Finds where the north pole rectangles overlap with the subdomain"""
        masks = []
        jst, jed, ist, ied = box
        jsth, jedh, isth, iedh = jst-halo, jed+halo, ist-halo, ied+halo
        for jstm, jedm, istm, iedm in self.north_mask:
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

    @staticmethod
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

    def create_subdomains(self, pelayout, tgt_halo=0, x_sym=True, y_sym=False, norm_lon=True, eds=None, subset_eds=True, src_halo=0,
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
        if pelayout[1]==1 and tgt_halo>0:
            print('WARNING: only 1 subdomain in i-direction, which may not work with bi-polar cap.')
        if (not x_sym) and self.fold_n:
            print(('WARNING: domain decomposition is not guaranteed to be symmetric in x direction, ',
                   'which may not work with bi-polar cap.'))

        j_domain = Domain.decompose_domain(self.nj, pelayout[0], symmetric=y_sym)
        i_domain = Domain.decompose_domain(self.ni, pelayout[1], symmetric=x_sym)
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
                lon = Domain.slice(self.lon, box=box_data, cyclic_zonal=self.reentrant_x, fold_north=self.fold_n)
                lat = Domain.slice(self.lat, box=box_data, cyclic_zonal=self.reentrant_x, fold_north=self.fold_n)
                if norm_lon: lon = Domain.normlize_longitude(lon, lat)

                masks = self.find_local_masks((jst, jed, ist, ied), tgt_halo)
                chunks[pe_j, pe_i] = RefineWrapper(lon=lon, lat=lat, id=(pe_j, pe_i),
                                                   eds=eds, subset_eds=subset_eds, src_halo=src_halo,
                                                   mask_recs=masks, refine_loop_args=refine_loop_args)
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
        lon = Domain.slice(self.lon, box=mask_halo, cyclic_zonal=False, fold_north=True)
        lat = Domain.slice(self.lat, box=mask_halo, cyclic_zonal=False, fold_north=True)
        if norm_lon: lon = Domain.normlize_longitude(lon, lat)

        return Domain(lon=lon, lat=lat, reentrant_x=False, num_north_pole=1, pole_radius=pole_radius)

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
            for mask in north_masks:
                mask_domain = self.create_mask_domain(mask=mask, tgt_halo=tgt_halo, pole_radius=90.0-latc)
                refine_loop_args['singularity_radius'] = mask_domain.pole_radius
                mask_domain.regrid_topography(pelayout=pelayout, tgt_halo=tgt_halo, nprocs=nprocs, eds=eds, src_halo=src_halo,
                                              refine_loop_args=refine_loop_args, calc_args=calc_args, hitmap=hitmap,
                                              bnd_tol_level=bnd_tol_level)
                self.stitch_mask_domain(mask_domain, mask, tgt_halo, tolerance=bnd_tol_level, **calc_args)
            self.stitch_mask_fold_north(tolerance=bnd_tol_level, do_effective=calc_args['do_effective'])
            north_masks = self.find_north_pole_rectangles(north_pole_cutoff_lat=latc, num_north_pole=2)
            latc += lat_step

    @staticmethod
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

    @staticmethod
    def slice(var, box, position='corner', fold_north=True, cyclic_zonal=True, fold_south=False):
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
        var : 2D ndarray
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
        Nj, Ni = var.shape
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

        NoWe, North, NoEa = var[yn, xnw], var[yn, xn], var[yn, xne]
        West, Center, East = var[yc, xw], var[yc, xc], var[yc, xe]
        SoWe, South, SoEa = var[ys, xsw], var[ys, xs], var[ys, xse]
        # print(NoWe.shape, North.shape, NoEa.shape)
        # print(West.shape, Center.shape, East.shape)
        # print(SoWe.shape, South.shape, SoEa.shape)
        return numpy.r_[numpy.c_[SoWe, South, SoEa], numpy.c_[West, Center, East], numpy.c_[NoWe, North, NoEa]]

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

def convol( levels, h, f, verbose=False ):
    """Coarsens the product of h*f across all levels"""
    levels[-1].height = ( h * f ).reshape(levels[-1].nj,levels[-1].ni)
    for k in range( len(levels) - 1, 0, -1 ):
        if verbose: print('Coarsening {} -> {}'.format(k,k-1))
        levels[k].coarsenby2( levels[k-1] )
    return levels[0].height

def topo_gen(grid, periodicity=True, do_roughness=False, do_gradient=False, do_thinwalls=False, do_effective=True, save_hits=True, verbose=True, timers=False):
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
            tw.set_edge_to_step()
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

    if do_roughness or do_gradient:
        h2min=1.e-7
        nx = 2**( len(levels) - 1 )
        x = ( numpy.arange(nx) - ( nx - 1 ) /2 ) * numpy.sqrt( 12 / ( nx**2 - 1 ) ) # This formula satisfies <x>=0 and <x^2>=1
        X, Y = numpy.meshgrid( x, x )
        X, Y = X.reshape(1,nx,1,nx), Y.reshape(1,nx,1,nx)
        h = levels[-1].height.reshape(levels[0].nj,nx,levels[0].ni,nx)
        HX = convol( levels, h, X ) # mean of h * x
        HY = convol( levels, h, Y ) # mean of h * y
        if do_roughness:
            H2 = convol( levels, h, h ) # mean of h^2
            tw.roughness = H2 - tw.c_simple.ave**2 - HX - HY + h2min
        if do_gradient:
            tw.gradient = numpy.sqrt(HX**2 + HY**2)
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

def write_output(domain, filename, do_center_only=False, do_roughness=False, do_gradient=False,
                 format='NETCDF3_64BIT_OFFSET', history='', description='',
                 inverse_sign=True, elev_unit='m', dtype=numpy.float64, dtype_int=numpy.int32):
    """Output to netCDF
    """
    # def clip(field, clip_height=clip_height, positive=(not inverse_sign)):
    #     if isinstance(clip_height, float):
    #         field[field>clip_height] = clip_height
    #     return (float(positive)*2-1)*field
    def signed(field, positive=(not inverse_sign)):
        return (float(positive)*2-1)*field

    def write_variable(nc, field, field_name, field_dim, units=elev_unit, long_name=' ', dtype=dtype):
        if field_dim == 'c':
            dim = ('ny', 'nx')
        elif field_dim == 'u':
            dim = ('ny', 'nxq')
        elif field_dim == 'v':
            dim = ('nyq', 'nx')
        else: raise Exception('Wrong field dimension.')
        varout = nc.createVariable(field_name, dtype, dim)
        varout[:] = field
        varout.units = units
        varout.long_name = long_name

    if description=='':
        # if do_mean_only:
        #     description = "Mean topography at cell-centers"
        if do_center_only:
            description = "Min, mean and max elevation at cell-centers"
        else:
            description = "Min, mean and max elevation at cell-centers and u/v-edges"

    ny, nx = domain.shape

    ncout = netCDF4.Dataset(filename, mode='w', format=format)
    ncout.createDimension('nx', nx)
    ncout.createDimension('ny', ny)
    ncout.createDimension('nxq', nx+1)
    ncout.createDimension('nyq', ny+1)

    if do_center_only:
        write_variable(ncout, signed(domain.c_simple.ave), 'depth', 'c',
                    long_name='Simple cell-center mean topography')
        write_variable(ncout, signed(domain.c_simple.hgh), 'depth_hgh', 'c',
                    long_name='Simple cell-center highest topography')
        write_variable(ncout, signed(domain.c_simple.low), 'depth_low', 'c',
                    long_name='Simple cell-center lowest topography')
        write_variable(ncout, domain.c_rfl, 'c_rfl', 'c',
                    long_name='Refinement level at cell-centers', units='nondim', dtype=dtype_int)
    else:
        # cell-centers
        write_variable(ncout, signed(domain.c_simple.ave), 'c_simple_ave', 'c',
                    long_name='Simple cell-center mean topography')
        write_variable(ncout, signed(domain.c_simple.hgh), 'c_simple_hgh', 'c',
                    long_name='Simple cell-center highest topography')
        write_variable(ncout, signed(domain.c_simple.low), 'c_simple_low', 'c',
                    long_name='Simple cell-center lowest topography')

        write_variable(ncout, signed(domain.c_effective.ave), 'c_effective_ave', 'c',
                    long_name='Effective cell-center mean topography')
        write_variable(ncout, signed(domain.c_effective.hgh), 'c_effective_hgh', 'c',
                    long_name='Effective cell-center highest topography')
        write_variable(ncout, signed(domain.c_effective.low), 'c_effective_low', 'c',
                    long_name='Effective cell-center lowest topography')
        # u-edges
        write_variable(ncout, signed(domain.u_simple.ave), 'u_simple_ave', 'u',
                    long_name='Simple u-edge mean topography')
        write_variable(ncout, signed(domain.u_simple.hgh), 'u_simple_hgh', 'u',
                    long_name='Simple u-edge highest topography')
        write_variable(ncout, signed(domain.u_simple.low), 'u_simple_low', 'u',
                    long_name='Simple u-edge lowest topography')

        write_variable(ncout, signed(domain.u_effective.ave), 'u_effective_ave', 'u',
                    long_name='Effective u-edge mean topography')
        write_variable(ncout, signed(domain.u_effective.hgh), 'u_effective_hgh', 'u',
                    long_name='Effective u-edge highest topography')
        write_variable(ncout, signed(domain.u_effective.low), 'u_effective_low', 'u',
                    long_name='Effective u-edge lowest topography')
        # v-edges
        write_variable(ncout, signed(domain.v_simple.ave), 'v_simple_ave', 'v',
                    long_name='Simple v-edge mean topography')
        write_variable(ncout, signed(domain.v_simple.hgh), 'v_simple_hgh', 'v',
                    long_name='Simple v-edge highest topography')
        write_variable(ncout, signed(domain.v_simple.low), 'v_simple_low', 'v',
                    long_name='Simple v-edge lowest topography')

        write_variable(ncout, signed(domain.v_effective.ave), 'v_effective_ave', 'v',
                    long_name='Effective v-edge mean topography')
        write_variable(ncout, signed(domain.v_effective.hgh), 'v_effective_hgh', 'v',
                    long_name='Effective v-edge highest topography')
        write_variable(ncout, signed(domain.v_effective.low), 'v_effective_low', 'v',
                    long_name='Effective v-edge lowest topography')
        # refinement levels
        write_variable(ncout, domain.c_rfl, 'c_rfl', 'c',
                    long_name='Refinement level at cell-centers', units='nondim', dtype=dtype_int)
        write_variable(ncout, domain.u_rfl, 'u_rfl', 'u',
                    long_name='Refinement level at u-edges', units='nondim', dtype=dtype_int)
        write_variable(ncout, domain.v_rfl, 'v_rfl', 'v',
                    long_name='Refinement level at v-edges', units='nondim', dtype=dtype_int)
    if do_roughness:
        write_variable(ncout, domain.roughtness, 'h2', 'c',
                       long_name='Sub-grid plane-fit roughness', units='m2')
    if do_gradient:
        write_variable(ncout, domain.gradient, 'gradh', 'c',
                       long_name='Sub-grid plane-fit gradient', units='m2')
    ncout.description = description
    ncout.history = history
    ncout.close()

def write_hitmap(hitmap, filename, format='NETCDF3_64BIT_OFFSET', history='', description='', dtype=numpy.int32):
    """Output to netCDF
    """
    if description=='':
        description = 'Map of "hits" of the source data'

    ny, nx = hitmap.shape

    ncout = netCDF4.Dataset(filename, mode='w', format=format)
    ncout.createDimension('nx', nx)
    ncout.createDimension('ny', ny)
    ncout.createDimension('nxq', nx+1)
    ncout.createDimension('nyq', ny+1)

    varout = ncout.createVariable('hitmap', dtype, ('ny','nx'))
    varout[:] = hitmap[:]
    varout.units = 'nondim'
    varout.long_name = '>0 source data is used; 0 not'

    varout = ncout.createVariable('lat', numpy.float64, ('nyq',))
    varout[:] = hitmap.lat[:,0]
    varout.units = 'degree'
    varout.long_name = 'Latitude at the nodes'

    varout = ncout.createVariable('lon', numpy.float64, ('nxq',))
    varout[:] = hitmap.lon[0,:]
    varout.units = 'degree'
    varout.long_name = 'Longitude at the nodes'

    ncout.description = description
    ncout.history = history
    ncout.close()

def main(argv):
    parser = argparse.ArgumentParser(description='Objective topography regridding',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-v", "--verbose", action='store_true')
    parser.add_argument("--verbosity", default=0, help='Granularity of log output')

    parser_tgt = parser.add_argument_group('Target grid')
    parser_tgt.add_argument("--target_grid", default='', help='File name of the target grid')
    parser_tgt.add_argument("--lon_tgt", default='x', help='Field name in target grid file for longitude')
    parser_tgt.add_argument("--lat_tgt", default='y', help='Field name in target grid file for latitude')
    parser_tgt.add_argument("--non-supergrid", action='store_true',
                            help='If specified, the target grid file is not on a supergrid. Currently not supported')
    parser_tgt.add_argument("--mono_lon", action='store_true',
                            help='If specified, a 360-degree shift will be made to guarantee the last row of lon is monotonic.')
    parser_tgt.add_argument("--tgt_halo", default=0, type=int, help='Halo size at both directions for target grid subdomain')

    parser_src = parser.add_argument_group('Source data')
    parser_src.add_argument("--source", default='', help='File name of the source data')
    parser_src.add_argument("--lon_src", default='lon', help='Field name in source file for longitude')
    parser_src.add_argument("--lat_src", default='lat', help='Field name in source file for latitude')
    parser_src.add_argument("--src_halo", default=0, type=int, help='Halo size of at both directions for subsetting source data')
    parser_src.add_argument("--elev", default='elevation', help='Field name in source file for elevation')
    parser_src.add_argument("--remove_src_repeat_lon", action='store_true',
                            help=('If specified, the repeating longitude in the last column is removed. '
                                  'Elevation along that longitude will be the mean.'))

    parser_cc = parser.add_argument_group('Calculation options')
    parser_cc.add_argument("--do_thinwalls", action='store_true', help='Calculate thin wall paraemeters')
    parser_cc.add_argument("--do_thinwalls_effective", action='store_true', help='Calcuate effective depth in porous topography.')
    parser_cc.add_argument("--do_roughness", action='store_true', help='Calcuate roughness')
    parser_cc.add_argument("--do_gradient", action='store_true', help='Calcuate sub-grid gradient')
    parser_cc.add_argument("--save_hits", action='store_true', help='Save hitmap to a file')

    parser_pe = parser.add_argument_group('Parallelism options')
    parser_pe.add_argument("--nprocs", default=0, type=int, help='Number of processors used in parallel')
    parser_pe.add_argument("--pe", nargs='+', type=int, help='Domain decomposition layout')
    parser_pe.add_argument("--pe_p", nargs='+', type=int, help='Domain decomposition layout for the North Pole rectangles')
    parser_pe.add_argument("--max_mb", default=10240, type=float, help='Memory limit per processor')
    parser_pe.add_argument("--bnd_tol_level", default=2, type=int, help='Shared boundary treatment strategy')

    parser_rgd = parser.add_argument_group('Regrid options')
    parser_rgd.add_argument("--use_corner", action='store_true', help='Use cell corners for nearest neighbor.')
    parser_rgd.add_argument("--fixed_refine_level", default=-1, type=int, help='Force refinement to a specific level.')
    parser_rgd.add_argument("--refine_in_3d", action='store_true', help='If specified, use great circle for grid interpolation.')
    parser_rgd.add_argument("--no_resolution_limit", action='store_true',
                            help='If specified, do not use resolution constraints to exit refinement')
    parser_rgd.add_argument("--pole_start", default=85.0, type=float, help='North Pole start lat')
    parser_rgd.add_argument("--pole_step", default=0.5, type=float, help='Pole steps')
    parser_rgd.add_argument("--pole_end", default=89.75, type=float, help='North Pole step lat')

    parser_out = parser.add_argument_group('Output options')
    parser_out.add_argument("--output", default='')

    args = parser.parse_args(argv)

    clock = TimeLog(['Read source', 'Read target', 'Setup', 'Regrid main', 'Regrid masked', 'Write output'])
    # Read source data
    print('Reading source data from ', args.source)
    if args.verbose:
        print(  "'"+args.lon_src+"'", '-> lon_src')
        print(  "'"+args.lat_src+"'", '-> lat_src')
        print(  "'"+args.elev+"'", '-> elev')
    lon_src = netCDF4.Dataset(args.source)[args.lon_src][:]
    lat_src = netCDF4.Dataset(args.source)[args.lat_src][:]
    elev = netCDF4.Dataset(args.source)[args.elev][:]
    if args.remove_src_repeat_lon:
        lon_src = lon_src[:-1]
        elev = numpy.c_[(elev[:,1]+elev[:,-1])*0.5, elev[:,1:-1]]
    eds = GMesh.UniformEDS(lon_src, lat_src, elev)
    clock.delta('Read source')

    # Read target grid
    if args.non_supergrid: raise Exception('Only supergrid is supported.')
    print('Reading target grid from ', args.target_grid)
    if args.verbose:
        print(  "'"+args.lon_tgt+"'[::2, ::2]", '-> lonb_tgt')
        print(  "'"+args.lat_tgt+"'[::2, ::2]", '-> latb_tgt')
    lonb_tgt = netCDF4.Dataset(args.target_grid).variables['x'][::2, ::2].data
    latb_tgt = netCDF4.Dataset(args.target_grid).variables['y'][::2, ::2].data
    if args.mono_lon:
        for ix in range(lonb_tgt.shape[1]-1):
            if lonb_tgt[-1,ix+1]<lonb_tgt[-1,ix]: lonb_tgt[-1,ix+1] += 360.0
    clock.delta('Read target')

    # Domain decomposition
    pe = args.pe
    pe_p = args.pe_p
    if pe_p is None: pe_p = pe
    nprocs = args.nprocs

    # Calculation options
    do_effective = args.do_thinwalls_effective and args.do_thinwalls
    calc_args = {'do_thinwalls': args.do_thinwalls,
                 'do_effective': do_effective,
                 'do_roughness': args.do_roughness,
                 'do_gradient': args.do_gradient
                }

    # Regridding and topo_gen options
    north_pole_lat = args.pole_start
    np_lat_end = args.pole_end
    np_lat_step = args.pole_step
    resolution_limit = (not args.no_resolution_limit) and (args.fixed_refine_level>0)
    if args.fixed_refine_level>0:
        resolution_limit = False
    refine_options = {'use_center': not args.use_corner,
                      'resolution_limit': resolution_limit,
                      'fixed_refine_level': args.fixed_refine_level,
                      'work_in_3d': args.refine_in_3d,
                      'singularity_radius': 90.0-args.pole_start,
                      'max_mb': args.max_mb
                      }

    if args.verbose:
        print('fixed_refine_level: ', refine_options['fixed_refine_level'])
        print('use_resolution_limit: ', refine_options['resolution_limit'])
        print('refine_in_3d: ', refine_options['work_in_3d'])
        print('north_pole_lat: ', north_pole_lat)
        print('np_lat_end: ', np_lat_end)
        print('np_lat_step: ', np_lat_step)

    # Create the target grid domain
    dm = Domain(lon=lonb_tgt, lat=latb_tgt, reentrant_x=True, fold_n=True, num_north_pole=2, pole_radius=refine_options['singularity_radius'])
    if args.save_hits:
        hm = HitMap(lon=lon_src, lat=lat_src, from_cell_center=True)
    else:
        hm = None
    clock.delta('Setup')

    # Regrid
    if args.verbose:
        print('Starting regridding the domain')
    if args.do_thinwalls:
        bnd_tol_level = args.bnd_tol_level
    else:
        bnd_tol_level = 0
    # dm.regrid_topography(pelayout=pe, tgt_halo=args.tgt_halo, nprocs=nprocs, eds=eds, src_halo=args.src_halo,
    #                      refine_loop_args=refine_options, calc_args=calc_args, hitmap=hm,
    #                      bnd_tol_level=bnd_tol_level, verbose=args.verbose)
    subdomains = dm.create_subdomains(pelayout=pe, tgt_halo=args.tgt_halo, eds=eds, subset_eds=False, src_halo=args.src_halo,
                                      refine_loop_args=refine_options, verbose=False)
    # clock.delta('Domain decomposition')

    topo_gen_args = calc_args.copy()
    topo_gen_args.update({'save_hits': not (hm is None), 'verbose': True, 'timers': True})

    if nprocs>1:
        twlist = topo_gen_mp(subdomains.flatten(), nprocs=nprocs, topo_gen_args=topo_gen_args)
    else: # with nprocs==1, multiprocessing is not used.
        twlist = [topo_gen(sdm, **topo_gen_args) for sdm in subdomains.flatten()]

    if topo_gen_args['save_hits']:
        twlist, hitlist = zip(*twlist)
        hm.stitch_hits(hitlist)

    dm.stitch_subdomains(twlist, tolerance=bnd_tol_level, verbose=args.verbose, **calc_args)

    clock.delta('Regrid main')
    if args.fixed_refine_level<0:
        if args.verbose:
            print('Starting regridding masked North Pole')
        # Donut update near the (geographic) north pole
        dm.regrid_topography_masked(lat_end=np_lat_end, lat_step=np_lat_step, pelayout=pe_p, nprocs=nprocs, tgt_halo=args.tgt_halo, eds=eds, src_halo=args.src_halo,
                                    refine_loop_args=refine_options, calc_args=calc_args, hitmap=hm, verbose=args.verbose)
    clock.delta('Regrid masked')

    # Output to a netCDF file
    write_output(dm, args.output, do_center_only=(not args.do_thinwalls), do_roughness=args.do_roughness, do_gradient=args.do_gradient,
                 format='NETCDF3_64BIT_OFFSET', history=' '.join(argv))
    if args.save_hits:
        write_hitmap(hm, 'hitmap.nc')
    clock.delta('Write output')

    clock.print()

if __name__ == "__main__":
    main(sys.argv[1:])

import argparse
import sys
import numpy
import multiprocessing
import functools
import netCDF4
# sys.path.insert(0,'/Users/hewang/lib/thin-wall-topography/python')
sys.path.insert(0,'/Users/hewang/lib/porous-topography-generator/thin-wall-topography/python')
import GMesh
import ThinWalls
# import importlib
# GMesh = importlib.import_module('thin-wall-topography.python.GMesh')
# ThinWalls = importlib.import_module('thin-wall-topography.python.ThinWalls')

class SourceData(object):
    """A container for source coords and data"""
    def __init__(self, lon=None, lat=None, elev=None, remove_repeat=False):
        assert len(lon.shape)==1, "Longitude is not 1D."
        assert len(lat.shape)==1, "Latitude is not 1D."
        assert elev.shape==(lat.shape[0], lon.shape[0]), "Elevation and coords shape mismatch."
        self.lon = lon
        self.lat = lat
        self.elev = elev
        if remove_repeat: self.remove_repeat()
    def _remove_repeat(self):
        pass

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
    """
    def __init__(self, lon=None, lat=None, domain_id=(0,0), move_north_pole_lon=False,
                 src=None, fit_src_lon=False, fit_src_lat=False, src_halo=0, mask_recs=[], refine_loop_args={}):
        """
        Parameters
        ----------
        lon, lat : array of float
            Cell corner coordinates
        domain_id : tuple, optional
            An identifier for the current subdomain. Used for easily linking subdomain to the parent domain.
        move_north_pole_lon : bool, optional
            If true, re-assign the longitude of the north pole (lat==90N) (if it is inside of the domain) to the longitude
            of the neighbor grid. Default is False.
        lon_src, lat_src : GMesh.IntCoord objects, optional
            Integerized source data coordinates.
        elev_src : array of float, optional
            Source elevation field.
        fit_src_lon, fit_src_lat : bool, optional
            If true, source grid lon_src and lat_src are tailored to encompass target grid, with a halo decided by src_halo.
            elev_src is also trimmed accordingly. Default is False.
        src_halo : integer, optional
            Halo size of the source grid in either direction.
        mask_recs : list, optional
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
        """
        self.domain_id = domain_id
        super().__init__(lon=lon, lat=lat)
        if move_north_pole_lon: self._move_north_pole()

        self.north_masks = mask_recs
        self.refine_loop_args = refine_loop_args
        self._fit_src_coords(src.lon, src.lat, src.elev, do_fit_lon=fit_src_lon, do_fit_lat=fit_src_lat, halo=src_halo)

    def __str__(self):
        disp = [str(type(self)),
                "Sub-domain identifier: {}".format(self.domain_id),
                "Target grid size (nj ni): ({:9d}, {:9d})".format( self.nj, self.ni ),
                "Source grid size (nj ni): ({:9d}, {:9d}), indices: {}".format( self.lat_src.size, self.lon_src.size,
                                                                                (self.lat_src.n0, self.lat_src.n1,
                                                                                 self.lon_src.n0, self.lon_src.n1) ),
                ("Target grid range (lat lon): "+
                 "({:10.6f}, {:10.6f})  ({:10.6f}, {:10.6f})").format( self.lat.min(), self.lat.max(),
                                                                       numpy.mod(self.lon.min(), 360),
                                                                       numpy.mod(self.lon.max(), 360) ),
                ("Source grid range (lat lon): "+
                 "({:10.6f}, {:10.6f})  ({:10.6f}, {:10.6f})").format( self.lat_src.bounds[0], self.lat_src.bounds[-1],
                                                                       numpy.mod(self.lon_src.bounds[0], 360),
                                                                       numpy.mod(self.lon_src.bounds[-1], 360) )
               ]
        if len(self.north_masks)>0:
            disp.append('North Pole rectangles: ')
            for box in self.north_masks:
                disp.append('  js,je,is,ie: %s, shape: (%i,%i)'%(box, box[1]-box[0], box[3]-box[2]))
        return '\n'.join(disp)

    def _move_north_pole_lon(self):
        """Move the North Pole longitude to that of its neighbor points
        To avoid potential unnecessarily large source grid coverage with bipolar cap
        """
        jj, ii = numpy.nonzero(self.lat==90.0)
        if ii.size>0:
            for jjj, iii in zip(jj,ii):
                try:
                    self.lon[jjj, iii] = self.lon[jjj, iii-1]
                except IndexError:
                    self.lon[jjj, iii] = self.lon[jjj, iii+1]

    def _fit_src_coords(self, lon_src, lat_src, elev_src, do_fit_lon=True, do_fit_lat=True, halo=0):
        """Returns the four-element indices of source grid that covers the current domain."""
        sni, snj = lon_src.size, lat_src.size
        dellon, dellat = (lon_src[-1]-lon_src[0])/(sni-1), (lat_src[-1]-lat_src[0])/(snj-1)
        if do_fit_lon:
            # ist_src = (numpy.mod(numpy.floor(numpy.mod(self.lon.min()-lon_src[0]+0.5*dellon,360)/dellon)-halo+sni),sni).astype(int)
            # ied_src = (numpy.mod(numpy.floor(numpy.mod(self.lon.max()-lon_src[0]+0.5*dellon,360)/dellon)+halo+sni),sni).astype(int)+1
            ist_src = int(numpy.floor(numpy.mod(self.lon.min()-lon_src[0]+0.5*dellon,360)/dellon)-halo)
            if ist_src<0:
                ist_src += sni
            ied_src = int(numpy.floor(numpy.mod(self.lon.max()-lon_src[0]+0.5*dellon,360)/dellon)+halo)+1
            if ied_src>sni:
                ied_src -= sni
            if ist_src+1==ied_src:
                ist_src, ied_src = 0, sni # All longitudes are included.
        else:
            ist_src, ied_src = 0, sni
        if do_fit_lat:
            jst_src = int(min(max(numpy.floor(0.5+(self.lat.min()-lat_src[0])/dellat-halo),0.0),snj-1))
            jed_src = int(min(max(numpy.floor(0.5+(self.lat.max()-lat_src[0])/dellat+halo),0.0),snj-1))+1
        else:
            jst_src, jed_src = 0, snj

        if ist_src>ied_src:
            self.elev_src = numpy.c_[elev_src[jst_src:jed_src,ist_src:], elev_src[jst_src:jed_src,:ied_src]]
        else:
            self.elev_src = elev_src[jst_src:jed_src:,ist_src:ied_src]

        self.lon_src = GMesh.IntCoord(lon_src[0], dellon, sni, ist_src, ied_src)
        self.lat_src = GMesh.IntCoord(lat_src[0], dellat, snj, jst_src, jed_src)

    def refine_loop(self, verbose=True):
        """A self-contained version of GMesh.refine_loop()"""
        return super().refine_loop(self.lon_src, self.lat_src, verbose=verbose, mask_res=self.north_masks,
                                   **self.refine_loop_args)

class Domain(ThinWalls.ThinWalls):
    """A container for regrided topography
    """
    def __init__(self, lon=None, lat=None, reentrant_x=False, bipolar_n=False, num_north_pole=0, pole_radius=0.25):
        """
        Parameters
        ----------
        lon, lat : float
            Cell corner coordinates to construct ThinWalls
        reentrant_x : bool, optional
            If true, the domain is reentrant in x-direction. Used for halos and assigning depth to the westernmost and easternmost u-edges.
            Default is False.
        bipolar_n : bool, optional
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
        self.bipolar_n = bipolar_n
        if self.bipolar_n:
            assert self.ni%2==0, 'An odd number ni does not work with bi-polar cap.'
            num_north_pole=2
        self.c_refinelevel = numpy.zeros( self.shape, dtype=numpy.int32 )
        self.u_refinelevel = numpy.zeros( (self.shape[0],self.shape[1]+1), dtype=numpy.int32 )
        self.v_refinelevel = numpy.zeros( (self.shape[0]+1,self.shape[1]), dtype=numpy.int32 )

        self.pole_radius = pole_radius
        self._find_north_pole_rectangles(num_north_pole=num_north_pole)

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

    def _find_north_pole_rectangles(self, north_pole_cutoff_lat=None, num_north_pole=0):
        """Returns the extented rectangles of the grids enclosed by a latitudinal circle
        full_circles : bool, optional
            If true, the grid is assumed to be a full bi-polar cap and therefore has two north pole rectangles.
        """
        if north_pole_cutoff_lat is None:
            north_pole_cutoff_lat = 90.0 - self.pole_radius
        jj, ii = numpy.where(self.lat>north_pole_cutoff_lat)

        if jj.size==0 or ii.size==0 or num_north_pole==0:
            self.north_mask = []
        elif num_north_pole==1:
            self.north_mask = [(jj.min(), jj.max(), ii.min(), ii.max())]
        elif num_north_pole==2:
            jjw = jj[ii<self.ni//2]; iiw = ii[ii<self.ni//2]
            jje = jj[ii>self.ni//2]; iie = ii[ii>self.ni//2]
            assert numpy.all(jjw==jje), 'nj in the two mask domains mismatch.'
            jj = jjw
            assert (jjw.max()==self.nj), 'mask domains do not reach the north boundary.'
            assert (iiw.min()+iie.max()==self.ni) and ((iiw.max()+iie.min()==self.ni)), 'ni in the two mask domains mismatch.'
            self.north_mask = [(jj.min(), jj.max(), iiw.min(), iiw.max()),
                               (jj.min(), jj.max(), iie.min(), iie.max())]
            # self.north_mask = [(jj.min(), 2*self.nj-jj.min(), iiw.min(), iiw.max()),
            #                    (jj.min(), 2*self.nj-jj.min(), iie.min(), iie.max())] # extend passing the northern boundary for halos
        else:
            raise Exception('Currently only two north pole rectangles are supported.')

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

    def create_subdomains(self, pelayout, tgt_halo=0, x_sym=True, y_sym=False, src=None, src_halo=0,
                          refine_loop_args={}, verbose=False):
        """Creates a list of sub-domains with corresponding source lon, lat and elev sections.

        Parameters
        ----------
        pelayout : tuple of integers, (nj, ni)
            Number of sub-domains in each direction
        x_sym, y_sym : boo, optional
            Whether try to use symmetric domain decomposition in x or y.
            Default is x_sym=True and y_sym=False.
        src : SourceData object, optional
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
        if (not x_sym) and self.bipolar_n:
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
                lon = Domain.slice(self.lon, box=box_data, cyclic_zonal=self.reentrant_x, fold_north=self.bipolar_n)
                lat = Domain.slice(self.lat, box=box_data, cyclic_zonal=self.reentrant_x, fold_north=self.bipolar_n)
                lon = self.offset_halo_lon(lon, (jst, jed, ist, ied), tgt_halo)

                masks = self.find_local_masks((jst, jed, ist, ied), tgt_halo)

                fit_src_lon = True
                if jed==self.nj and tgt_halo>0:
                    fit_src_lon=False
                fit_src_lat = True
                chunks[pe_j, pe_i] = RefineWrapper(lon=lon, lat=lat, domain_id=(pe_j, pe_i),
                                                   src=src, fit_src_lon=fit_src_lon, fit_src_lat=fit_src_lat, src_halo=src_halo,
                                                   mask_recs=masks, refine_loop_args=refine_loop_args)
                if verbose:
                    print(chunks[pe_j, pe_i], '\n')
        return chunks

    def create_mask_domain(self, mask, tgt_halo=0, pole_radius=0.25):
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

        lon = self.offset_halo_lon(lon, (jst, jed, ist, ied), tgt_halo)

        return Domain(lon=lon, lat=lat, reentrant_x=False, num_north_pole=1, pole_radius=pole_radius)

    @staticmethod
    def compare_edges(edge1, edge2, rfl1, rfl2, except_level=0, verbose=True, message=''):
        """Check if two edges are identical and if not, return the proper one.

        Parameters
        ----------
        edge1, edge2 : ThinWalls.Stats object
            Edges at the same grid location from two sources (e.g. different tiles).
        rfl1, rfl2 : int
            Corresponding maximum refinement levels of the two edges.
        except_level : int, optional
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
        str_rfls = '[rfl={:2d} vs rfl={:2d}]'.format(rfl1, rfl2)
        msg = ' '.join(['Edge differ', str_diff, ':', message, str_rfls])+'. '
        if ndiff_hgh+ndiff_ave+ndiff_low!=0:
            if except_level==0:
                raise Exception(msg)
            if rfl1!=rfl2:
                if verbose:
                    print(msg+'Use higher rfl')
                if rfl1>rfl2:
                    return edge1
                else:
                    return edge2
            else:
                if except_level==1:
                    raise Exception(msg)
                if verbose:
                    print(msg+'Use shallower depth')
                edge = ThinWalls.Stats(edge1.shape)
                edge.low = numpy.maximum(edge1.low, edge2.low)
                edge.ave = numpy.maximum(numpy.maximum(edge1.ave, edge2.ave), edge.low)
                edge.hgh = numpy.maximum(numpy.maximum(edge1.hgh, edge2.hgh), edge.ave)
                return edge
        else:
            if rfl1!=rfl2 and verbose: # This should hardly happen.
                print(message+' have the same edge but different refinement levels '+str_rfls+'.')
            return edge1

    def stitch_subdomains(self, thinwalls_list, except_level=0, verbose=True):
        """Stitch subdomain depth fields (u,v,c)

        Parameters
        ----------
        thinwalls_list : list
            A list of ThinWalls class in the subdomains
        except_level : int, optional
            Input for compare_edges
        verbose : bool, optional
            Input for compare_edges
        """

        def subdomain_edge(tw1, tw2, halo=0, axis=0, fold_north=False, except_level=2, verbose=True):
            """A wrapper for treating shared edges between two subdomains with compare_edges method.

            Parameters
            ----------
            tw1, tw2 : ThinWalls.ThinWalls object
                Two neighboring subdomains, tw1 (left/bottom) and tw2 (right/top)
            halo : int, optional
                Target grid halo size.
            axis : integer, optional
                The axis index, 0 for y direction and 1 for x direction
            fold_north : bool, optional
                If true, the two subdomains are connected at the folding northern boundary.
            except_level : int, optional
                Input for compare_edges
            verbose : bool, optional
                Input for compare_edges

            Output
            ----------
            edge_s : ThinWall.Stat object
                Edge Simple depth parameters (hgh, ave and low)
            edge_e : ThinWall.Stat object
                Edge Effective depth parameters (hgh, ave and low)
            rfl : int
                Refinement level
            """
            nj1, ni1 = tw1.shape
            nj2, ni2 = tw2.shape
            msg = '{} and {}'.format(tw1.domain_id, tw2.domain_id)
            if axis==0:
                assert ni1==ni2, msg+' have different Ni.'
                s1 = (nj1-halo, slice(halo,ni1-halo))
                if fold_north:
                    s2 = (nj2-halo, slice(ni1-halo-1,-ni1+halo-1,-1))
                else:
                    s2 = (halo, slice(halo,ni2-halo))
                edge1 = {'simple':tw1.v_simple[s1], 'effective':tw1.v_effective[s1]}
                edge2 = {'simple':tw2.v_simple[s2], 'effective':tw2.v_effective[s2]}
            elif axis==1:
                assert nj1==nj2, msg+' have different Nj.'
                s1, s2 = (slice(halo,nj1-halo), ni1-halo), (slice(halo,nj1-halo), halo)
                edge1 = {'simple':tw1.u_simple[s1], 'effective':tw1.u_effective[s1]}
                edge2 = {'simple':tw2.u_simple[s2], 'effective':tw2.u_effective[s2]}
            else: raise Exception('Axis error')

            edge_s = Domain.compare_edges(edge1['simple'], edge2['simple'], tw1.max_rfl, tw2.max_rfl,
                                          except_level=except_level, verbose=verbose, message=msg+' (simple)')
            edge_e = Domain.compare_edges(edge1['effective'], edge2['effective'], tw1.max_rfl, tw2.max_rfl,
                                          except_level=except_level, verbose=verbose, message=msg+ ' (effective)')
            return edge_s, edge_e, max(tw1.max_rfl, tw2.max_rfl)

        npj, npi = self.pelayout.shape

        # Put the list of ThinWalls on a 2D array to utilize numpy array's slicing
        boxes = numpy.empty( (npj, npi), dtype=object )
        for tw in thinwalls_list:
            boxes[tw.domain_id] = tw

        # aliasing
        Cs, Us, Vs = self.c_simple, self.u_simple, self.v_simple
        Ce, Ue, Ve = self.c_effective, self.u_effective, self.v_effective
        Cr, Ur, Vr = self.c_refinelevel, self.u_refinelevel, self.v_refinelevel

        for iy in range(npj):
            for ix in range(npi):
                (jst, jed, ist, ied), halo = self.pelayout[iy, ix]
                # cell center sizes
                nj, ni = boxes[iy,ix].shape

                Cs[jst:jed, ist:ied] = boxes[iy,ix].c_simple[halo:nj-halo, halo:ni-halo]
                Ce[jst:jed, ist:ied] = boxes[iy,ix].c_effective[halo:nj-halo, halo:ni-halo]
                Cr[jst:jed, ist:ied] = boxes[iy,ix].max_rfl

                Us[jst:jed, ist:ied+1] = boxes[iy,ix].u_simple[halo:nj-halo, halo:ni+1-halo]
                Ue[jst:jed, ist:ied+1] = boxes[iy,ix].u_effective[halo:nj-halo, halo:ni+1-halo]
                Ur[jst:jed, ist:ied+1] = boxes[iy,ix].max_rfl

                Vs[jst:jed+1, ist:ied] = boxes[iy,ix].v_simple[halo:nj+1-halo, halo:ni-halo]
                Ve[jst:jed+1, ist:ied] = boxes[iy,ix].v_effective[halo:nj+1-halo, halo:ni-halo]
                Vr[jst:jed+1, ist:ied] = boxes[iy,ix].max_rfl

                # Shared boundaries
                if ix<npi-1:
                    Us[jst:jed, ied], Ue[jst:jed, ied], Ur[jst:jed, ied] \
                        = subdomain_edge(boxes[iy,ix], boxes[iy,ix+1], halo=halo, axis=1,
                                         except_level=except_level, verbose=verbose)
                if iy<npj-1:
                    Vs[jed, ist:ied], Ve[jed, ist:ied], Vr[jed, ist:ied] \
                        = subdomain_edge(boxes[iy,ix], boxes[iy+1,ix], halo=halo, axis=0,
                                         except_level=except_level, verbose=verbose)

        # Cyclic/folding boundaries
        if self.reentrant_x:
            for iy in range(npj):
                (jst, jed, _, _), halo = self.pelayout[iy,-1]
                Us[jst:jed, -1], Ue[jst:jed, -1], Ur[jst:jed, -1] \
                    = subdomain_edge(boxes[iy,-1], boxes[iy,0], halo=halo, axis=1,
                                     except_level=except_level, verbose=verbose)
                Us[jst:jed, 0], Ue[jst:jed, 0], Ur[jst:jed, 0] \
                    = Us[jst:jed, -1], Ue[jst:jed, -1], Ur[jst:jed, -1]
        if self.bipolar_n:
            for ix in range(npi//2):
                (_, _, ist, ied), halo = self.pelayout[-1,ix]
                Vs[-1,ist:ied], Ve[-1,ist:ied], Vr[-1,ist:ied] \
                    = subdomain_edge(boxes[-1,ix], boxes[-1,npi-ix-1], halo=halo, axis=0, fold_north=True,
                                     except_level=except_level, verbose=verbose)
                (_, _, istf, iedf), _ = self.pelayout[-1,npi-ix-1]
                Vs[-1,istf:iedf], Ve[-1,istf:iedf], Vr[-1,istf:iedf] \
                    = Vs[-1,ist:ied][0,::-1], Ve[-1,ist:ied][0,::-1], Vr[-1,ist:ied][::-1]
            if npi%2!=0:
                (_, _, ist, ied), halo = self.pelayout[-1,npi//2]
                thisbox = boxes[-1,npi//2]
                nj, ni = thisbox.shape
                nhf = (ied-ist)//2
                msg = '{}'.format(thisbox.domain_id)
                Vs[-1,ist:ist+nhf] = Domain.compare_edges(thisbox.v_simple[nj-halo, halo:halo+nhf],
                                                          thisbox.v_simple[nj-halo,halo+nhf:ni-halo][0,::-1],
                                                          thisbox.max_rfl, thisbox.max_rfl,
                                                          except_level=except_level, verbose=verbose, message=msg)
                Ve[-1,ist:ist+nhf] = Domain.compare_edges(thisbox.v_effective[nj-halo, halo:halo+nhf],
                                                          thisbox.v_effective[nj-halo,halo+nhf:ni-halo][0,::-1],
                                                          thisbox.max_rfl, thisbox.max_rfl,
                                                          except_level=except_level, verbose=verbose, message=msg)
                Vs[-1,ist+nhf:ied] = Vs[-1,ist:ist+nhf][0,::-1]
                Ve[-1,ist+nhf:ied] = Ve[-1,ist:ist+nhf][0,::-1]
                Vr[-1,ist:ied] = thisbox.max_rfl

    def stitch_mask_domain(self, mask_domain, mask, halo, except_level=2, verbose=False):
        """
        The assumption is the masked domain has the higher refine level
        """
        jst, jed, ist, ied = mask

        # Cell center sizes
        nj, ni = jed-jst+2*halo, ied-ist+2*halo

        # aliasing
        Cs, Us, Vs = self.c_simple, self.u_simple, self.v_simple
        Ce, Ue, Ve = self.c_effective, self.u_effective, self.v_effective
        Cr, Ur, Vr = self.c_refinelevel, self.u_refinelevel, self.v_refinelevel

        mCs, mUs, mVs = mask_domain.c_simple, mask_domain.u_simple, mask_domain.v_simple
        mCe, mUe, mVe = mask_domain.c_effective, mask_domain.u_effective, mask_domain.v_effective
        mCr, mUr, mVr = mask_domain.c_refinelevel, mask_domain.u_refinelevel, mask_domain.v_refinelevel

        # middle part
        Cs[jst:jed, ist:ied] = mCs[halo:nj-halo, halo:ni-halo]
        Ce[jst:jed, ist:ied] = mCe[halo:nj-halo, halo:ni-halo]
        Cr[jst:jed, ist:ied] = mCr[halo:nj-halo, halo:ni-halo]

        Us[jst:jed, ist+1:ied] = mUs[halo:nj-halo, halo+1:ni-halo]
        Ue[jst:jed, ist+1:ied] = mUe[halo:nj-halo, halo+1:ni-halo]
        Ur[jst:jed, ist+1:ied] = mUr[halo:nj-halo, halo+1:ni-halo]

        Vs[jst+1:jed, ist:ied] = mVs[halo+1:nj-halo, halo:ni-halo]
        Ve[jst+1:jed, ist:ied] = mVe[halo+1:nj-halo, halo:ni-halo]
        Vr[jst+1:jed, ist:ied] = mVr[halo+1:nj-halo, halo:ni-halo]

        # shared edges
        for jj in range(jst,jed):
            msg = 'np mask ({:d},{:d}) (simple)'.format(jj,ist)
            Us[jj,ist] = Domain.compare_edges(Us[jj,ist], mUs[halo+jj-jst,halo], Ur[jj,ist], mUr[halo+jj-jst,halo],
                                              except_level=except_level, verbose=verbose, message=msg)
            msg = 'np mask ({:d},{:d}) (effective)'.format(jj,ist)
            Ue[jj,ist] = Domain.compare_edges(Ue[jj,ist], mUe[halo+jj-jst,halo], Ur[jj,ist], mUr[halo+jj-jst,halo],
                                              except_level=except_level, verbose=verbose, message=msg)
            msg = 'np mask ({:d},{:d}) (simple)'.format(jj,ied)
            Us[jj,ied] = Domain.compare_edges(Us[jj,ied], mUs[halo+jj-jst,ni-halo], Ur[jj,ied], mUr[halo+jj-jst,ni-halo],
                                              except_level=except_level, verbose=verbose, message=msg)
            msg = 'np mask ({:d},{:d}) (effective)'.format(jj,ied)
            Ue[jj,ied] = Domain.compare_edges(Ue[jj,ied], mUe[halo+jj-jst,ni-halo], Ur[jj,ied], mUr[halo+jj-jst,ni-halo],
                                              except_level=except_level, verbose=verbose, message=msg)
            Ur[jj,ist] = max(Ur[jj,ist], mUr[halo+jj-jst,halo])
            Ur[jj,ied] = max(Ur[jj,ied], mUr[halo+jj-jst,ni-halo])

        for ii in range(ist,ied):
            msg = 'np mask ({:d},{:d}) (simple)'.format(jst,ii)
            Vs[jst,ii] = Domain.compare_edges(Vs[jst,ii], mVs[halo,halo+ii-ist], Vr[jst,ii], mVr[halo,halo+ii-ist],
                                              except_level=except_level, verbose=verbose, message=msg)
            msg = 'np mask ({:d},{:d}) (effective)'.format(jst,ii)
            Ve[jst,ii] = Domain.compare_edges(Ve[jst,ii], mVe[halo,halo+ii-ist], Vr[jst,ii], mVr[halo,halo+ii-ist],
                                              except_level=except_level, verbose=verbose, message=msg)
            msg = 'np mask ({:d},{:d}) (simple)'.format(jed,ii)
            Vs[jed,ii] = Domain.compare_edges(Vs[jed,ii], mVs[nj-halo,halo+ii-ist], Vr[jed,ii], mVr[nj-halo,halo+ii-ist],
                                              except_level=except_level, verbose=verbose, message=msg)
            msg = 'np mask ({:d},{:d}) (effective)'.format(jed,ii)
            Ve[jed,ii] = Domain.compare_edges(Ve[jed,ii], mVe[nj-halo,halo+ii-ist], Vr[jed,ii], mVr[nj-halo,halo+ii-ist],
                                              except_level=except_level, verbose=verbose, message=msg)
            Vr[jst,ii] = max(Vr[jst,ii], mVr[halo,halo+ii-ist])
            Vr[jed,ii] = max(Vr[jed,ii], mVr[nj-halo,halo+ii-ist])

    def stitch_mask_fold_north(self, except_level=2, verbose=False):
        if not self.bipolar_n:
            return
        _, _, istw, iedw = self.north_mask[0]
        _, _, iste, iede = self.north_mask[1]

        Vs, Ve, Vr = self.v_simple, self.v_effective, self.v_refinelevel

        for iiw, iie in zip(numpy.arange(istw,iedw,1), numpy.arange(iede-1,iste-1,-1)):
            msg = 'northern boundary ({:d},{:d}) (simple)'.format(iiw,iie)
            Vs[-1,iiw] = Domain.compare_edges(Vs[-1,iiw], Vs[-1,iie], Vr[-1,iiw], Vr[-1,iie],
                                              except_level=except_level, verbose=verbose, message=msg)
            msg = 'northern boundary ({:d},{:d}) (effective)'.format(iiw,iie)
            Ve[-1,iiw] = Domain.compare_edges(Ve[-1,iiw], Ve[-1,iie], Vr[-1,iiw], Vr[-1,iie],
                                              except_level=except_level, verbose=verbose, message=msg)
            Vr[-1,iiw] = max(Vr[-1,iiw], Vr[-1,iie])
            Vs[-1,iie], Ve[-1,iie], Vr[-1,iie] = Vs[-1,iiw], Ve[-1,iiw], Vr[-1,iiw]

    def regrid_topography(self, pelayout=None, tgt_halo=0, nprocs=1, src=None, src_halo=0,
                          refine_loop_args={}, topo_gen_args={}, hitmap=None, bnd_tol_level=1, verbose=False):
        """"A wrapper for getting elevation from a domain"""
        subdomains = self.create_subdomains(pelayout, tgt_halo=tgt_halo, src=src, src_halo=src_halo,
                                            refine_loop_args=refine_loop_args, verbose=verbose)
        topo_gen_args['verbose'] = verbose
        topo_gen_args['save_hits'] = not (hitmap is None)

        if nprocs>1:
            twlist = topo_gen_mp(subdomains.flatten(), nprocs=nprocs, topo_gen_args=topo_gen_args)
        else:
            twlist = [topo_gen(dm, **topo_gen_args) for dm in subdomains.flatten()]

        if topo_gen_args['save_hits']:
            twlist, hitlist = zip(*twlist)
            hitmap.stitch_hits(hitlist)

        self.stitch_subdomains(twlist, except_level=bnd_tol_level, verbose=verbose)

    def regrid_topography_masked(self, lat_start=None, lat_end=89.75, lat_step=0.5,
                                 pelayout=None, tgt_halo=0, nprocs=1, src=None, src_halo=0,
                                 refine_loop_args={}, topo_gen_args={}, hitmap=None, bnd_tol_level=1, verbose=True):

        # Backup the initial mask domain indices
        north_mask_org = self.north_mask

        if lat_start is None: lat_start = 90.0 - self.pole_radius
        print(lat_start, lat_end, lat_step)
        latc = lat_start + lat_step
        while latc<=lat_end:
            print(latc)
            for mask in self.north_mask:
                mask_domain = self.create_mask_domain(mask=mask, tgt_halo=tgt_halo, pole_radius=90.0-latc)
                refine_loop_args['singularity_radius'] = mask_domain.pole_radius
                mask_domain.regrid_topography(pelayout=pelayout, tgt_halo=tgt_halo, nprocs=nprocs, src=src, src_halo=src_halo,
                                              refine_loop_args=refine_loop_args, topo_gen_args=topo_gen_args, hitmap=hitmap,
                                              bnd_tol_level=bnd_tol_level)
                self.stitch_mask_domain(mask_domain, mask, tgt_halo, except_level=bnd_tol_level)
            self.stitch_mask_fold_north(except_level=bnd_tol_level)
            self._find_north_pole_rectangles(north_pole_cutoff_lat=latc, num_north_pole=2)
            latc += lat_step
        self.north_mask = north_mask_org

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

        cyc_w, c, cyc_e = var[yc, xw], var[yc, xc], var[yc, xe]
        fd_nw, fd_no, fd_ne = var[yn, xnw], var[yn, xn], var[yn, xne]
        no_sw, no_so, no_se = var[ys, xsw], var[ys, xs], var[ys, xse]
        # print(fd_nw.shape, fd_no.shape, fd_ne.shape)
        # print(cyc_w.shape, c.shape, cyc_e.shape)
        # print(no_sw.shape, no_so.shape, no_se.shape)
        return numpy.r_[numpy.c_[no_sw, no_so, no_se], numpy.c_[cyc_w, c, cyc_e], numpy.c_[fd_nw, fd_no, fd_ne]]

    def offset_halo_lon(self, lon, box_data, halo):
        """Offset longitude (corners) for cyclic N-W boundary and folding northern boundary."""
        jst, jed, ist, ied = box_data
        nj, ni = lon.shape
        assert nj==jed-jst+halo*2+1 and ni==ied-ist+halo*2+1, 'offset_halo_lon: indices do not match with lon.'
        lon_out = lon.copy()
        if self.reentrant_x:
            if self.bipolar_n: # no offset for the northeast and northwest corner blocks.
                j_end = self.nj-(jst-halo)+1
            else:
                j_end = None
            lon_out[:j_end, :-ni-(ist-halo)] -= 360.0
            lon_out[:j_end, self.ni-(ist-halo)+1:] += 360.0
        if self.bipolar_n:
            iq1, iq3 = self.ni//4, self.ni//4*3
            lon_out[self.nj-(jst-halo)+1:, -ni-(ist-halo):max(iq1-(ist-halo),0)] -= 360.0
            lon_out[self.nj-(jst-halo)+1:, max(iq3-(ist-halo)+1,0):self.ni-(ist-halo)+1] += 360.0
        return lon_out

def topo_gen(grid, do_center_only=False, do_effective=True, save_hits=True, verbose=True):
    """Generate topography

    Parameters
    ----------
    grid : RefineWrapper object

    Returns
    ----------
    tw : ThinWalls.ThinWalls object
    """

    if do_center_only:
        do_effective = False
    use_center = grid.refine_loop_args['use_center'] and do_center_only # reserve this for mean only for now
    if verbose:
        print(grid.domain_id)

    # Step 1: Refine grid and save the finest grid
    finest_grid = grid.refine_loop(verbose=verbose)[-1]
    nrfl = finest_grid.rfl

    if save_hits:
        hits = HitMap(lon=grid.lon_src.bounds, lat=grid.lat_src.bounds, from_cell_center=False)
        hits[:] = finest_grid.source_hits(grid.lon_src, grid.lat_src, use_center=use_center, singularity_radius=0.0)
        hits.box = (grid.lat_src.n0, grid.lat_src.n1, grid.lon_src.n0, grid.lon_src.n1)

    # Step 2: Create a ThinWalls object on the finest grid and coarsen back
    tw = ThinWalls.ThinWalls(lon=finest_grid.lon, lat=finest_grid.lat, rfl=finest_grid.rfl)
      # Spawn elevation data to the finest grid
    tw.project_source_data_onto_target_mesh(grid.lon_src, grid.lat_src, grid.elev_src, use_center=use_center)

      # Interpolate elevation to cell centers and edges (simple)
    if use_center:
        tw.set_cell_mean(tw.height)
    else:
        tw.set_center_from_corner()

    if not do_center_only:
        tw.set_edge_from_corner()
        if do_effective:
            # Initialize effective depths
            tw.init_effective_values()

      # Coarsen back
    for _ in range(nrfl):
        if do_effective:
            tw.push_corners(verbose=verbose)
            tw.lower_tallest_buttress(verbose=verbose)
            # tw.fold_out_central_ridges(er=True, verbose=verbose)
            tw.fold_out_central_ridges(verbose=verbose)
            tw.invert_exterior_corners(verbose=verbose)
        tw = tw.coarsen()

    # Step 3: Decorate the coarsened ThinWalls object
    tw.domain_id = grid.domain_id
    tw.max_rfl = nrfl

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

def write_output(domain, filename, do_center_only=False, format='NETCDF3_64BIT_OFFSET', history='', description='',
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
        write_variable(ncout, domain.c_refinelevel, 'c_refinelevel', 'c',
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
        write_variable(ncout, domain.c_refinelevel, 'c_refinelevel', 'c',
                    long_name='Refinement level at cell-centers', units='nondim', dtype=dtype_int)
        write_variable(ncout, domain.u_refinelevel, 'u_refinelevel', 'u',
                    long_name='Refinement level at u-edges', units='nondim', dtype=dtype_int)
        write_variable(ncout, domain.v_refinelevel, 'v_refinelevel', 'v',
                    long_name='Refinement level at v-edges', units='nondim', dtype=dtype_int)

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
    parser_tgt.add_argument("--no_mono_lon", action='store_true',
                            help='If specified, no 360-degree shift will be made to guarantee the last row of lon is monotonic.')
    parser_tgt.add_argument("--tgt_halo", default=0, type=int, help='Halo size at both directions for target grid subdomain')

    parser_src = parser.add_argument_group('Source data')
    parser_src.add_argument("--source", default='', help='File name of the source data')
    parser_src.add_argument("--lon_src", default='lon', help='Field name in source file for longitude')
    parser_src.add_argument("--lat_src", default='lat', help='Field name in source file for latitude')
    parser_src.add_argument("--src_halo", default=0, type=int, help='Halo size of at both directions for subsetting source data')
    parser_src.add_argument("--elev", default='elevation', help='Field name in source file for elevation')
    parser_src.add_argument("--remove_src_repeat_lon", action='store_true',
                            help='If specified, the repeating longitude in the last column is removed. Elevation along that longitude will be the mean.')

    parser_cc = parser.add_argument_group('Calculation options')
    parser_cc.add_argument("--do_center_only", action='store_true', help='Calculate only the mean topography at cell centers.')
    parser_cc.add_argument("--do_porous_effective", action='store_true', help='Calcuate effective depth in porous topography.')
    parser_cc.add_argument("--save_hits", action='store_true', help='Save hitmap to a file.')

    parser_pe = parser.add_argument_group('Parallelism options')
    parser_pe.add_argument("--use_serial", action='store_true', help='If specified, use serial.')
    parser_pe.add_argument("--nprocs", default=0, type=int, help='Number of processors used in parallel')
    parser_pe.add_argument("--pe", nargs='+', type=int, help='Domain decomposition layout')
    parser_pe.add_argument("--pe_p", nargs='+', type=int, help='Domain decomposition layout for the North Pole rectangles')
    parser_pe.add_argument("--max_mb", default=10240, type=float, help='Memory limit per processor')
    parser_pe.add_argument("--bnd_tol_level", default=2, type=int, help='Shared boundary treatment strategy')

    parser_rgd = parser.add_argument_group('Regrid options')
    parser_rgd.add_argument("--use_center", action='store_true', help='Use cell centers for nearest neighbor.')
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
    src = SourceData(lon_src, lat_src, elev)

    # Read target grid
    if args.non_supergrid: raise Exception('Only supergrid is supported.')
    print('Reading target grid from ', args.target_grid)
    if args.verbose:
        print(  "'"+args.lon_tgt+"'[::2, ::2]", '-> lonb_tgt')
        print(  "'"+args.lat_tgt+"'[::2, ::2]", '-> latb_tgt')
    lonb_tgt = netCDF4.Dataset(args.target_grid).variables['x'][::2, ::2]
    latb_tgt = netCDF4.Dataset(args.target_grid).variables['y'][::2, ::2]
    if not args.no_mono_lon:
        for ix in range(lonb_tgt.shape[1]-1):
            if lonb_tgt[-1,ix+1]<lonb_tgt[-1,ix]: lonb_tgt[-1,ix+1] += 360.0

    # Domain decomposition
    pe = args.pe
    pe_p = args.pe_p
    if pe_p is None: pe_p = pe
    use_mp = not args.use_serial
    nprocs = args.nprocs

    # Calculation options
    do_effective = False if args.do_center_only else args.do_porous_effective
    topo_gen_args = {'do_center_only':args.do_center_only,
                     'do_effective':do_effective,
                     }

    # Regridding and topo_gen options
    north_pole_lat = args.pole_start
    np_lat_end = args.pole_end
    np_lat_step = args.pole_step
    refine_options = {'use_center': args.use_center,
                      'resolution_limit': not args.no_resolution_limit,
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
    dm = Domain(lon=lonb_tgt, lat=latb_tgt, reentrant_x=True, bipolar_n=True, pole_radius=refine_options['singularity_radius'])
    hm = None
    if args.save_hits:
        hm = HitMap(lon=lon_src, lat=lat_src, from_cell_center=True)

    # Regrid
    bnd_tol_level = args.bnd_tol_level
    if args.do_center_only:
        bnd_tol_level = 0
    dm.regrid_topography(pelayout=pe, tgt_halo=args.tgt_halo, nprocs=nprocs, src=src, src_halo=args.src_halo,
                         refine_loop_args=refine_options, topo_gen_args=topo_gen_args, hitmap=hm,
                         bnd_tol_level=bnd_tol_level, verbose=args.verbose)
    if args.fixed_refine_level<0:
        # Donut update near the (geographic) north pole
        dm.regrid_topography_masked(lat_end=np_lat_end, lat_step=np_lat_step, pelayout=pe_p, nprocs=nprocs, src=src, src_halo=args.src_halo,
                                    refine_loop_args=refine_options, topo_gen_args=topo_gen_args, hitmap=hm, verbose=args.verbose)
    # Output to a netCDF file
    write_output(dm, args.output, do_center_only=args.do_center_only, format='NETCDF3_64BIT_OFFSET', history=' '.join(argv))
    if args.save_hits:
        write_hitmap(hm, 'hitmap.nc')

if __name__ == "__main__":
    main(sys.argv[1:])

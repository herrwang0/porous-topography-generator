import numpy
from .tile_utils import slice_array, normlize_longitude, box_halo
from .topo_regrid import Domain

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

    def stitch_mask_domain(self, mask, rec, halo, config=CalcConfig(), tolerance=2, verbose=False):
        """
        The assumption is the masked domain has the higher refine level
        """
        for rec in self.masks:
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

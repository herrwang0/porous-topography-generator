import numpy as np
import time
import logging

from ptopo.external.thinwall.python import GMesh
from ptopo.external.thinwall.python import ThinWalls
from .roughness import subgrid_roughness_gradient
from .configs import CalcConfig, TileConfig, RefineConfig, NorthPoleConfig
from .domain_mask import NorthPoleMask

import multiprocessing
import functools

logger = logging.getLogger(__name__)

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
        self._hits = np.zeros( self.shape )
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
        return np.all(self[:,:], axis=1)
    def pcolormesh(self, axis, **kwargs):
        """Plots hit map"""
        return axis.pcolormesh( self.lon, self.lat, self[:,:], **kwargs )

def topo_gen(grid, calc_cfg=CalcConfig(), refine_cfg=RefineConfig(), verbosity=0, timers=False):
    """
    Generate topography fields on a domain tile.

    This is the main driver routine for topography generation. It operates
    in-place on the provided ``Domain`` object, computing thin-wall statistics
    and related fields including sub-grid roughness and gradient

    Parameters
    ----------
    grid : Domain
        A self-contained domain tile on which topography will be generated.
        The object is modified in-place.
    calc_cfg : CalcConfig, optional
        Configuration options controlling which topographic quantities are
        computed (e.g., thin walls, effective thin walls, roughness, gradients).
    refine_cfg : RefineConfig, optional
        Configuration options passed to ``GMesh.refine_loop()`` for mesh
        refinement during topography generation.
    verbose : bool, optional
        If True, print progress and status messages to stdout.
    timers : bool, optional
        If True, print timing information for major stages of the computation.

    Returns
    -------
    out : dict
        Dictionary containing the results of the topography generation:

        - ``'tw'`` : Domain
          The input ``grid`` object, updated with computed topographic fields.
        - ``'hits'`` : HitMap
          Map of "hits" produced during the calculation.
    """

    if (not refine_cfg.use_center) and (calc_cfg.calc_roughness or calc_cfg.calc_gradient):
        raise Exception('"use_center" needs to be used for roughness or gradient')

    if timers:
        clock = TimeLog(
            ['setup topo_gen', 'refine grid', 'assign data', 'init thinwalls', 'roughness/gradient', 'total',
             'refine loop (total)', 'refine loop (effective thinwalls)', 'refine loop (coarsen grid)']
        )
        clock.delta('setup topo_gen')

    # ============================================================
    # Refine the grid
    # ============================================================
    if verbosity > 0:
        logger.info(f'topo_gen(): refine grid for tile {grid.bbox.position}')
    levels = grid.refine_loop(
        grid.eds, verbose=(verbosity >= 2), timers=False, mask_res=grid.mask_res, **refine_cfg.to_kwargs()
    )

    # Deepest refine level
    nrfl = levels[-1].rfl
    if timers: clock.delta('refine grid')

    # ============================================================
    # Project elevation to the finest grid
    # ============================================================
    levels[-1].project_source_data_onto_target_mesh(
        grid.eds, use_center=refine_cfg.use_center, work_in_3d=refine_cfg.work_in_3d
    )

    if calc_cfg.save_hits:
        lon_src, lat_src = grid.eds.lon_coord, grid.eds.lat_coord
        hits = HitMap( shape=(lat_src.size, lon_src.size) )
        hits[:] = levels[-1].source_hits(grid.eds, use_center=refine_cfg.use_center, singularity_radius=0.0)
        hits.box = (lat_src.start, lat_src.stop, lon_src.start, lon_src.stop)

    if timers:
        clock.delta('assign data')

    # ============================================================
    # Initialize a ThinWalls object on the finest grid
    # ============================================================
    tw = ThinWalls.ThinWalls( lon=levels[-1].lon, lat=levels[-1].lat, rfl=levels[-1].rfl )

    if refine_cfg.use_center:
        tw.set_cell_mean(levels[-1].height)
        if calc_cfg.calc_thinwalls:
            tw.set_edge_to_step(calc_cfg.thinwalls_interp)
    else:
        tw.set_center_from_corner(levels[-1].height)
        if calc_cfg.calc_thinwalls:
            tw.set_edge_from_corner(levels[-1].height)

    if calc_cfg.calc_effective_tw:
        tw.init_effective_values()

    if timers:
        clock.delta('init thinwalls')

    if verbosity > 0:
        logger.info(f'topo_gen(): coarsen grid for tile {grid.bbox.position}')
        if verbosity >= 2: logger.debug('Refine level {:} {:}'.format(tw.rfl, tw))

    # ============================================================
    # Coarsen to the target grid
    # ============================================================
    if timers: loop_start = clock.now
    for _ in range(nrfl):
        if timers: clock.update_prev()
        if calc_cfg.calc_effective_tw:
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
        tw = tw.coarsen(do_thinwalls=calc_cfg.calc_thinwalls, do_effective=calc_cfg.calc_effective_tw)
        if verbosity >= 2: logger.debug('Refine level {:} {:}'.format(tw.rfl, tw))
        if timers: clock.delta('refine loop (coarsen grid)')
    grid.update_thinwalls_arrays(tw)
    if timers: clock.delta('refine loop (total)', ref_time=loop_start)

    # ============================================================
    # Calculate roughness and gradient
    # ============================================================
    out = subgrid_roughness_gradient(
        levels, grid.c_simple.ave, do_roughness=calc_cfg.calc_roughness, do_gradient=calc_cfg.calc_gradient,
        Idx=grid.Idx, Idy=grid.Idy
    )
    grid.roughness, grid.gradient = out['h2'], out['gh']
    if timers: clock.delta("roughness/gradient")

    # ============================================================
    # Update refine level arrays of the target grid
    # ============================================================
    grid.c_rfl[:] = nrfl
    if calc_cfg.calc_thinwalls:
        grid.u_rfl[:] = nrfl
        grid.v_rfl[:] = nrfl

    if timers:
        clock.delta('total')
        clock.print()

    out = dict.fromkeys(['tw', 'hits'])
    out['tw'] = grid
    if calc_cfg.save_hits: out['hits'] = hits
    return out

def topo_gen_tiles(domain, hm=None, tile_cfg=TileConfig(), calc_cfg=CalcConfig(), refine_cfg=RefineConfig(),
                   verbose=False, debug=False):
    """
    A wrapper function of tiling and topo_gen
    """

    if debug:
        verbosity = 2
    elif verbose:
        verbosity = 1
    else:
        verbosity = 0

    tiles = domain.make_tiles(config=tile_cfg, verbose=verbose)

    twlist, hitlist = [], []
    for tile in tiles.flatten():
        out = topo_gen(
            tile, calc_cfg=calc_cfg, refine_cfg=refine_cfg, verbosity=verbosity, timers=True,
        )
        twlist.append( out['tw'] )
        hitlist.append( out['hits'] )

    domain.stitch_tiles(twlist, tolerance=tile_cfg.bnd_tol_level, config=calc_cfg, verbose=(verbosity == 2))

    if hm and calc_cfg.save_hits:
        hm.stitch_hits(hitlist)

def topo_gen_mp(domain_list, nprocs=None, calc_cfg=CalcConfig(), refine_cfg=RefineConfig(), verbose=False, timers=False):
    """A wrapper for multiprocessing topo_gen"""
    if nprocs is None:
        nprocs = len(domain_list)
    pool = multiprocessing.Pool(processes=nprocs)
    tw_list = pool.map(functools.partial(topo_gen, verbose=verbose, timers=timers, calc_cfg=calc_cfg, refine_cfg=refine_cfg), domain_list)
    pool.close()
    pool.join()

    return tw_list

def progress_north_pole_ring(
        domain, hm, init_masks, np_cfg=NorthPoleConfig(), calc_cfg=CalcConfig(), refine_cfg=RefineConfig(),
        verbosity=0
    ):
    """
    Progressively update topography in latitude rings near the North Pole.

    This routine performs a ring-by-ring topography update starting from an
    initial set of outer latitude masks and moving poleward. At each step,
    a subdomain is constructed between successive latitude rings, topography
    is generated on that subdomain, and the results are stitched back into
    the parent domain.

    The operation modifies ``domain`` in-place.

    Parameters
    ----------
    domain : Domain
        The parent domain on which North Pole topography updates are applied.
        This object is modified in-place.
    hm :  HitMap
          Map of "hits" produced during the calculation.
    init_masks : list of NorthPoleMask
        Initial outer-ring masks defining the starting latitude band.
    np_cfg : NorthPoleConfig, optional
        Configuration controlling North Pole processing, including latitude
        range, ring spacing, and halo size.
    calc_cfg : CalcConfig, optional
        Configuration options controlling which topographic quantities are
        computed.
    refine_cfg : RefineConfig, optional
        Configuration options passed to ``GMesh.refine_loop()`` for mesh
        refinement during topography generation.
    verbosity : bool, optional
        Granularity of logging messages.

    Notes
    -----
    - Latitude rings are generated from ``np_cfg.lat_start`` to
      ``np_cfg.lat_stop`` with spacing ``np_cfg.lat_step``.
    - For each ring, a masked subdomain is created and processed independently.
    - Results are stitched back into the parent domain after each ring update.
    - North-fold boundaries are reconciled after each iteration.

    This function is intended for use in polar-region workflows where
    progressive refinement or masking is required near the North Pole.
    """

    outer_rings = init_masks
    start, stop, step = np_cfg.lat_start, np_cfg.lat_stop, np_cfg.lat_step
    for ring_lat in np.arange(start + step, stop + step, step):
        if verbosity > 0:
            logger.info('Mask latitude: %s', ring_lat)
        inner_rings = NorthPoleMask(domain, count=2, radius=90 - ring_lat)
        for outer, inner in zip(outer_rings, inner_rings):
            mask_domain = domain.make_subdomain(
                bbox=outer.with_halo(halo=np_cfg.pole_halo), norm_lon=True, global_masks=[ inner.to_box() ], subset_eds=False
            )
            topo_gen_tiles(mask_domain, hm, tile_cfg=np_cfg.tile_cfg, calc_cfg=calc_cfg, refine_cfg=refine_cfg, verbose=(verbosity>0), debug=(verbosity>1))

            domain.stitch_mask(mask_domain, verbose=(verbosity == 2))
        outer_rings = inner_rings
        domain.stitch_fold_n( tolerance=np_cfg.tile_cfg.bnd_tol_level, calc_effective=calc_cfg.calc_effective_tw, verbose=(verbosity>1) )
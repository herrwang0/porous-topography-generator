import numpy
import time
from ptopo.external.thinwall.python import GMesh
from ptopo.external.thinwall.python import ThinWalls
from .roughness import subgrid_roughness_gradient
from .configs import CalcConfig, TileConfig, RefineConfig

import multiprocessing
import functools

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

def topo_gen(grid, config=CalcConfig(), refine_config=RefineConfig(), verbose=True, timers=False):
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
        print('topo_gen() for domain {:}'.format(grid.bbox.position))

    if (not refine_config.use_center) and (config.calc_roughness or config.calc_gradient):
        raise Exception('"use_center" needs to be used for roughness or gradient')
    if timers: clock.delta('setup topo_gen')

    # Step 1: Refine grid
    levels = grid.refine_loop(grid.eds, verbose=True, timers=False, mask_res=grid.mask_res, **refine_config.to_kwargs())
    nrfl = levels[-1].rfl
    if timers: clock.delta('refine grid')

    # Step 2: Project elevation to the finest grid
    levels[-1].project_source_data_onto_target_mesh(
        grid.eds, use_center=refine_config.use_center, work_in_3d=refine_config.work_in_3d
    )
    if config.save_hits:
        lon_src, lat_src = grid.eds.lon_coord, grid.eds.lat_coord
        hits = HitMap(shape=(lat_src.size, lon_src.size))
        hits[:] = levels[-1].source_hits(grid.eds, use_center=refine_config.use_center, singularity_radius=0.0)
        hits.box = (lat_src.start, lat_src.stop, lon_src.start, lon_src.stop)
    if timers: clock.delta('assign data')

    # Step 2: Create a ThinWalls object on the finest grid and coarsen back
    tw = ThinWalls.ThinWalls(lon=levels[-1].lon, lat=levels[-1].lat, rfl=levels[-1].rfl)
    if refine_config.use_center:
        tw.set_cell_mean(levels[-1].height)
        if config.calc_thinwalls:
            tw.set_edge_to_step(config.thinwalls_interp)
    else:
        tw.set_center_from_corner(levels[-1].height)
        if config.calc_thinwalls:
            tw.set_edge_from_corner(levels[-1].height)
    if config.calc_effective_tw:
        tw.init_effective_values()
    if timers: clock.delta('init thinwalls')
    if verbose: print('Refine level {:} {:}'.format(tw.rfl, tw))

    if timers: loop_start = clock.now
    for _ in range(nrfl):
        if timers: clock.update_prev()
        if config.calc_effective_tw:
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
        tw = tw.coarsen(do_thinwalls=config.calc_thinwalls, do_effective=config.calc_effective_tw)
        if verbose: print('Refine level {:} {:}'.format(tw.rfl, tw))
        if timers: clock.delta('refine loop (coarsen grid)')
    if timers: clock.delta('refine loop (total)', ref_time=loop_start)

    out = subgrid_roughness_gradient(
        levels, tw.c_simple.ave, do_roughness=config.calc_roughness, do_gradient=config.calc_gradient,
        Idx=grid.Idx, Idy=grid.Idy
    )
    tw.roughness, tw.gradient = out['h2'], out['gh']
    if timers: clock.delta("roughness/gradient")

    # Step 3: Decorate the coarsened ThinWalls object
    tw.bbox = grid.bbox
    tw.mrfl = nrfl

    if timers:
        clock.delta('total')
        clock.print()

    out = dict.fromkeys(['tw', 'hits'])
    out['tw'] = tw
    if config.save_hits: out['hits'] = hits
    return out

def topo_gen_tiles(domain, hm=None, tile_config=TileConfig(), calc_config=CalcConfig(), refine_config=RefineConfig(),
                   verbose=False):
    """
    A wrapper function of tiling and topo_gen
    """
    tiles = domain.make_tiles(
        config=tile_config, verbose=verbose
    )

    twlist, hitlist = [], []
    for tile in tiles.flatten():
        out = topo_gen(
            tile, config=calc_config, refine_config=refine_config, verbose=verbose, timers=True,
        )
        twlist.append( out['tw'] )
        hitlist.append( out['hits'] )

    domain.stitch_tiles(twlist, tolerance=tile_config.bnd_tol_level, config=calc_config, verbose=verbose)

    if hm and calc_config.save_hits:
        hm.stitch_hits(hitlist)

def topo_gen_mp(domain_list, nprocs=None, config=CalcConfig(), refine_config=RefineConfig(), verbose=False, timers=False):
    """A wrapper for multiprocessing topo_gen"""
    if nprocs is None:
        nprocs = len(domain_list)
    pool = multiprocessing.Pool(processes=nprocs)
    tw_list = pool.map(functools.partial(topo_gen, verbose=verbose, timers=timers, config=config, refine_config=refine_config), domain_list)
    pool.close()
    pool.join()

    return tw_list
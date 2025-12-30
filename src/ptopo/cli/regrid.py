import sys
import argparse
import numpy
import netCDF4
from ptopo.external.thinwall.python import GMesh
from ptopo.regrid.kernel import HitMap, TimeLog, CalcConfig, RefineConfig, topo_gen_mp, topo_gen
from ptopo.regrid.output_utils import write_output, write_hitmap
from ptopo.regrid.topo_regrid import Domain
from ptopo.regrid.domain_mask import NorthPoleMask

def add_regrid_parser(subparsers):
    parser = subparsers.add_parser(
        "regrid", description='Objective topography regridding', formatter_class=argparse.RawTextHelpFormatter
    )
    parser.set_defaults(func=regrid)

    parser.add_argument("-v", "--verbose", action='store_true')
    parser.add_argument("--verbosity", default=0, help='Granularity of log output')

    parser_tgt = parser.add_argument_group('Target grid')
    parser_tgt.add_argument("--target-grid", default='', help='File name of the target grid')
    parser_tgt.add_argument("--lon-tgt", default='x', help='Field name in target grid file for longitude')
    parser_tgt.add_argument("--lat-tgt", default='y', help='Field name in target grid file for latitude')
    parser_tgt.add_argument("--non-supergrid", action='store_true',
                            help='If specified, the target grid file is not on a supergrid. Currently not supported')
    parser_tgt.add_argument("--mono-lon", action='store_true',
                            help='If specified, a 360-degree shift will be made to guarantee the last row of lon is monotonic.')
    parser_tgt.add_argument("--tgt-halo", default=0, type=int, help='Halo size at both directions for target grid subdomain')
    parser_tgt.add_argument("--tgt-regional", action='store_true', help='If true, target grid is regional rather than global.')

    parser_src = parser.add_argument_group('Source data')
    parser_src.add_argument("--source", default='', help='File name of the source data')
    parser_src.add_argument("--source-coord", default='', help='File name of the source coordinate (can be different from source)')
    parser_src.add_argument("--lon-src", default='lon', help='Field name in source file for longitude')
    parser_src.add_argument("--lat-src", default='lat', help='Field name in source file for latitude')
    parser_src.add_argument("--src-halo", default=0, type=int, help='Halo size of at both directions for subsetting source data')
    parser_src.add_argument("--elev", default='elevation', help='Field name in source file for elevation')
    parser_src.add_argument("--remove-src-repeat-lon", action='store_true',
                            help=('If specified, the repeating longitude in the last column is removed. '
                                  'Elevation along that longitude will be the mean.'))

    parser_cc = parser.add_argument_group('Calculation options')
    parser_cc.add_argument("--mean-only", action='store_true', help='Output cell-mean topography only')
    parser_cc.add_argument("--do_thinwalls", action='store_true', help='Calculate thin wall parameters')
    parser_cc.add_argument("--thinwalls_interp", default='max', help='Interpolation method for getting thin walls.')
    parser_cc.add_argument("--do_thinwalls_effective", action='store_true', help='Calculate effective depth in porous topography.')
    parser_cc.add_argument("--do_roughness", action='store_true', help='Calculate roughness')
    parser_cc.add_argument("--do_gradient", action='store_true', help='Calculate sub-grid gradient')
    parser_cc.add_argument("--save_hits", action='store_true', help='Save hitmap to a file')

    parser_pe = parser.add_argument_group('Parallelism options')
    # parser_pe.add_argument("--nprocs", default=0, type=int, help='Number of processors used in parallel')
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

    parser.set_defaults(func=regrid)

    return parser

def regrid(args):
    clock = TimeLog(['Read source', 'Read target', 'Setup', 'Regrid main', 'Regrid masked', 'Write output'])

    # Read source data
    print('Reading source data from ', args.source)
    if args.verbose:
        print(  "'"+args.lon_src+"'", '-> lon_src')
        print(  "'"+args.lat_src+"'", '-> lat_src')
        print(  "'"+args.elev+"'", '-> elev')
    if args.source_coord:
        source_coord = args.source_coord
    else:
        source_coord = args.source
    lon_src = netCDF4.Dataset(source_coord)[args.lon_src][:]
    lat_src = netCDF4.Dataset(source_coord)[args.lat_src][:]
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
    if args.do_gradient:
        dxs = netCDF4.Dataset(args.target_grid).variables['dx'][:].data
        dys = netCDF4.Dataset(args.target_grid).variables['dy'][:].data
        dx = dxs[1::2,::2] + dxs[1::2,1::2]
        dy = dys[::2,1::2] + dys[1::2,1::2]
        Idx, Idy = 1.0/dx, 1.0/dy
        Idx[dx==0.0], Idy[dy==0.0] = 0.0, 0.0
    else:
        Idx, Idy = None, None
    if args.tgt_regional:
        tgt_reentrant_x = False
        tgt_fold_n = False
    else:
        tgt_reentrant_x = True
        tgt_fold_n = True
    clock.delta('Read target')

    # Domain decomposition
    pe = args.pe
    pe_p = args.pe_p
    if pe_p is None: pe_p = pe
    nprocs = args.nprocs

    # Calculation options
    calc_config = CalcConfig(
        calc_cell_stats=(not args.mean_only),
        _thinwalls=args.do_thinwalls, _effective_tw=args.do_thinwalls_effective, thinwalls_interp=args.thinwalls_interp,
        calc_roughness=args.do_roughness, calc_gradient=args.do_gradient
    )

    # Regridding and topo_gen options
    north_pole_lat = args.pole_start
    np_lat_end = args.pole_end
    np_lat_step = args.pole_step
    resolution_limit = (not args.no_resolution_limit) and (args.fixed_refine_level<0)
    if args.fixed_refine_level>0:
        resolution_limit = False

    refine_config = RefineConfig(
        use_center = not args.use_corner, resolution_limit=resolution_limit, fixed_refine_level=args.fixed_refine_level,
        work_in_3d=args.refine_in_3d, singularity_radius=90.0-args.pole_start, max_mb=args.max_mb
    )

    if args.verbose:
        refine_config.print_options()
        print('north_pole_lat: ', north_pole_lat)
        print('np_lat_end: ', np_lat_end)
        print('np_lat_step: ', np_lat_step)

    # Create the target grid domain
    domain = Domain(lon=lonb_tgt, lat=latb_tgt, Idx=Idx, Idy=Idy, reentrant_x=tgt_reentrant_x, fold_n=tgt_fold_n, eds=eds)
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
    tiles = domain.make_tiles(
        pelayout=pe, tgt_halo=args.tgt_halo, subset_eds=True, src_halo=args.src_halo, verbose=False
    )
    # clock.delta('Domain decomposition')

    twlist, hitlist = [], []
    for tile in tiles.flatten():
        out = topo_gen(
            tile, refine_config=refine_config, save_hits=args.save_hits, verbose=True, timers=True
        )
        twlist.append( out['tw'] )
        hitlist.append( out['hits'] )

    # if nprocs>1:
    #     twlist = topo_gen_mp(subdomains.flatten(), nprocs=nprocs,
    #                          refine_config=refine_config, save_hits=(not (hm is None)), verbose=True, timers=True, tw_interp=args.thinwalls_interp)
    # else: # with nprocs==1, multiprocessing is not used.
    #     twlist = [topo_gen(sdm, refine_config=refine_config, save_hits=(not (hm is None)), verbose=True, timers=True, tw_interp=args.thinwalls_interp) for sdm in subdomains.flatten()]

    if args.save_hits:
        hm.stitch_hits(hitlist)

    domain.stitch_tiles(
        twlist, tolerance=bnd_tol_level, config=calc_config, verbose=args.verbose
    )

    clock.delta('Regrid main')

    # if args.fixed_refine_level<0:
    #     if args.verbose:
    #         print('Starting regridding masked North Pole')
    #     # Donut update near the (geographic) north pole
    #     dm.regrid_topography_masked(lat_end=np_lat_end, lat_step=np_lat_step, pelayout=pe_p, nprocs=nprocs, tgt_halo=args.tgt_halo, eds=eds, src_halo=args.src_halo,
    #                                 refine_loop_args=refine_config, calc_args={}, hitmap=hm, verbose=args.verbose)
    clock.delta('Regrid masked')

    # Output to a netCDF file
    write_output(
        domain, args.output, config=calc_config, output_refine=True, format='NETCDF3_64BIT_OFFSET', history=' '.join(sys.argv)
    )

    if args.save_hits:
        write_hitmap(hm, 'hitmap.nc')

    clock.delta('Write output')

    clock.print()

# if __name__ == "__main__":
#     main(sys.argv)

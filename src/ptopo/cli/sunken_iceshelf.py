import sys
import numpy as np
import argparse
from netCDF4 import Dataset as ncds

from ptopo.external.thinwall.python import ThinWalls
from ptopo.masking.ice9 import ice9it, copy_var
from ptopo.masking.mask_uv import mask_subgrid
from ptopo.iceshelf.sunken_iceshelf import stack_ice

def add_iceshelf_parser(subparsers):
    parser = subparsers.add_parser(
        "iceshelf",
        help='Sunken Antarctica iceshelf',
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.set_defaults(func=sunken_Antarctica_iceshelf)

    parser.add_argument("--file_topoice", default='', help='land and ice surface')
    parser.add_argument("--file_topobed", default='', help='under-ice seabed')
    parser.add_argument("--file_icethick", default='', help='ice thickness file')
    parser.add_argument("--file_out", default='', help='output file')
    parser.add_argument("--var_topoice", default='depth')
    parser.add_argument("--var_topobed", default='depth')
    parser.add_argument("--var_icethick", default='thk_ice')
    parser.add_argument("--file_subgrid", default=None)
    parser.add_argument("--inverse_depth", default=True, type=bool)
    parser.add_argument("--flood_depth", default=None)

    return parser

def sunken_Antarctica_iceshelf(args):
    """
    Add Antarctica iceshelf thickness to bed topography
    """

    # Topography
    topo_ice = ncds(args.file_topoice)[args.var_topoice][:]
    topo_bed = ncds(args.file_topobed)[args.var_topobed][:]

    # iceshelf
    thk_ice = ncds(args.file_icethick)[args.var_icethick][:]

    topo_ice_sunken = stack_ice(topo_ice, topo_bed, thk_ice, inverse_depth=args.inverse_depth)

    # Mask sub-grid
    if args.file_subgrid is not None:
        tw = ThinWalls.ThinWalls(shape=topo_ice_sunken.shape)

        # variable names are hard-coded for now...
        tw.c_simple.hgh = ncds(args.file_subgrid)['c_simple_hgh'][:]
        # tw.c_simple.ave = ncds(args.file_subgrid)['c_simple_ave'][:]
        tw.c_simple.ave = topo_ice_sunken
        tw.c_simple.low = ncds(args.file_subgrid)['c_simple_low'][:]

        tw.u_simple.hgh = ncds(args.file_subgrid)['u_simple_hgh'][:]
        tw.u_simple.ave = ncds(args.file_subgrid)['u_simple_ave'][:]
        tw.u_simple.low = ncds(args.file_subgrid)['u_simple_low'][:]

        tw.v_simple.hgh = ncds(args.file_subgrid)['v_simple_hgh'][:]
        tw.v_simple.ave = ncds(args.file_subgrid)['v_simple_ave'][:]
        tw.v_simple.low = ncds(args.file_subgrid)['v_simple_low'][:]

        # tw.u_effective.hgh = ncds(fn_depeff)['u_effective_hgh'][:]
        # tw.u_effective.ave = ncds(fn_depeff)['u_effective_ave'][:]
        # tw.u_effective.low = ncds(fn_depeff)['u_effective_low'][:]

        # tw.v_effective.hgh = ncds(fn_depeff)['v_effective_hgh'][:]
        # tw.v_effective.ave = ncds(fn_depeff)['v_effective_ave'][:]
        # tw.u_effective.low = ncds(fn_depeff)['v_effective_low'][:]

        # Mask porous barriers and media where iceshelf is sunken
        mask_subgrid(tw, (thk_ice>0), mask_val=-9999, reentrant_x=True, fold_n=True)

    # ice9 mask
    if isinstance(args.flood_depth, (int, float)):
        ny, nx = topo_ice_sunken.shape
        wet = ice9it(-tw.c_simple.low, start=(ny//2, nx//2), dc=args.flood_depth)
        tw.c_simple.hgh[wet==0.0] = args.flood_depth
        tw.c_simple.ave[wet==0.0] = args.flood_depth
        tw.c_simple.low[wet==0.0] = args.flood_depth

    # write
    ncsrc = ncds(args.file_topoice, 'r')
    ncout = ncds(args.file_out, 'w')

    for name, dimension in ncsrc.dimensions.items():
        ncout.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))

    if args.file_subgrid is not None:
        vars = {
            'c_simple_hgh': tw.c_simple.hgh,
            'c_simple_ave': tw.c_simple.ave,
            'c_simple_low': tw.c_simple.low,
            'u_simple_hgh': tw.u_simple.hgh,
            'u_simple_ave': tw.u_simple.ave,
            'u_simple_low': tw.u_simple.low,
            'v_simple_hgh': tw.v_simple.hgh,
            'v_simple_ave': tw.v_simple.ave,
            'v_simple_low': tw.c_simple.low
        }
    else:
        vars = {args.var_topoice: topo_ice_sunken}

    for vnm, val in vars.items():
        copy_var(ncsrc, ncout, vnm, val)

    varout = ncout.createVariable('nx', np.float64, ('nx',)); varout.cartesian_axis = 'X'
    varout = ncout.createVariable('ny', np.float64, ('ny',)); varout.cartesian_axis = 'Y'

    ncout.history = ' '.join(sys.argv)

    ncout.close()
    ncsrc.close()
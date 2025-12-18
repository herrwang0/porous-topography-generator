import argparse
import numpy as np
import netCDF4
from pathlib import Path
from ptopo.masking.ice9 import ice9it, copy_var, mask_uv

def add_masking_parser(subparsers):
    parser = subparsers.add_parser(
        "ice9",
        help="Flood and mask topography",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--file-in", required=True, help="topo file")
    parser.add_argument("--file-out", default=None, help="output file")
    parser.add_argument("--var-in", default="depth")
    parser.add_argument("--var-out", default=None)
    parser.add_argument("--starting-point", nargs=2, type=int, default=None,
                        help="Starting point (J I)")
    parser.add_argument("--flood-depth", default=0.0, type=float,
                        help="Elevation cutoff (positive above sea level)")
    parser.add_argument("--mask-value", default=None, type=float,
                        help="Depth at dry points")
    parser.add_argument("--do-subgrid", action="store_true",
                        help="Mask subgrid topography")
    parser.add_argument("--subgrid-c-var", action="extend", nargs="+", default=[])
    parser.add_argument("--subgrid-u-var", action="extend", nargs="+", default=[])
    parser.add_argument("--subgrid-v-var", action="extend", nargs="+", default=[])
    parser.add_argument("-q", "--quiet", action="store_true")

    parser.set_defaults(func=run_masking)

    return parser

def run_masking(args):
    verbose = not args.quiet

    var_out = args.var_out or args.var_in

    if args.file_out is None:
        # mask_str = 'msk_{:0.0f}m'.format(-args.flood_depth).replace('-', 'm')
        if args.flood_depth<=0:
            mask_str = f'msk_{np.abs(args.flood_depth):.0f}m'
        else:
            mask_str = f'msk_m{args.flood_depth:.0f}m'
        file_out = Path(args.file_in).stem + '_' + mask_str + '.nc'
    else:
        file_out = args.file_out

    mask_value = -args.flood_depth if args.mask_value is None else args.mask_value

    if verbose:
        print('Generate file ', file_out)
        print('    from ', args.file_in)

    ncsrc = netCDF4.Dataset(args.file_in, 'r')
    depth = ncsrc[args.var_in][:]
    ny, nx = depth.shape

    if args.starting_point is None:
        starting_point = (ny//2, nx//2)
    else:
        starting_point = args.starting_point

    if verbose:
        print('  Starting point (j,i): ', starting_point)
        print('  Wet depth: ', -args.flood_depth)

    maskc = ice9it(-depth, start=starting_point, dc=args.flood_depth, to_mask=True, to_float=False)

    if args.do_subgrid:
        masku, maskv = mask_uv(~maskc, reentrant_x=True, fold_n=True, to_mask=True, to_float=False)

        cvar = dict().fromkeys(args.subgrid_c_var)
        uvar = dict().fromkeys(args.subgrid_u_var)
        vvar = dict().fromkeys(args.subgrid_v_var)

        for vname in args.subgrid_c_var:
            if vname not in ncsrc.variables:
                print(f'  Warning: subgrid cell variable {vname} not found in {args.file_in}, skip.')
                continue
            cvar[vname] = ncsrc[vname][:]
            cvar[vname][maskc] = mask_value

        for vname in args.subgrid_u_var:
            if vname not in ncsrc.variables:
                print(f'  Warning: subgrid u variable {vname} not found in {args.file_in}, skip.')
                continue
            uvar[vname] = ncsrc[vname][:]
            uvar[vname][masku] = mask_value

        for vname in args.subgrid_v_var:
            if vname not in ncsrc.variables:
                print(f'  Warning: subgrid v variable {vname} not found in {args.file_in}, skip.')
                continue
            vvar[vname] = ncsrc[vname][:]
            vvar[vname][maskv] = mask_value

        # csh, csa, csl = ncsrc['c_simple_hgh'][:], ncsrc['c_simple_ave'][:], ncsrc['c_simple_low'][:]
        # ush, usa, usl = ncsrc['u_simple_hgh'][:], ncsrc['u_simple_ave'][:], ncsrc['u_simple_low'][:]
        # vsh, vsa, vsl = ncsrc['v_simple_hgh'][:], ncsrc['v_simple_ave'][:], ncsrc['v_simple_low'][:]

        # csa[maskc], csh[maskc], csl[maskc] = mask_value, mask_value, mask_value
        # ush[masku], usa[masku], usl[masku] = mask_value, mask_value, mask_value
        # vsh[maskv], vsa[maskv], vsl[maskv] = mask_value, mask_value, mask_value
    else:
        depth[maskc] = mask_value

    if verbose:
        print(f"  New topography has {ny*nx-maskc.sum()} out of {ny*nx} wet points.")

    # write
    ncout = netCDF4.Dataset(file_out, 'w')
    for name, dimension in ncsrc.dimensions.items():
        ncout.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))

    varout = ncout.createVariable('nx', np.float64, ('nx',)); varout.cartesian_axis = 'X'
    varout = ncout.createVariable('ny', np.float64, ('ny',)); varout.cartesian_axis = 'Y'

    varout = ncout.createVariable('wet', np.int16, ('ny','nx'))
    varout[:] = np.double(~maskc)
    varout.long_name = 'Values: 1=Ocean, 0=Land'

    if args.do_subgrid:
        varout = ncout.createVariable('wetu', np.int16, ('ny','nxq'))
        varout[:] = np.double(~masku)
        varout.long_name = 'Values: 1=Ocean, 0=Land'

        varout = ncout.createVariable('wetv', np.int16, ('nyq','nx'))
        varout[:] = np.double(~maskv)
        varout.long_name = 'Values: 1=Ocean, 0=Land'

        vars = cvar | uvar | vvar
        for vnm, val in vars.items():
            copy_var(ncsrc, ncout, vnm, val)
    else:
        copy_var(ncsrc, ncout, var_out, depth)

    ncout.history = args.cmdline
    ncout.close()
    ncsrc.close()

    if verbose:
        print('Done ice9')
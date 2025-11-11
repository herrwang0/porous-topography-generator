import sys
import argparse
import numpy as np
import netCDF4
from pathlib import Path

def copy_var(src, dst, varname, value):
    src_var = src.variables[varname]
    dst_var = dst.createVariable(varname, src_var.dtype, src_var.dimensions)

    # Copy attributes
    dst_var.setncatts({attr: src_var.getncattr(attr) for attr in src_var.ncattrs()})
    dst_var[:] = value

def ice9it(depth, start=None, lon=None, lat=None, dc=0, to_mask=False, to_float=False):
    """
    Modified "ice 9" from MOM6-examples/ice_ocean_SIS2/OM4_025/preprocessing/ice9.py

    Parameters:
    ----------
    depth : np.array
        In height unit, i.e. positive for land and negative for ocean.
    start : tuple, optional
        Starting point. Needs to be a point in the connected ocean.
    lon, lat : np.array, optional
        Coordinates of depth. Used to find the indices of Point Nemo as the starting point.
    dc : float, optional
        Critical height for wet. Default is zero.
    to_mask : bool, optional
        If True, convert result to a land/sea mask, i.e. True (1) = dry and False (0) = dry
    to_float : bool, optional
        If True, convert result to a np.double array.

    Output:
    ----------
    wetMask, np.array of np.bool_
        True (1) = wet and False (0) = dry
    """
    (nj, ni) = depth.shape
    wetMask = np.full((nj, ni), False)

    if start is None:
        # Point Nemo
        lon0, lat0 = -123.39333, -48.876667
        idx = (((lon-lon0)%360)**2 + (lat-lat0)**2).argmin()
        iy, ix = idx//ni, idx%ni
        print('Starting location (lon, lat, depth)', (lon[iy,ix], lat[iy,ix], depth[iy,ix]))
    else:
        iy, ix = start

    stack = set()
    stack.add( (iy,ix) )
    while stack:
        (j,i) = stack.pop()
        if wetMask[j,i] or depth[j,i] >= dc: continue
        wetMask[j,i] = True
        if i>0: stack.add( (j,i-1) )
        else: stack.add( (j,ni-1) )
        if i<ni-1: stack.add( (j,i+1) )
        else: stack.add( (0,j) )
        if j>0: stack.add( (j-1,i) )
        if j<nj-1: stack.add( (j+1,i) )
        else: stack.add( (j,ni-1-i) )

    if to_mask: wetMask = ~wetMask
    if to_float: wetMask = np.double(wetMask)
    return wetMask

def mask_uv(wet, reentrant_x=True, fold_n=True, to_mask=False, to_float=False):
    """
    Make wetu and wetv from wet
    wet: 1=Ocean, 0=Land
    """
    ny, nx = wet.shape
    wetu = np.full( (ny, nx+1), False)
    wetv = np.full( (ny+1, nx), False)

    if wet.dtype is not np.bool_:
        wet = wet.astype(bool)

    wetu[:, 1:-1] = (wet[:, 1:] & wet[:, :-1])
    wetv[1:-1, :] = (wet[1:, :] & wet[:-1, :])

    # East/West
    if reentrant_x:
        wetu[:, 0] = (wet[:, 0] & wet[:, -1])
        wetu[:, -1] = wetu[:, 0]
    else:
        wetu[:, 0] = wet[:, 0]
        wetu[:, -1] = wet[:, -1]

    # North
    if fold_n:
        assert nx%2==0, "Cannot do folding north with odd nx."
        wetv[-1, :nx//2] = (wet[-1, :nx//2] & wet[-1, nx:nx//2-1:-1])
        wetv[-1, nx//2:] = wetv[-1, nx//2-1::-1]
    else:
        wetv[-1, :] = wet[-1, :]

    # South
    wetv[0, :] = wet[0, :]

    if to_mask: wetu, wetv = ~wetu, ~wetv
    if to_float: wetu, wetv = np.double(wetu), np.double(wetv)
    return wetu, wetv

def main(argv):
    parser = argparse.ArgumentParser(description='Flood and mask topography',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--file-in", default='', help='topo file')
    parser.add_argument("--file-out", default=None, help='output file')
    parser.add_argument("--var-in", default='depth')
    parser.add_argument("--var-out", default=None)
    parser.add_argument("--starting-point", nargs=2, type=int, default=None, help='Staring point (J,I)')
    parser.add_argument("--flood-depth", default=0.0, type=float, help='elevation (positive above sea level) cutoff')
    parser.add_argument("--mask-value", default=None, type=float, help='depth at dry points')
    parser.add_argument("--do-subgrid", action='store_true', help='mask subgrid topography')
    parser.add_argument("--subgrid-c-var", action="extend", nargs="+", type=str, default=[])
    parser.add_argument("--subgrid-u-var", action="extend", nargs="+", type=str, default=[])
    parser.add_argument("--subgrid-v-var", action="extend", nargs="+", type=str, default=[])
    parser.add_argument("-q", "--quiet", action='store_true')
    args = parser.parse_args(argv[1:])

    verbose = not args.quiet

    if args.var_out is None:
        var_out = args.var_in
    else:
        var_out = args.var_out

    if args.file_out is None:
        # mask_str = 'msk_{:0.0f}m'.format(-args.flood_depth).replace('-', 'm')
        if args.flood_depth<=0:
            mask_str = 'msk_{:0.0f}m'.format(np.abs(args.flood_depth))
        else:
            mask_str = 'msk_m{:0.0f}m'.format(args.flood_depth)
        file_out = Path(args.file_in).stem + '_' + mask_str + '.nc'
    else:
        file_out = args.file_out

    if args.mask_value is None:
        mask_value = -args.flood_depth
    else:
        mask_value = args.mask_value

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
                print('  Warning: subgrid cell variable {:} not found in {:}, skip.'.format(vname, args.file_in))
                continue
            cvar[vname] = ncsrc[vname][:]
            cvar[vname][maskc] = mask_value

        for vname in args.subgrid_u_var:
            if vname not in ncsrc.variables:
                print('  Warning: subgrid u variable {:} not found in {:}, skip.'.format(vname, args.file_in))
                continue
            uvar[vname] = ncsrc[vname][:]
            uvar[vname][masku] = mask_value

        for vname in args.subgrid_v_var:
            if vname not in ncsrc.variables:
                print('  Warning: subgrid v variable {:} not found in {:}, skip.'.format(vname, args.file_in))
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
        print('  New topography has {:} out of {:} wet points.'.format(ny*nx-maskc.sum(), ny*nx))

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

    ncout.history = ' '.join(argv)
    ncout.close()
    ncsrc.close()

    if verbose:
        print('Done ice9')

if __name__ == "__main__":
    main(sys.argv)
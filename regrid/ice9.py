import sys
import argparse
import numpy as np
import netCDF4
from pathlib import Path

def ice9it(depth, start=None, lon=None, lat=None, dc=0):
    """
    Modified "ice 9" from MOM6-examples/ice_ocean_SIS2/OM4_025/preprocessing/ice9.py

    lon, lat : coordinates
    Depth : positive for land and negative for ocean
    dc : critical depth for wet
    """
    wetMask = 0*depth
    (nj, ni) = wetMask.shape

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
        wetMask[j,i] = 1
        if i>0: stack.add( (j,i-1) )
        else: stack.add( (j,ni-1) )
        if i<ni-1: stack.add( (j,i+1) )
        else: stack.add( (0,j) )
        if j>0: stack.add( (j-1,i) )
        if j<nj-1: stack.add( (j+1,i) )
        else: stack.add( (j,ni-1-i) )
    return wetMask

def main(argv):
    parser = argparse.ArgumentParser(description='Flood and mask topography',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--file_in", default='', help='topo file')
    parser.add_argument("--file_out", default=None, help='output file')
    parser.add_argument("--var_in", default='depth')
    parser.add_argument("--var_out", default=None)
    parser.add_argument("--starting_point", nargs=2, type=int, default=None, help='Staring point (J,I)')
    parser.add_argument("--flood_depth", default=0.0, type=float, help='elevation (positive above sea level) cutoff')
    parser.add_argument("-q", "--quiet", action='store_true')
    args = parser.parse_args(argv[1:])

    verbose = not args.quiet

    if args.var_out is None:
        var_out = args.var_in
    else:
        var_out = args.var_out

    if args.file_out is None:
        mask_str = 'msk_{:0.0f}m'.format(-args.flood_depth).replace('-', 'm')
        file_out = Path(args.file_in).stem + '_' + mask_str + '.nc'
    else:
        file_out = args.file_out

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

    wet = ice9it(-depth, start=starting_point, dc=args.flood_depth)
    depth[wet==0] = -args.flood_depth

    if verbose:
        print('  New topography has {:} out of {:} wet points.'.format(wet.sum(), ny*nx))

    # write
    ncout = netCDF4.Dataset(file_out, 'w')

    for name, dimension in ncsrc.dimensions.items():
        ncout.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))

    varout = ncout.createVariable('wet', np.int16, ('ny','nx'))
    varout[:] = wet
    varout.long_name = 'Values: 1=Ocean, 0=Land'

    varout = ncout.createVariable(var_out, np.float64, ('ny','nx'))
    varout[:] = depth
    varout.units = ncsrc[args.var_in].units
    varout.long_name = ncsrc[args.var_in].long_name

    varout = ncout.createVariable('nx', np.float64, ('nx',)); varout.cartesian_axis = 'X'
    varout = ncout.createVariable('ny', np.float64, ('ny',)); varout.cartesian_axis = 'Y'

    ncout.history = ' '.join(argv)
    ncout.close()
    ncsrc.close()

    if verbose:
        print('Done ice9')

if __name__ == "__main__":
    main(sys.argv)
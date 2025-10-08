import sys
import numpy as np
import argparse
from netCDF4 import Dataset as ncds
from regrid.ice9 import ice9it, mask_uv, copy_var

def sunken_Antarctica_iceshelf(file_topoice, file_topobed, file_icethick, file_out,
                               var_topoice='c_simple_ave', var_topobed='c_simple_ave', var_icethick='thk_ice',
                               file_subgrid=None, inverse_depth=True, flood_depth=None, dmask_subgrid=-9999, history=''):
    """Add Antarctica iceshelf thickness to bed topography """

    if inverse_depth:
        depth_to_elev = -1
    else:
        depth_to_elev = 1

    # Topography
    topo_ice = ncds(file_topoice)[var_topoice][:]
    topo_bed = ncds(file_topobed)[var_topobed][:]

    # iceshelf
    thk_ice = ncds(file_icethick)[var_icethick][:]
    ny_ice = (thk_ice.max(axis=1)==0).argmax()

    # New topography
    topo_ice_sunken = topo_ice.copy()
    topo_ice_sunken[:ny_ice,:] = topo_bed[:ny_ice,:] + depth_to_elev * thk_ice[:ny_ice,:]

    # Mask sub-grid
    if file_subgrid is not None:

        # variable names are hard-coded for now...
        csh, csa, csl = ncds(file_subgrid)['c_simple_hgh'][:], ncds(file_subgrid)['c_simple_ave'][:], ncds(file_subgrid)['c_simple_low'][:]
        ush, usa, usl = ncds(file_subgrid)['u_simple_hgh'][:], ncds(file_subgrid)['u_simple_ave'][:], ncds(file_subgrid)['u_simple_low'][:]
        vsh, vsa, vsl = ncds(file_subgrid)['v_simple_hgh'][:], ncds(file_subgrid)['v_simple_ave'][:], ncds(file_subgrid)['v_simple_low'][:]

        # ueh, uea, uel = -ncds(fn_depeff)['u_effective_hgh'][:], -ncds(fn_depeff)['u_effective_ave'][:], -ncds(fn_depeff)['u_effective_low'][:]
        # veh, vea, vel = -ncds(fn_depeff)['v_effective_hgh'][:], -ncds(fn_depeff)['v_effective_ave'][:], -ncds(fn_depeff)['v_effective_low'][:]

        # Mask porous barriers and media where iceshelf is sunken
        maskc = (thk_ice>0)
        masku, maskv = mask_uv(~maskc, reentrant_x=True, fold_n=True, to_mask=True)

        csa = topo_ice_sunken
        csh[maskc] = csa[maskc]
        csl[maskc] = csa[maskc]

        ush[masku], usa[masku], usl[masku] = dmask_subgrid, dmask_subgrid, dmask_subgrid
        vsh[maskv], vsa[maskv], vsl[maskv] = dmask_subgrid, dmask_subgrid, dmask_subgrid

    # ice9 mask
    if isinstance(flood_depth, (int, float)):
        ny, nx = topo_ice_sunken.shape
        wet = ice9it(-csl, start=(ny//2, nx//2), dc=flood_depth)
        csh[wet==0.0], csa[wet==0.0], csl[wet==0.0] = flood_depth, flood_depth, flood_depth

    # write
    ncsrc = ncds(file_topoice, 'r')
    ncout = ncds(file_out, 'w')

    for name, dimension in ncsrc.dimensions.items():
        ncout.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))

    if file_subgrid is not None:
        vars = {'c_simple_hgh': csh,
                'c_simple_ave': csa,
                'c_simple_low': csl,
                'u_simple_hgh': ush,
                'u_simple_ave': usa,
                'u_simple_low': usl,
                'v_simple_hgh': vsh,
                'v_simple_ave': vsa,
                'v_simple_low': vsl}
    else:
        vars = {var_topoice: topo_ice_sunken}

    for vnm, val in vars.items():
        copy_var(ncsrc, ncout, vnm, val)

    varout = ncout.createVariable('nx', np.float64, ('nx',)); varout.cartesian_axis = 'X'
    varout = ncout.createVariable('ny', np.float64, ('ny',)); varout.cartesian_axis = 'Y'

    ncout.history = history

    ncout.close()
    ncsrc.close()

def main(argv):
    parser = argparse.ArgumentParser(description='Sunken Antarctica iceshelf',
                                     formatter_class=argparse.RawTextHelpFormatter)
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
    args = parser.parse_args(argv[1:])

    sunken_Antarctica_iceshelf(**vars(args), history=' '.join(argv))

if __name__ == "__main__":
    main(sys.argv)
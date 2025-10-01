import argparse
import sys
import numpy as np
import netCDF4
from external.nsidc0756.ll2xy import ll2xy
from external.thinwall.python import GMesh
from topo_regrid import Domain

def do_work(source_file, grid_file, output, history):
    # source_file = '/work5/h1w/topo/bedmachine/BedMachineAntarctica-v3.nc'
    # grid_file = '/work/h1w/grid/tides/' + grid
    ddomain = Domain.decompose_domain

    def coarsen( levels, eds, verbose=False ):
        """Coarsens the product of h*f across all levels"""
        levels[-1].project_source_data_onto_target_mesh( eds )
        for k in range( len(levels) - 1, 0, -1 ):
            if verbose: print('Coarsening {} -> {}'.format(k,k-1))
            levels[k].coarsenby2( levels[k-1] )
        return levels[0].height

    def regrid(fullG, Ni, Nj, max_refinement, eds):
        Hcnt = np.zeros((fullG.nj, fullG.ni)) # Diagnostic: counting which cells we are working on
        Htarg = np.zeros((fullG.nj, fullG.ni))
        ci, cj = ddomain(fullG.ni, Ni), ddomain(fullG.nj, Nj)
        gtic = GMesh.GMesh._toc(None,"")
        for j in range( len(cj) ):
            for i in range( len(ci) ):
                (Js, Je), (Is, Ie) = cj[j], ci[i]
                Hcnt[Js:Je,Is:Ie] = Hcnt[Js:Je,Is:Ie] + 1 # Diagnostic: counting which cells we are working on
                G = GMesh.GMesh( lon=fullG.lon[Js:Je+1,Is:Ie+1], lat=fullG.lat[Js:Je+1,Is:Ie+1], is_geo_coord=fullG.is_geo_coord )
                print('J,I={},{} {:.1f}%, {}\n   window lon={}:{}, lat={}:{}\n   jslice={}, islice={}'.format( \
                    j, i, 100*(j*NtileI+i)/(NtileI*NtileJ), G, G.lon.min(), G.lon.max(), G.lat.min(), G.lat.max(), slice(Js,Je), slice(Is,Ie) ))
                # This recursively refines the mesh until some criteria is met ...
                levels = G.refine_loop( eds, resolution_limit=False, fixed_refine_level=max_refinement, timers=False )
                # # Use nearest neighbor topography to populate the finest grid
                # levels[-1].project_source_data_onto_target_mesh( eds, lonlat=False )
                # # Now recursively coarsen
                # h, h2 = rough( levels, levels[-1].height )
                # # Store window in final array
                # Htarg[csj,csi] = h
                # H2targ[csj,csi] = h2
                Htarg[Js:Je,Is:Ie] = coarsen( levels, eds )
        GMesh.GMesh._toc(gtic,"Whole workflow")
        print( Hcnt.min(), Hcnt.max(), '<-- should both be 1 for full model' )
        return Htarg

    # Target
    lonb = netCDF4.Dataset(grid_file)['x'][::2,::2]
    latb = netCDF4.Dataset(grid_file)['y'][::2,::2]

    jj, _ = np.nonzero(latb<=-60)
    Je = jj.max()
    print(Je, latb[Je, 0])

    # Polar stereographic projection
    Xb, Yb = ll2xy(latb[:Je,:], lonb[:Je,:])

    fullG = GMesh.GMesh(lon=Xb, lat=Yb, is_geo_coord=False)
    print(fullG.shape)

    # Max refine level
    dlon, dlat = fullG.max_spacings()
    mfl = GMesh.GMesh.max_refine_levels(dlon, dlat, 500, 500)

    print(mfl.max())

    NtileI, NtileJ, max_refinement = 10, 5, mfl.max().astype(int)
    print('full model nj,ni=',fullG.nj, fullG.ni)

    # Source
    x = netCDF4.Dataset(source_file)['x'][:]
    y = netCDF4.Dataset(source_file)['y'][:]
    xb = np.r_[1.5*x[0]-0.5*x[1], (x[1:]+x[:-1])*0.5, 1.5*x[-1] - 0.5*x[-2]]
    yb = np.r_[1.5*y[0]-0.5*y[1], (y[1:]+y[:-1])*0.5, 1.5*y[-1] - 0.5*y[-2]]

    xcoord = GMesh.RegularCoord(n=x.size, origin=xb[0], periodic=False, delta=500)
    ycoord = GMesh.RegularCoord(n=y.size, origin=yb[-1], periodic=False, delta=500)

    eds = GMesh.UniformEDS()
    eds.lon_coord, eds.lat_coord = xcoord, ycoord

    thk = netCDF4.Dataset(source_file)['thickness'][:]
    sfc = netCDF4.Dataset(source_file)['surface'][:]
    bed = netCDF4.Dataset(source_file)['bed'][:]
    mask  = netCDF4.Dataset(source_file)['mask'][:]
    geoid = netCDF4.Dataset(source_file)['geoid'][:]

    # Fuse ice surface and topography
    bedpice = bed.copy()
    bedpice[sfc>0] = sfc[sfc>0]

    # Regrid ice-cap and lay it over bed
    icecap = thk.copy()
    # mask=3 (floating ice). Only add below sea level ice thickness to bed, so that depth = ice shelf cavity
    icecap[mask==3] = thk[mask==3] - sfc[mask==3]

    eds.data = icecap[::-1,:]
    thk_ice = regrid(fullG, NtileI, NtileJ, max_refinement, eds)

    # write
    ny, nx = lonb.shape[0]-1, lonb.shape[1]-1

    fn = output
    # fn = './ice_thickness_below_sfc_'+grid_name +'_bmv3.nc'
    ncout = netCDF4.Dataset(fn, 'w')
    ncout.createDimension('nx', nx)
    ncout.createDimension('ny', ny)

    vardimx = ncout.createVariable('nx', np.float64, ('nx',)); vardimx.cartesian_axis = 'X'
    vardimy = ncout.createVariable('ny', np.float64, ('ny',)); vardimy.cartesian_axis = 'Y'

    varthk = ncout.createVariable('thk_ice', np.float64, ('ny', 'nx'))
    varthk[:] = 0.0
    varthk[:Je-1, :] = thk_ice

    ncout.history = history
    ncout.close()

def main(argv):
    parser = argparse.ArgumentParser(description='Ice cap thickness',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--source", default='', help='File name of the source data')
    parser.add_argument("--target_grid", default='', help='File name of the target grid')
    parser.add_argument("--output", default='', help='File name of the output')
    args = parser.parse_args(argv[1:])

    do_work(args.source, args.target_grid, args.output, ' '.join(argv))
if __name__ == "__main__":
    main(sys.argv)
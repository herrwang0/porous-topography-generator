def ice9it(lon, lat, depth, dc=0):
    """
    Modified "ice 9" from MOM6-examples/ice_ocean_SIS2/OM4_025/preprocessing/ice9.py

    lon, lat : coordinates
    Depth : positive for land and negative for ocean
    dc : critical depth for wet
    """
    wetMask = 0*depth
    (nj, ni) = wetMask.shape

    # Point Nemo
    lon0, lat0 = -123.39333, -48.876667
    idx = (((lon-lon0)%360)**2 + (lat-lat0)**2).argmin()
    iy, ix = idx//ni, idx%ni
    print('Starting location (lon, lat, depth)', (lon[iy,ix], lat[iy,ix], depth[iy,ix]))

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
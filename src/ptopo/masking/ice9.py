import numpy as np

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
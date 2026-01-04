import numpy as np

def wet_uv(wet, reentrant_x=True, fold_n=True, to_mask=False, to_float=False):
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

def mask_subgrid(tw, mask, mask_val=-9999, reentrant_x=True, fold_n=True):
    """
    Effectively mask out subgrid topography from a cell-center mask
    """
    masku, maskv = wet_uv(~mask, reentrant_x=reentrant_x, fold_n=fold_n, to_mask=True)

    # Set cell-center stats identical
    tw.c_simple.hgh[mask] = tw.c_simple.ave[mask]
    tw.c_simple.low[mask] = tw.c_simple.ave[mask]

    # Set edge stats to mask_val
    tw.u_simple.hgh[masku] = mask_val
    tw.u_simple.ave[masku] = mask_val
    tw.u_simple.low[masku] = mask_val

    tw.v_simple.hgh[maskv] = mask_val
    tw.v_simple.ave[maskv] = mask_val
    tw.v_simple.low[maskv] = mask_val
import numpy as np

def stack_ice(topo_ice, topo_bed, thk_ice, inverse_depth=True):
    """
    Sink Antarctica iceshelf by stacking a layer of ice to bed topography
    """

    if inverse_depth:
        depth_to_elev = -1
    else:
        depth_to_elev = 1

    ny_ice = np.max(np.nonzero(thk_ice.max(axis=1))) if np.any(thk_ice.max(axis=1)) else -1

    # New topography
    topo_ice_sunken = topo_ice.copy()
    topo_ice_sunken[:ny_ice+1,:] = topo_bed[:ny_ice+1,:] + depth_to_elev * thk_ice[:ny_ice+1,:]

    return topo_ice_sunken


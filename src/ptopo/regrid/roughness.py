"""
roughness.py

A function to calculate roughness and gradient.

Modified from code by A. Adcroft
"""

import numpy

def convol( levels, h, f, verbose=False ):
    """Coarsens the product of h*f across all levels"""
    levels[-1].height = ( h * f ).reshape(levels[-1].nj,levels[-1].ni)
    for k in range( len(levels) - 1, 0, -1 ):
        if verbose: print('Coarsening {} -> {}'.format(k,k-1))
        levels[k].coarsenby2( levels[k-1] )
    return levels[0].height

def subgrid_roughness_gradient(levels, depth, do_roughness=True, do_gradient=True, Idx=None, Idy=None, h2min=1.e-7):
    """Calculate subgrid roughness and gradient from a plane-fit

    Parameters
    ----------
    levels : list
        A hierarchy of refined GMesh
    depth : ndarray
        Mean depth on the target grid
    do_roughness : bool, optional
        Switch for roughness
    do_gradient : bool, optional
        Switch for gradient
    Idx, Idy : ndarray, optional
        Inverse of grid spacing in x and y, used to calculate gradient
    h2min : float, optional
        A minmium h2

    Returns
    ----------
    out : dict
        {'h2': roughness, 'gh': gradient}
    """

    out = dict.fromkeys(['h2', 'gh'])
    if not (do_roughness or do_gradient): return out

    nx = 2**( len(levels) - 1 )
    x = ( numpy.arange(nx) - ( nx - 1 ) /2 ) * numpy.sqrt( 12 / ( nx**2 - 1 ) ) # This formula satisfies <x>=0 and <x^2>=1
    X, Y = numpy.meshgrid( x, x )
    X, Y = X.reshape(1,nx,1,nx), Y.reshape(1,nx,1,nx)
    h = levels[-1].height.reshape(levels[0].nj,nx,levels[0].ni,nx)
    HX = convol( levels, h, X ) # mean of h * x
    HY = convol( levels, h, Y ) # mean of h * y

    if do_roughness:
        H2 = convol( levels, h, h ) # mean of h^2
        out['h2'] =  H2 - depth**2 - HX**2 - HY**2 + h2min
    if do_gradient and (Idx and Idy):
        out['gh'] = numpy.sqrt( (HX*Idx)**2 + (HY*Idy)**2)
    return out
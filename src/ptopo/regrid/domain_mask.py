import numpy
from .tile_utils import BoundaryBox

class NorthPoleMask:
    def __init__(self, grid, counts=0, radius=0.25):
        """
        Parameters
        ----------
        grid : GMesh object
            The domain where the mask is to be found.
        counts : integer, optional
            Number of north poles in the target grid. E.g. there are two north poles in the bi-polar cap.
        radius : float
            The radius of the north pole region used to a) decide the mask of north pole in the target grid;
            b) ignore the hits in source grid
        verbose : bool
        """
        self.grid = grid
        self.counts = counts
        self.radius = radius
        self._masks = self._find_north_pole_rectangles()

    def __str__(self):
        disp = [
            f'North Pole rectangles (radius = {self.radius:5.2f}{chr(176):1s}) in {repr(self.grid)}'
        ]
        for box in self:
            idx_str = ','.join([f"{idx:d}" for idx in box])
            disp.append( f'  js,je,is,ie: {idx_str}, shape: ({box[1]-box[0]}, {box[3]-box[2]})' )
        return '\n'.join(disp)

    def __getitem__(self, index):
        return self._masks[index]

    def _find_north_pole_rectangles(self):
        """Returns the extented rectangles of the grids enclosed by a latitudinal circle
        Output:
        ----------
        recs : list of tuples
            Indices of rectangle boxes. Number of boxes depends on num_north_pole.
        """
        grid = self.grid
        jj, ii = numpy.where(grid.lat > (90.0 - self.radius))

        if jj.size==0 or ii.size==0 or self.counts==0:
            recs = []
        elif self.counts==1:
            recs = BoundaryBox( j0=jj.min(), j1=jj.max(), i0=ii.min(), i1=ii.max() )
        elif self.counts==2:
            jjw = jj[ii<grid.ni//2]; iiw = ii[ii<grid.ni//2]
            jje = jj[ii>grid.ni//2]; iie = ii[ii>grid.ni//2]
            assert numpy.all(jjw==jje), 'nj in the two mask domains mismatch.'
            jj = jjw
            assert (jjw.max()==grid.nj), 'mask domains do not reach the north boundary.'
            assert (iiw.min()+iie.max()==grid.ni) and ((iiw.max()+iie.min()==grid.ni)), \
                'ni in the two mask domains mismatch.'
            recs = [
                  BoundaryBox( j0=jj.min(), j1=jj.max(), i0=iiw.min(), i1=iiw.max() ),
                  BoundaryBox( j0=jj.min(), j1=jj.max(), i0=iie.min(), i1=iie.max() )
            ]
            # self.masks = [(jj.min(), 2*grid.nj-jj.min(), iiw.min(), iiw.max()),
            #                    (jj.min(), 2*grid.nj-jj.min(), iie.min(), iie.max())] # extend passing the northern boundary for halos
        else:
            raise Exception('Currently only two north pole rectangles are supported.')
        return recs

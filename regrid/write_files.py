import numpy
import netCDF4

def write_output(domain, filename, mode='center', do_effective=False, do_roughness=False, do_gradient=False, output_refine=True,
                 format='NETCDF3_64BIT_OFFSET', history='', description=None,
                 inverse_sign=True, elev_unit='m', dtype=numpy.float64, dtype_int=numpy.int32):
    """Output to netCDF
    """
    # def clip(field, clip_height=clip_height, positive=(not inverse_sign)):
    #     if isinstance(clip_height, float):
    #         field[field>clip_height] = clip_height
    #     return (float(positive)*2-1)*field
    def signed(field, positive=(not inverse_sign)):
        return (float(positive)*2-1)*field

    def write_variable(nc, field, field_name, field_dim, units=elev_unit, long_name=' ', dtype=dtype):
        if field_dim == 'c':
            dim = ('ny', 'nx')
        elif field_dim == 'u':
            dim = ('ny', 'nxq')
        elif field_dim == 'v':
            dim = ('nyq', 'nx')
        else: raise Exception('Wrong field dimension.')
        varout = nc.createVariable(field_name, dtype, dim)
        varout[:] = field
        varout.units = units
        varout.long_name = long_name

    ny, nx = domain.shape

    ncout = netCDF4.Dataset(filename, mode='w', format=format)
    ncout.createDimension('nx', nx)
    ncout.createDimension('ny', ny)
    ncout.createDimension('nxq', nx+1)
    ncout.createDimension('nyq', ny+1)

    varout = ncout.createVariable('nx', numpy.float64, ('nx',)); varout.cartesian_axis = 'X'
    varout = ncout.createVariable('ny', numpy.float64, ('ny',)); varout.cartesian_axis = 'Y'

    if mode == 'mean':
        description = "Mean topography at cell-centers" if description is None else description
        write_variable(
            ncout, signed(domain.c_simple.ave), 'depth', 'c', long_name='Cell-center mean topography')

    elif mode == 'center':
        description = "Min, mean and max elevation at cell-centers" if description is None else description
        write_variable(
            ncout, signed(domain.c_simple.ave), 'depth', 'c', long_name='Simple cell-center mean topography')
        write_variable(
            ncout, signed(domain.c_simple.hgh), 'depth_hgh', 'c', long_name='Simple cell-center highest topography')
        write_variable(
            ncout, signed(domain.c_simple.low), 'depth_low', 'c', long_name='Simple cell-center lowest topography')

    elif mode == 'all':
        description = "Min, mean and max elevation at cell-centers and u/v-edges"  if description is None else description
        # cell-centers
        write_variable(
            ncout, signed(domain.c_simple.ave), 'c_simple_ave', 'c', long_name='Simple cell-center mean topography')
        write_variable(
            ncout, signed(domain.c_simple.hgh), 'c_simple_hgh', 'c', long_name='Simple cell-center highest topography')
        write_variable(
            ncout, signed(domain.c_simple.low), 'c_simple_low', 'c', long_name='Simple cell-center lowest topography')

        # u-edges
        write_variable(
            ncout, signed(domain.u_simple.ave), 'u_simple_ave', 'u', long_name='Simple u-edge mean topography')
        write_variable(
            ncout, signed(domain.u_simple.hgh), 'u_simple_hgh', 'u', long_name='Simple u-edge highest topography')
        write_variable(
            ncout, signed(domain.u_simple.low), 'u_simple_low', 'u', long_name='Simple u-edge lowest topography')

        # v-edges
        write_variable(
            ncout, signed(domain.v_simple.ave), 'v_simple_ave', 'v', long_name='Simple v-edge mean topography')
        write_variable(
            ncout, signed(domain.v_simple.hgh), 'v_simple_hgh', 'v', long_name='Simple v-edge highest topography')
        write_variable(
            ncout, signed(domain.v_simple.low), 'v_simple_low', 'v', long_name='Simple v-edge lowest topography')

        if do_effective:
            # cell-centers
            write_variable(
                ncout, signed(domain.c_effective.ave), 'c_effective_ave', 'c', long_name='Effective cell-center mean topography')
            write_variable(
                ncout, signed(domain.c_effective.hgh), 'c_effective_hgh', 'c', long_name='Effective cell-center highest topography')
            write_variable(
                ncout, signed(domain.c_effective.low), 'c_effective_low', 'c', long_name='Effective cell-center lowest topography')

            # u-edges
            write_variable(
                ncout, signed(domain.u_effective.ave), 'u_effective_ave', 'u', long_name='Effective u-edge mean topography')
            write_variable(
                ncout, signed(domain.u_effective.hgh), 'u_effective_hgh', 'u', long_name='Effective u-edge highest topography')
            write_variable(
                ncout, signed(domain.u_effective.low), 'u_effective_low', 'u', long_name='Effective u-edge lowest topography')

            # v-edges
            write_variable(
                ncout, signed(domain.v_effective.ave), 'v_effective_ave', 'v', long_name='Effective v-edge mean topography')
            write_variable(
                ncout, signed(domain.v_effective.hgh), 'v_effective_hgh', 'v', long_name='Effective v-edge highest topography')
            write_variable(
                ncout, signed(domain.v_effective.low), 'v_effective_low', 'v', long_name='Effective v-edge lowest topography')

    else:
        raise ValueError(f"write_output: unknown mode = {mode}")

    # refinement levels
    if output_refine:
        write_variable(
            ncout, domain.c_rfl, 'c_rfl', 'c', long_name='Refinement level at cell-centers', units='nondim', dtype=dtype_int)
        if mode == 'all':
            write_variable(
                ncout, domain.u_rfl, 'u_rfl', 'u', long_name='Refinement level at u-edges', units='nondim', dtype=dtype_int)
            write_variable(
                ncout, domain.v_rfl, 'v_rfl', 'v', long_name='Refinement level at v-edges', units='nondim', dtype=dtype_int)

    if do_roughness:
        write_variable(
            ncout, domain.roughness, 'h2', 'c', long_name='Sub-grid plane-fit roughness', units='m2')
    if do_gradient:
        write_variable(
            ncout, domain.gradient, 'gradh', 'c', long_name='Sub-grid plane-fit gradient', units='nondim')

    ncout.description = description
    ncout.history = history
    ncout.close()

def write_hitmap(hitmap, filename, format='NETCDF3_64BIT_OFFSET', history='', description='', dtype=numpy.int32):
    """Output to netCDF
    """
    if description=='':
        description = 'Map of "hits" of the source data'

    ny, nx = hitmap.shape

    ncout = netCDF4.Dataset(filename, mode='w', format=format)
    ncout.createDimension('nx', nx)
    ncout.createDimension('ny', ny)
    ncout.createDimension('nxq', nx+1)
    ncout.createDimension('nyq', ny+1)

    varout = ncout.createVariable('hitmap', dtype, ('ny','nx'))
    varout[:] = hitmap[:]
    varout.units = 'nondim'
    varout.long_name = '>0 source data is used; 0 not'

    varout = ncout.createVariable('lat', numpy.float64, ('nyq',))
    varout[:] = hitmap.lat[:,0]
    varout.units = 'degree'
    varout.long_name = 'Latitude at the nodes'

    varout = ncout.createVariable('lon', numpy.float64, ('nxq',))
    varout[:] = hitmap.lon[0,:]
    varout.units = 'degree'
    varout.long_name = 'Longitude at the nodes'

    ncout.description = description
    ncout.history = history
    ncout.close()

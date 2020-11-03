import os
import warnings
import cftime

import calendar

import scipy.odr as odr
from scipy import stats

import numpy as np
import xarray as xr


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    lon1, lat1 := scalars
    lon2, lat2 := 1D arrays
    """

    Re = 6378.137

    # convert decimal degrees to radians
    deg2rad = np.pi / 180.
    lon1 = np.array(lon1) * deg2rad
    lat1 = np.array(lat1) * deg2rad
    lon2 = np.array(lon2) * deg2rad
    lat2 = np.array(lat2) * deg2rad

    if lon2.shape:
        N = lon2.shape[0]
        lon1 = np.repeat(lon1, N)
        lat1 = np.repeat(lat1, N)

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.)**2. + np.cos(lat1) * \
        np.cos(lat2) * np.sin(dlon / 2.)**2.
    c = 2. * np.arcsin(np.sqrt(a))
    km = Re * c
    return km


def points_in_range(clon, clat, plon, plat, range_km):
    """Find points within range of a point."""
    if hasattr(clon, '__len__'):
        m = np.zeros(plon.shape, dtype=bool)
        for cx, cy in zip(clon, clat):
            mi = points_in_range(cx, cy, plon, plat, range_km)
            m = (m | mi)
    else:
        mask = np.array(
            (haversine(
                clon,
                clat,
                plon.ravel(),
                plat.ravel()) <= range_km))
        if mask.ndim != plon.ndim:
            m = mask.reshape(plon.shape)
        else:
            m = mask
    return m


def _is_p_inside_points_hull(points, p):
    """Return points inside convex hull."""

    from scipy.spatial import ConvexHull

    hull = ConvexHull(points)
    new_points = np.append(points, p, axis=0)
    new_hull = ConvexHull(new_points)
    if list(hull.vertices) == list(new_hull.vertices):
        return True
    else:
        return False


def griddata(x, y, z, xgrid_edges, ygrid_edges, use_rbf=True, smooth=0, trim_to_hull=True):
    """Bin data to a 2D grid and interpolate with a radial basis function
       interpolant.
    """

    import scipy.interpolate

    # get rid of missing values
    mask = np.isnan(x) | np.isnan(y) | np.isnan(z)
    x = x[~mask]
    y = y[~mask]
    z = z[~mask]

    # bin the data and compute means within each bin
    N, _, _ = np.histogram2d(x, y, bins=[xgrid_edges, ygrid_edges], density=False)
    S, _, _ = np.histogram2d(x, y, bins=[xgrid_edges, ygrid_edges], weights=z, density=False)
    N[N == 0] = np.nan
    ZI = S / N

    # compute grid-cell centers
    xi = np.vstack((xgrid_edges[:-1], xgrid_edges[1:])).mean(axis=0)
    yi = np.vstack((ygrid_edges[:-1], ygrid_edges[1:])).mean(axis=0)
    XI, YI = np.meshgrid(xi, yi)
    ZI = ZI.transpose()
    N = N.transpose()
    
    if not use_rbf:
        return XI, YI, ZI, N
    
    # interpolate
    xb = XI.ravel()[~np.isnan(ZI.ravel())]
    yb = YI.ravel()[~np.isnan(ZI.ravel())]
    zb = ZI.ravel()[~np.isnan(ZI.ravel())]

    rbf = scipy.interpolate.Rbf(xb, yb, zb, function='linear', smooth=smooth)
    ZI = rbf(XI, YI)

    # trim the intoplated field with convex hull of input points
    if trim_to_hull:
        points = np.array([x, y]).transpose()
        p = np.empty((1, 2))
        for i in range(len(xi)):
            for j in range(len(yi)):
                p[0, :] = [XI[j, i], YI[j, i]]
                if not _is_p_inside_points_hull(points, p):
                    ZI[j, i] = np.nan

    return XI, YI, ZI, N

   
def lat_weights_regular_grid(lat):
    """
    Generate latitude weights for equally spaced (regular) global grids.
    Weights are computed as sin(lat+dlat/2)-sin(lat-dlat/2) and sum to 2.0.
    """   
    dlat = np.abs(np.diff(lat))
    np.testing.assert_almost_equal(dlat, dlat[0])
    w = np.abs(np.sin(np.radians(lat + dlat[0] / 2.)) - np.sin(np.radians(lat - dlat[0] / 2.)))

    if np.abs(lat[0]) > 89.9999: 
        w[0] = np.abs(1. - np.sin(np.radians(np.pi / 2 - dlat[0])))

    if np.abs(lat[-1]) > 89.9999:
        w[-1] = np.abs(1. - np.sin(np.radians(np.pi / 2 - dlat[0])))

    return w


def compute_grid_area(ds, check_total=True):
    """Compute the area of grid cells.
    
    Parameters
    ----------
    
    ds : xarray.Dataset
      Input dataset with latitude and longitude fields
    
    check_total : Boolean, optional
      Test that total area is equal to area of the sphere.
      
    Returns
    -------
    
    area : xarray.DataArray
       DataArray with area field.
    
    """
    
    radius_earth = 6.37122e6 # m, radius of Earth
    area_earth = 4.0 * np.pi * radius_earth**2 # area of earth [m^2]e
        
    def infer_lon_name(ds):
        lon_names = ['longitude', 'lon']
        for n in lon_names:
            if n in ds:
                return n
        raise ValueError('could not determine lon name')  
    
    def infer_lat_name(ds):
        lat_names = ['latitude', 'lat']
        for n in lat_names:
            if n in ds:
                return n
        raise ValueError('could not determine lat name')    

    lon_name = infer_lon_name(ds)       
    lat_name = infer_lat_name(ds)        
    
    weights = lat_weights_regular_grid(ds[lat_name])
    area = weights + 0.0 * ds[lon_name] # add 'lon' dimension
    area = (area_earth / area.sum(dim=(lat_name, lon_name))) * area
    
    if check_total:
        np.testing.assert_approx_equal(np.sum(area), area_earth)
        
    return xr.DataArray(area, dims=(lat_name, lon_name), attrs={'units': 'm^2', 'long_name': 'area'})  

    
def day_of_year_noleap(dates):
    """Convert dates to Day of Year (omitting leap days)."""
    d0 = np.datetime64('2001-01-01') - 1
    doy_list = []
    months = (dates.astype('datetime64[M]').astype(int) % 12 + 1).astype(int)
    days = (dates.astype('datetime64[D]') - dates.astype('datetime64[M]') + 1).astype(int)

    for mm, dd in zip(months, days):
        if mm == 2 and dd == 29:
            doy = 0
        else:
            d = np.datetime64(f'2001-{mm:02d}-{dd:02d}')
            doy = ((d - d0) / np.timedelta64(1, 'D')).astype(int)
        doy_list.append(doy)

    return doy_list


def daily_climatology(ds):
    """Compute a day-of-year climatology."""

    dso = xr.Dataset()
    ds = ds.copy()
    doy = day_of_year_noleap(ds.time.values)
    ds['doy'] = xr.DataArray(doy, dims=('time'))

    # copy coords
    for v in ds.coords:
        if 'time' not in ds[v].dims:
            dso[v] = ds[v].copy()
    
    first_var = True
    for v in ds.variables:
        if 'time' not in ds[v].dims:
            dso[v] = ds[v].copy()
            continue            
        elif v not in ['doy', 'time']:
            shape = list(ds[v].shape)
            dims = ds[v].dims
            shape[0] = 365

            dso[v] = xr.DataArray(np.empty(shape), dims=dims)
            count = np.zeros((365,))
            for doy, idx in ds.groupby('doy').groups.items():
                if doy == 0:
                    if first_var:
                        print('skipping leap days')
                else:
                    count[doy-1] += len(idx)
                    dso[v].data[doy-1,...] = ds[v].isel(time=idx).mean('time')
                    
            first_var = False

        dso['time'] = xr.DataArray(np.arange(1, 366, 1), dims=('time'))

    return dso


def austral_year_daily(x, y):
    """rearrange time to put austral summer in the middle."""
    if isinstance(x, xr.DataArray):
        x = x.values
    
    jfmamj = x < 182.
    jasond = x >= 182.
    
    x_jasond = []
    y_jasond = []
    if any(jasond):
        x_jasond = x[jasond] - 181
        y_jasond = y[jasond]

    x_jfmamj = []
    y_jfmamj = []
    if any(jfmamj):
        x_jfmamj = x[jfmamj] + 184
        y_jfmamj = y[jfmamj]

    xout = np.concatenate([xi for xi in [x_jasond, x_jfmamj] if len(xi)])
    yout = np.concatenate([yi for yi in [y_jasond, y_jfmamj] if len(yi)])
    
    return xout, yout


def austral_year_monthly(*args):
    if len(args) == 1:
        return _antyear_monthly_y(args[0])
    else:
        x = args[0]
        y = args[1]
        jfmamj = x <= 6
        jasond = x > 6
        oargs = [np.concatenate((x[jasond]-6, x[jfmamj]+6))]
        for arg in args[1:]:
            oargs.append(np.concatenate((arg[jasond], arg[jfmamj])))
            
        return tuple(oargs)


def _austral_year_monthly_y(y):
    jfmamj = slice(0, 6)
    jasond = slice(6, 12)
    return np.concatenate((y[jasond], y[jfmamj]))


def doy_midmonth():
    eomday = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]).cumsum()
    bomday = np.array([1, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]).cumsum()
    return np.vstack((eomday, bomday)).mean(axis=0)    


def eomday(year, month):
    """end of month day"""
    if hasattr(year, '__iter__'):
        assert hasattr(month, '__iter__')
        return np.array([calendar.monthrange(y, m)[-1] for y, m in zip(year, month)])
    else:
        return calendar.monthrange(year, month)[-1]


def year_frac(year, month, day):
    """compute year fraction"""
    
    time_units='days since 0001-01-01 00:00:00'
    to_datenum = lambda y, m, d: cftime.date2num(cftime.datetime(y, m, d), units=time_units)
    nday_per_year = lambda y: 366 if eomday(y, 2) == 29 else 365
    
    if hasattr(year, '__iter__'):
        assert hasattr(month, '__iter__')
        assert hasattr(day, '__iter__')
        t0_year = np.array([to_datenum(y, 1, 1) - 1 for y in year])
        t_year = np.array([to_datenum(y, m, d) for y, m, d in zip(year, month, day)])
        nday_year = np.array([nday_per_year(y) for y in year])
    else:
        t0_year = to_datenum(year, 1, 1) - 1
        t_year = to_datenum(year, month, day)
        nday_year = nday_per_year(year)
    
    return year + (t_year - t0_year) / nday_year


def ann_mean(ds, season=None, time_bnds_varname='time_bnds', time_centered=True, n_req=None):
    """Compute annual means, or optionally seasonal means"""
    
    ds = ds.copy() #deep=True)

    if n_req is None:
        if season is not None:
            n_req = 2
        else:
            n_req = 8
    
    if time_bnds_varname is None and not time_centered:
        raise NotImplementedError('time_bnds_varname cannot be "None" if time_centered=False')
        
    if not time_centered:
        time_units = ds.time.encoding['units']
        time_calendar = ds.time.encoding['calendar']

        # compute time bounds array
        time_bound_data = cftime.date2num(
                ds[time_bnds_varname].data, 
                units=time_units, 
                calendar=time_calendar)    

        # center time
        time_centered = cftime.num2date(
            time_bound_data.mean(axis=1),
            units=time_units, 
            calendar=time_calendar
        )        
        time_attrs = ds.time.attrs
        time_encoding = ds.time.encoding

        ds['time'] = xr.DataArray(
            time_centered,
            dims=('time')
        )    
    
    ones = xr.DataArray(
        np.ones((len(ds.time))), 
        dims=('time'), 
        coords={'time': ds.time},
    )
    time_mask = xr.DataArray(
        np.ones((len(ds.time))), 
        dims=('time'), 
        coords={'time': ds.time},
    )

    group_by_year = 'time.year'
    rename = {'year': 'time'}
    
    if season is not None:
        season = season.upper()
        if season not in ['DJF', 'MAM', 'JJA', 'SON']:
            raise ValueError(f'unknown season: {season}')            

        ds['austral_year'] = xr.where(ds['time.month'] > 6, ds['time.year'] + 1, ds['time.year'])
        ds = ds.set_coords('austral_year')
        ones = ones.assign_coords({'austral_year': ds.austral_year})
        time_mask = time_mask.assign_coords({'austral_year': ds.austral_year})
        time_mask = time_mask.where(ds['time.season'] == season).fillna(0)
        
        if season == 'DJF':
            group_by_year = 'austral_year'
            rename = {'austral_year': 'time'}
            
    if time_bnds_varname is not None:
        time_wgt = ds[time_bnds_varname].diff(dim=ds[time_bnds_varname].dims[1])
        if time_wgt.dtype == '<m8[ns]':
            time_wgt = time_wgt / np.timedelta64(1, 'D')
    else:        
        time_wgt = xr.DataArray(
            np.ones((len(ds.time))), 
            dims=('time'), 
            coords={'time': ds.time},
        )
        time_wgt = time_wgt.assign_coords(
            {c: da for c, da in ds.coords.items() if 'time' in da.dims}
        )
                       
    time_wgt = time_wgt.where(time_mask==1) #.fillna(0.)

    ones = ones.where(time_mask==1)
    time_wgt_grouped = time_wgt.groupby(group_by_year, restore_coord_dims=False)
    time_wgt = time_wgt_grouped / time_wgt_grouped.sum(dim=xr.ALL_DIMS)

    nyr = len(time_wgt_grouped.groups)
         
    time_wgt = time_wgt.squeeze()

    idx_not_nans = ~np.isnan(time_wgt)
    sum_wgt = time_wgt.groupby(group_by_year).sum(dim=xr.ALL_DIMS)
    idx_not_nans = (sum_wgt > 0)

    np.testing.assert_almost_equal(
        sum_wgt[idx_not_nans], 
        np.ones(idx_not_nans.sum().values)
    )

    nontime_vars = set([v for v in ds.variables if 'time' not in ds[v].dims]) - set(ds.coords)
    dsop = ds.drop_vars(nontime_vars)

    if time_bnds_varname is not None:
        dsop = dsop.drop_vars(time_bnds_varname)    
    
    def weighted_mean_arr(darr, wgts=None):
        # if NaN are present, we need to use individual weights
        cond = darr.isnull()
        ones = xr.where(cond, 0.0, 1.0)
        if season is None:
            mask = (
                darr.resample({'time': 'A'}, restore_coord_dims=False).mean(dim='time').notnull()
            )
            da_sum = (
                (darr * wgts).resample({'time': 'A'}, restore_coord_dims=False).sum(dim='time')
            )
            ones_out = (
                (ones * wgts).resample({'time': 'A'}, restore_coord_dims=False).sum(dim='time')
            )
            count = (
                (ones * wgts.notnull()).resample({'time': 'A'}, restore_coord_dims=False).sum(dim='time')
            )
        else:
            mask = (
                darr.groupby(group_by_year, restore_coord_dims=False).mean(dim='time').notnull()
            ).rename(rename)
            
            da_sum = (
                (darr * wgts).groupby(group_by_year, restore_coord_dims=False).sum(dim='time')
            ).rename(rename)
            
            ones_out = (
                (ones * wgts).groupby(group_by_year, restore_coord_dims=False).sum(dim='time')
            ).rename(rename)
            
            count = (
                 (ones * wgts.notnull()).groupby(group_by_year, restore_coord_dims=False).sum(dim='time')
            ).rename(rename)

        ones_out = ones_out.where(ones_out > 0.0)
        da_weighted_mean = da_sum / ones_out

        return da_weighted_mean.where(mask).where(count >= n_req)    

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ds_ann = dsop.map(weighted_mean_arr, wgts=time_wgt)

    # copy attrs
    for v in ds_ann:
        ds_ann[v].attrs = ds[v].attrs

    # restore coords
    ds_ann = xr.merge((ds_ann, ds[list(nontime_vars)]))

    # eliminate partials
    ndx = (time_wgt_grouped.count(dim=xr.ALL_DIMS) >= n_req).values
    if not ndx.all():
        ds_ann = ds_ann.isel(time=ndx)

    return ds_ann


def datetime64_parts(da_time):
    """convert datetime64 to year, month, day
       accepts multiple types    
    """
    if isinstance(da_time, xr.DataArray):
        return _datetime64_parts_arr(da_time.values)
    elif isinstance(da_time, list) | isinstance(da_time, tuple):
        return _datetime64_parts_arr(np.array(da_time))
    else:
        return _datetime64_parts_arr(da_time)

    
def _datetime64_parts_arr(da_time):
    year = da_time.astype('datetime64[Y]').astype(int) + 1970
    month = (da_time.astype('datetime64[M]').astype(int) % 12 + 1).astype(int)    
    day = (da_time.astype('datetime64[D]') - da_time.astype('datetime64[M]') + 1).astype(int)
    return year, month, day


def list_set(seq):
    """return a list with unique entries, but don't change the order"""
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def mavg_periodic(x, n):
    nx = len(x)
    x = np.concatenate((x, x, x))
    return np.convolve(x, np.ones((n,))/n, mode='same')[nx:nx*2]


def mavg_periodic_ds(dset, n):   
    dso = dset.copy()
    for v in dset.data_vars:
        if 'time' in dset[v].dims and v != 'time':
            non_time_dims = [d for d in dset[v].dims if d != 'time']
            if non_time_dims:
                data = dset[v].stack(non_time_dims=non_time_dims)
                data_out = data.copy()
                for i in range(len(data.non_time_dims)):
                    data_out[:, i] = mavg_periodic(data[:, i], n)
                dso[v].data = data_out.unstack('non_time_dims')
            else:
                dso[v].data = mavg_periodic(dset[v], n)

    return dso


class linreg_odr(object):
    
    
    """Perform Orthogonal distance regression"""
    def __init__(self, x, y, xerr=None, yerr=None):
        
        self.n = len(x)
        self.data = odr.Data(x, y, wd=xerr, we=yerr) 
        self.odr = odr.ODR(self.data, odr.unilinear).run()
        
        self.beta = self.odr.beta
        self.cov_beta = self.odr.cov_beta
        self.stderr_beta = self.odr.sd_beta

        self.xhat = np.sort(x)
        self.yhat = self.predict(self.xhat)
        
        self.r2 = self._calc_r2()
        self.rmse = self._calc_rmse()
        self.pval = self._calc_pval()
        
    def predict(self, xp):
        return self.beta[0] * xp + self.beta[1]

    def _calc_rmse(self):
        sse = np.sum((self.data.y - self.predict(self.data.x))**2)
        return np.sqrt(sse / self.n)

    def _calc_pval(self):
        """Compute p value of slope"""
        t = self.beta / self.stderr_beta
        return (2. * (1. - stats.t.cdf(np.abs(t), self.n - 2)))[0]
            
    def _calc_r2(self):
        """compute coefficient of determination"""
        sse = np.sum((self.data.y - self.predict(self.data.x))**2)
        sst = np.sum((self.data.y - self.data.y.mean())**2)
        return (1. - sse/sst)  
    
    def to_dict(self):
        persist_keys = [
            'beta',
            'stderr_beta',
            'cov_beta',
            'r2',
            'rmse',
            'pval',
        ]
        return {k: self.__dict__[k] for k in persist_keys}        
       

def canvas(*args, figsize=(6, 4)):
    """generate a figure with subplots"""

    assert len(args), 'Args required'
    assert len(args) <= 2, 'Too many args'
    
    if len(args) == 2:
        nrow = args[0]
        ncol = args[1]
    else:
        npanel = args[0]
        nrow = int(np.sqrt(npanel))
        ncol = int(npanel/nrow) + min(1, npanel%nrow)
        
    return plt.subplots(
        nrow, ncol, 
        figsize=(figsize[0]*ncol, figsize[1]*nrow),                       
        constrained_layout=False,
        squeeze=False,
    )                


def label_plots(fig, axs, xoff=-0.04, yoff=0.02):
    alp = [chr(i).upper() for i in range(97,97+26)]
    for i, ax in enumerate(axs):    
        p = ax.get_position()
        x = p.x0 + xoff
        y = p.y1 + yoff
        fig.text(x, y , f'{alp[i]}',
                 fontsize=14,
                 fontweight='semibold')    
        
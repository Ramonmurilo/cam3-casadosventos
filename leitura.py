#import plot_cam3
#import recorte

import xarray as xr

dado = xr.open_dataset('dados/r211104.cam2.h0.2021-11.nc')
print(dado.groupby('time.month'))



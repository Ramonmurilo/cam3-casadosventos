import rioxarray
from shapely.geometry.geo import shape
from xarray.core import dataset
import sys
from typing import Tuple

def preparar_para_recorte_grib(dataset:dataset, crs="epsg:4326", xdim="longitude", ydim="latitude") -> dataset:
    """Trata coordenadas e dados para recorte | funcional para arquivos grib somente.
    Args:
        dataset (dataset): Dataset com a variável de interesse selecionada.
        crs (str, optional): Projeção cartográfica. Defaults to "epsg:4326".
        xdim (str, optional): Nome da dimensão x. Defaults to "longitude".
        ydim (str, optional): Nome da dimensão y. Defaults to "latitude".
    Returns:
        dataset: DataSet pré-cortado
    """
    dataset = dataset.assign_coords(longitude=(((dataset.longitude + 180) % 360) - 180)).sortby(xdim)
    dataset = dataset.rio.set_spatial_dims(x_dim=xdim, y_dim=ydim) 
    dataset = dataset.rio.write_crs(crs)
    return dataset

def preparar_para_recorte_nc(dataset:dataset, crs="epsg:4326", xdim="lon", ydim="lat") -> dataset:
    """Trata coordenadas e dados para recorte | funcional para arquivos netcdf4 somente.
    Args:
        dataset (dataset): Dataset com a variável de interesse selecionada.
        crs (str, optional): Projeção cartográfica. Defaults to "epsg:4326".
        xdim (str, optional): Nome da dimensão x. Defaults to "lon".
        ydim (str, optional): Nome da dimensão y. Defaults to "lat".
    Returns:
        dataset: DataSet pré-cortado
    """
    dataset = dataset.assign_coords(lon=(((dataset.lon + 180) % 360) - 180)).sortby(xdim)
    dataset = dataset.rio.set_spatial_dims(x_dim=xdim, y_dim=ydim) 
    dataset = dataset.rio.write_crs(crs)
    return dataset


def by_shapefile(dados: dataset, contorno:shape, tipo:str='grib') -> dataset:
    """Recorta dados de chuva com base em um shapefile de bacia.
    Esta função foi criada para trabalhar com shapefiles disponibilizados por Lis Andrade 
    em seu repositório do LAMMOC | outros arquivos podem precisar de alterações no código.
    Args:
        dados (dataset): Arquivo com a variável já selecionada. Obs.: forneça já na unidade convertida, caso necessário.
        contorno (shape): Contorno utilizado para recorte
        tipo (str, optional): aceita 'grib' ou 'nc'. Defaults to 'grib'.
    Returns:
        dataset: Dados recortados
    """
    if tipo == 'grib':
        dados_preparados = preparar_para_recorte_grib(dados)
    elif tipo == 'nc':
        dados_preparados = preparar_para_recorte_nc(dados)
    else:
        sys.exit('variável "tipo" incompatível. Selecione {grib ou nc}')

    dados_recortados = dados_preparados.rio.clip([contorno], "epsg:4326")
    
    return dados_recortados

def by_lat_lon(dados: dataset, lat_interval:Tuple, lon_interval=Tuple, tipo:str='grib') -> dataset:
    """Recorta dados de chuva com base em um intervalo de pares ordenados de latitude e longitude.
    Args:
        dados (dataset): Arquivo com a variável já selecionada. Obs.: forneça já na unidade convertida, caso necessário.
        contorno (shape): Contorno utilizado para recorte
        tipo (str, optional): aceita 'grib' ou 'nc'. Defaults to 'grib'.
    Returns:
        dataset: Dados recortados
    """
    if tipo == 'grib':
        dados_preparados = preparar_para_recorte_grib(dados)
    elif tipo == 'nc':
        dados_preparados = preparar_para_recorte_nc(dados)
    else:
        sys.exit('variável "tipo" incompatível. Selecione {grib ou nc}')

    dados_recortados = dados_preparados.sel(lon=slice(lon_interval[0], lon_interval[1]), lat=slice(lat_interval[0], lat_interval[1]))
    
    return dados_recortados
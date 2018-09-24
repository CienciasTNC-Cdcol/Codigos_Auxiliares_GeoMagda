# -*- coding: utf-8 -*-

"""codigo para validacion de indices espaciales empleando la matriz de contingencia"""


import xarray as xr
import numpy as np
from osgeo import ogr, gdal
import os
import xarray as xr
import statsmodels.api as sm
import pandas as pd
from osgeo import osr
from pylab import *





def get_file_list(Ruta_Dir, Ext):
    """Hace una lista de los objetos que reposan en una ruta de directorio
    es muy util para cargar una lista de archivos conla que despues hago algo
    Ruta_Dir - la ruta del directorio que se revisa
    Regresa la lista con ruta completa
    """
    fullfilename = os.listdir(Ruta_Dir)
    #print 'Nombre completo ', fullfilename
    lsfiles = []
    data_files = []
    for f in fullfilename:
        (shortname, ext) = os.path.splitext(f)
        if ext == Ext:
            lsfiles.append(shortname)
            data_files.append(Ruta_Dir + '\\' +shortname)
    return data_files, fullfilename


def shp_to_raster_to_array(path_vector,output_raster,cols,rows,geo_transform,project=4326):
    gdalformat = 'GTiff'

    datatype = gdal.GDT_Byte

    Shapefile =ogr.Open(path_vector)
    Shapefile_layer = Shapefile.GetLayer()
    print cols, rows

    Output = gdal.GetDriverByName(gdalformat).Create(output_raster,  cols,rows, 1, datatype,options=['ATTRIBUTE=value'])

    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(project)

    Output.SetProjection(outRasterSRS.ExportToWkt())
    Output.SetGeoTransform(geo_transform)
    Band = Output.GetRasterBand(1)
    Band.SetNoDataValue(-99999)
    gdal.RasterizeLayer(Output, [1], Shapefile_layer, options=['ATTRIBUTE=value'])
    # matriz = gdal.Open(output_raster).ReadAsArray()

    Band = None
    Output = None
    Image = None
    Shapefile = None



name = 'Tramo_1_WOFS'

shapefile = 'Active_Channel_R1.shp' #shapfile geometria observada

path_netcdf = name+'.nc' #indice espacial a evaluar en formato nedcdf

output_raster_obs = 'canal_activo_verdad.tif' #raster de salida de la geometria observada

dataset = xr.open_dataset(path_netcdf)

_coords=dataset.coords
_crs=dataset.crs
proj =_crs.crs_wkt
rows, cols= dataset.normalized_data.shape #dataset.ndwi.shape  #dataset.nnormalized_data.shape

geo_transform=(_coords["longitude"].values[0], 0.000269922107181,0.0, _coords["latitude"].values[0],0.0,0.000271265203309)


shp_to_raster_to_array(shapefile, output_raster_obs, cols, rows, geo_transform, project=4326)

matriz_observado = gdal.Open(output_raster_obs).ReadAsArray()




obs = np.ravel((np.asarray(matriz_observado))).T

coeficientes = np.arange(-0.3,0.9,0.1)#np.linspace(0,1,10) #rango de clasificacion indice espacial
lres = []
for c in coeficientes[0:]:
    print c
    agua = np.where(np.asarray(dataset['normalized_data']) > c, 1.0, 0.0) #normalized_data   -  ndwi

    sim = np.ravel((np.asarray(agua))).T

    contingency = pd.crosstab(sim, obs)



    ones_positivos = (contingency[1:][1]) #acertados

    pixeles_errados_one = float(ones_positivos.values[0]) -(contingency[0:1][1]).values[0]

    ones_totales = np.count_nonzero(obs == 1) #totales verdaderos



    pixeles_acertados_one = ones_positivos.values[0]

    pixeles_Noacertados_one = ones_totales - ones_positivos.values[0] # faltantes



    acertados = (float(ones_positivos.values[0]) / ones_totales) #* 100.0

    faltantes = 1.0- acertados

    errados = np.abs(float(pixeles_errados_one) / ones_totales) #* 100.0




    res = [c, ones_totales , pixeles_Noacertados_one,  pixeles_acertados_one, errados,acertados,faltantes]
    lres.append(res)

result = pd.DataFrame(lres,columns=['coeficiente', 'Pixeles 1 Totales', 'Pixeles no acertados', 'Pixeles acertados', '% Errados','% acertados','% faltantes'])


result.to_excel(name+'rendimiento.xlsx', sheet_name=str(name), merge_cells=False)

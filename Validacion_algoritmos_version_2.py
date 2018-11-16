__autor__ = "TNC - Ideam - Miguel Angel Canon Ramos"
__credits__ = ["none"]
__license__ = "Uso Libre"
__version__ = "1.0"
__correo__ = "cienciastnc@gmail.com, miguelca27@gmail.com"
__status__ = "Finalizado"


"""
Algoritmo para la validacion de resultados, reclasifica cada algoritmo entre 1 (agua), 0 (no agua) para la evaluacion del desempeno 
se emplea la metrica tabla de contigencia la cual evalua pixeles errados, pixeles faltantes y pixceles acertados entre observados 
y simulados
"""


# -*- coding: utf-8 -*-

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


def export_dataset_to_Raster_cubo(dataset, filename):
    def array2raster(newRasterfn, array, pixelWidth, pixelHeight, cols, rows, xorg, yorg):
        originX = xorg
        originY = yorg
        driver = gdal.GetDriverByName('GTiff')
        outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float64)

        outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
        outband = outRaster.GetRasterBand(1)
        outband.WriteArray(array)#[::-1]
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromEPSG(4326)
        outRaster.SetProjection(outRasterSRS.ExportToWkt())
        outband.FlushCache()

    def pixel(xmax, xmin, ymax, ymin, nx, ny):
        pixelWidth = abs(xmin - xmax) / nx
        pixelHeight = abs(ymin - ymax) / ny

        return pixelWidth, pixelHeight

    array = np.asarray(dataset['agua'])
    x = np.asarray(dataset['longitude'])
    y = np.asarray(dataset['latitude'])
    nx = array.shape[1]
    ny = array.shape[0]

    xmax = np.max(x)
    xmin = np.min(x)

    ymax = np.max(y)
    ymin = np.min(y)


    pixelW,pixelH=pixel(xmax,xmin,ymax,ymin,nx,ny)


    array2raster(newRasterfn=filename, array=array ,pixelWidth=pixelW, pixelHeight=pixelH,cols=nx,rows=ny,xorg=xmin,yorg=ymin)
def shp_to_raster_to_array(path_vector,output_raster,cols,rows,geo_transform,project=4326):
    gdalformat = 'GTiff'

    datatype = gdal.GDT_Byte

    Shapefile =ogr.Open(path_vector)
    Shapefile_layer = Shapefile.GetLayer()
    print cols, rows

    Output = gdal.GetDriverByName(gdalformat).Create(output_raster, cols, rows, 1, datatype,
                                                     options=['ATTRIBUTE=value'])

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





name_shp ='' # nombre capa shp observado o verdad



path_netcdf =r'' # directorio de resultados de algoritmo, archivos netcdf



shapefile = r'xxxx'+name_shp +'.shp' #directorio capa shp de la verdad

output_raster_shp = r'xxx'+name_shp+'.tif'#directorio de salida capa shp de verdad en gtiff


output_raster_obs = r'xxx/' #directorio de raster valores de 1 y 0

output_tablas = r'xxx' # directorio de tablas de salida, evaluacion del rendimiento

dataset_red =xr.open_dataset(r"Tramo_1_Aqua.nc") #algoritmo de referencia para caracteristicas de salida, gis

files, files_fullname = get_file_list(path_netcdf, '.nc')

for f in files[0:]:
    name = f[61:-4] # calibracion f[61:-4]f[60:-4]

    print name

    dataset = xr.open_dataset(f + ".nc")
    _coords=dataset.coords
    _crs=dataset_red.crs
    proj =_crs.crs_wkt


    rows, cols= dataset.variable.shape

    geo_transform=(_coords["longitude"].values[0], 0.000269922107181,0.0, _coords["latitude"].values[rows-1],0.0,0.000269922107181)


    shp_to_raster_to_array(shapefile, output_raster_shp, cols, rows, geo_transform, project=4326)

    matriz_observado = gdal.Open(output_raster_shp).ReadAsArray()




    obs = np.ravel((np.asarray(matriz_observado))).T

    coeficientes = np.arange(-0.8,0.6,0.1)
    lres = []
    for c in coeficientes[0:]:
        print c
        agua = np.where(np.asarray(dataset['variable'])[::-1]> c, 1.0, 0.0) #normalized_data   -  ndwi - mndwi

        sim = np.ravel((np.asarray(agua))).T

        salida =np.reshape(sim ,(agua.shape[0],agua.shape[1]))



        contingency = pd.crosstab(sim, obs)

        ones_totales = np.count_nonzero(obs == 1)
        if contingency.shape==(2,2):

            ones_positivos = (contingency[1:][1]) #acertados

            Miss =(contingency[0:1][1]).values[0]  # eventos identificados en observado diferente en simulado - faltantes
            Hit = (contingency[1:][1]).values[0]  # simulado y observado detectan el mismo evento
            false = (contingency[1:][0]).values[0] # falsa alarma,  eventos en el simulado diferentes al observado (errados)


            errados = float(false) / ones_totales

            acertados = (float(Hit) / ones_totales) #* 100.0

            faltantes = 1.0 - acertados

            POD = float(Hit) / (Hit + Miss)
            FAR = float(false) / (Hit + false)
            CSI = float(Hit) / (Hit + Miss + false)


        else:


            Miss =ones_totales
            Hit =-99999
            false =-99999



            ones_positivos = 0.0 # acertados

            acertados = 0.0

            faltantes = 1.0

            errados = 1.0

            POD =-99999
            FAR =-99999
            CSI =-99999



        res = [c, ones_totales ,  Hit, Miss, false, acertados,  faltantes,errados, POD, FAR, CSI ]
        lres.append(res)

        ncoords = []
        xdims = []
        xcords = {}

        coordenadas = ('latitude','longitude')
        for x in coordenadas:#dataset.coords:
            if (x != 'time'):
                ncoords.append((x, dataset.coords[x]))
                xdims.append(x)
                xcords[x] = dataset.coords[x]
        variables = {"agua": xr.DataArray(agua.astype(np.int8), dims=xdims, coords=ncoords)}
        output = xr.Dataset(variables, attrs={'crs': dataset_red.crs})
        for x in output.coords:
            output.coords[x].attrs["units"] = dataset.coords[x].units

        export_dataset_to_Raster_cubo(output, output_raster_obs +str(c)+ name + '.tif')

    result = pd.DataFrame(lres,columns=['coeficiente', 'Pixeles 1 Totales', 'Pixeles Exito', 'Pixeles Faltantes', 'Pixeles Erroneos', '% acertados', '% faltantes','% errados',
                                        'The Probability of Detection (POD)','The False Alarm Ratio (FAR)','The Critical Success Index (CSI)'])


    result.to_excel(output_tablas+name+'_rendimiento.xlsx', sheet_name=str(name), merge_cells=False)

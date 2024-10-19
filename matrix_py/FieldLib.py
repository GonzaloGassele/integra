#
# FieldLib -- Módulo para leer y procesar imágenes satelitales y rendimientos de cultivos
#

import datetime
import json
import math

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import rasterio
import rioxarray
import scipy
import scipy.ndimage as ndimage
import xarray
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import ListedColormap, Normalize
from pyproj import CRS
from rasterio.features import rasterize
from scipy.ndimage import generic_filter, label
from shapely import geometry
#from pystac_client import Client
from shapely.geometry import Point, Polygon
from skimage import measure
from skimage.measure import find_contours


class FieldLib:
    __version__ = 'Beta 3.0 -- Ago 12 2024'
    def __init__(self, field):
        self.field = field
        self.data = []
        self.clusters = []
        self.nitems = 0
        self.items = None
        self.imgd = None
        self.mat = None
        self.poly = None
        
        if type(field) == str:
            try:
                self.fld = gpd.read_file(self.field)
            except Exception as e:
                self.fld = None
        else:
            self.fld = field

        self.bounds = self.fld.geometry.bounds.values[0]
    
        if self.fld.crs.to_json_dict()['id']['code'] != 32720:
            self.fld_crs = self.fld.to_crs(32720)
        else:
            self.fld_crs = self.fld

    def drop_outliers(self, filt):
        if len(filt) != 3:
            raise Exception('formato incorrecto: {:s}'.format(filt))
        col = filt[0]
        if col not in self.fld.columns:
            raise Exception('no es una columna válida: {:s}'.format(col))
        vmin = filt[1]
        vmax = filt[2]
        self.fld_crs.drop(self.fld_crs[(self.fld_crs[col] <= vmin) | (self.fld_crs[col] >= vmax)].index, inplace=True)


    def _flatten_comprehension(self, matrix):
        return [item for row in matrix for item in row]

    # def search_img(self, time_of_interest, cloud_cover=30):
    #     client = Client.open("https://earth-search.aws.element84.com/v1")
    #     #client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    #     collection = "sentinel-2-l2a"  # Sentinel-2, Level 2A, Cloud Optimized GeoTiffs (COGs)

    #     search = client.search(
    #         max_items=300,
    #         collections=[collection],
    #         bbox=self.bounds,
    #         datetime=time_of_interest,
    #         query=["eo:cloud_cover<" + str(cloud_cover)],
    #         sortby="+properties.datetime"
    #     )
        
    #     self.nitems = search.matched()
    #     self.items = search.item_collection()
        
    #     return self.nitems, self.items

    def get_ndvi(self, items=None, verbose=False):

        # Obtengo la geometría del lote, para optimizar la descarga de las imágenes
        crop_geometry = [json.loads(self.fld.to_json())['features'][0]['geometry']]

        if items == None:
            items = range(self.nitems)

        for i in items:
            if verbose:
                print(i, self.items[i].properties['datetime'])
            red_uri = self.items[i].assets["red"].href
            nir_uri = self.items[i].assets["nir"].href

            red = rioxarray.open_rasterio(red_uri, masked=True, from_disk=True, lock=False, chunks=(1, -1, "auto"))
            nir = rioxarray.open_rasterio(nir_uri, masked=True, from_disk=True, lock=False, chunks=(1, -1, "auto"))

            red_clip = red.rio.clip(geometries=crop_geometry, crs=4326, from_disk=True)
            nir_clip = nir.rio.clip(geometries=crop_geometry, crs=4326, from_disk=True)

            # Hago red compatible con nir
            red_clip_matched = red_clip.rio.reproject_match(nir_clip)
            ndvi = (nir_clip - red_clip_matched)/ (nir_clip + red_clip_matched)


            self.data += [(self.items[i].properties, ndvi)]

        return self.data
    
    def get_gndvi(self, nitems=None, verbose=False):

        # Obtengo la geometría del lote, para optimizar la descarga de las imágenes
        crop_geometry = [json.loads(self.fld.to_json())['features'][0]['geometry']]

        if nitems == None:
            nitems = self.nitems

        for i in range(nitems):
            if verbose:
                print(i+1, self.items[i].properties['datetime'])
            green_uri = self.items[i].assets["green"].href
            nir_uri = self.items[i].assets["nir"].href

            green = rioxarray.open_rasterio(green_uri, masked=True, from_disk=True, lock=False, chunks=(1, -1, "auto"))
            nir = rioxarray.open_rasterio(nir_uri, masked=True, from_disk=True, lock=False, chunks=(1, -1, "auto"))

            green_clip = green.rio.clip(geometries=crop_geometry, crs=4326, from_disk=True)
            nir_clip = nir.rio.clip(geometries=crop_geometry, crs=4326, from_disk=True)

            # Hago red compatible con nir
            green_clip_matched = green_clip.rio.reproject_match(nir_clip)
            gndvi = (nir_clip - green_clip_matched)/ (nir_clip + green_clip_matched)


            self.data += [(self.items[i].properties, gndvi)]

        return self.data
    
    # Aplica un kernel k a la matriz m. Retorna la matriz resultado

    def conv2d(self,  size=3):
        img = self.imgd

        def nanmean_filter(values):
            valid_values = values[~np.isnan(values)]
            if len(valid_values) == 0:
                return np.nan
            return np.nanmean(valid_values)

        mask = ~np.isnan(img)

        # Aplicar el filtro genérico con una ventana de 3x3
        suavizado = generic_filter(img.data, nanmean_filter, size=size, mode='reflect')

        suavizado_final = np.where(mask, suavizado, np.nan)

        # Crear un nuevo DataArray con los datos suavizados y las mismas coordenadas
        self.imgd = xarray.DataArray(suavizado_final, coords=img.coords, dims=img.dims)
        return self.imgd


    def img2int(self, n_levels=5, bins=None, log=False):
        img = self.imgd.data[0]
        if bins != None:
            imgint = np.digitize(img, bins) - 1
            self.mat = np.where(np.isnan(img), np.nan, imgint)
            return self.mat

        v = [item for row in img for item in row if not np.isnan(item)]
        if log:
            v = [np.log(x) if x > 0 else 0 for x in v]
        bins = pd.cut(v,n_levels, right=False, labels=False, retbins=True)[1]
        if log:
            imgint = np.digitize(np.log(img, where=(img > 0)), bins) - 1
        else:
            imgint = np.digitize(img, bins) - 1

        self.mat = np.where(np.isnan(img), np.nan, imgint)

        return self.mat

    def matrix_reduce(self, rel_min=4, area_min=5, verbose=False):
        if self.mat is None:
            raise Exception('falta ejecutar img2mat')

        # Funciones útiles

        def f11(m, v):
            x,y = m.shape
            m2 = m.copy().astype(float)
            for i in range(x):
                for j in range(y):
                    if m[i,j] == v:
                        m2[i,j] = 1
                    else:
                        m2[i,j] = np.nan
            return m2

        def max_lon(matrix):
            # Encuentra las posiciones de los valores no NaN
            rows, cols = np.where(~np.isnan(matrix))

            # Encuentra las coordenadas mínimas y máximas
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()

            # Calcula la distancia euclidiana entre los puntos extremos
            return np.sqrt((max_row - min_row)**2 + (max_col - min_col)**2)

        def mfill(m,x,y,idx,jdx,v=1):
            for i in range(idx-1,idx+2):
                for j in range(jdx-1,jdx+2):
                    if i > 0 and i < x and j > 0 and j < y:
                        m[i,j] = 1
            return m

        def fneigh(m, v=1):
            x,y = m.shape
            m2 = m.copy()
            for i in range(x):
                for j in range(y):
                    if not np.isnan(m[i,j]):
                        mfill(m2,x,y,i,j,v)
            m3 = np.nansum([m2,-m],axis=0)
            cx, cy = np.where(m3 > 0)
            return cx,cy

        def mfv(mat,mt, v):
            cx, cy = fneigh(f11(mt,v))
            unique_values, counts = np.unique(mat[cx,cy], return_counts=True)
            if len(unique_values) <= 1:
                return unique_values,counts,np.nan
            max_count_index = np.argmax(counts[:-1])

            return unique_values,counts,unique_values[max_count_index]

        def mat_change(mat,mt,v):
            cx, cy = np.where(mt==1)
            mat[cx,cy] = v

        mat = self.mat
        levels = np.unique(mat)[:-1]
        for level in levels:
            binary_matrix = (mat == level)
            lm, nl = ndimage.label(binary_matrix)
            for i in range(nl):
                unique_counts, counts = np.unique(lm, return_counts=True)
                for j in unique_counts[1:]:
                    if j == np.nan:
                        continue
                    mt = f11(lm,j)
                    sup = np.nansum(mt)
                    rel = sup/max_lon(mt) if sup > 1 else 0
                    if sup > area_min:
                        continue
                    if verbose:
                        print('Reduciendo sup',sup,'con rel',rel)
                    uv, uv2, most_freq_val = mfv(mat,lm,j)
                    if np.isnan(most_freq_val):
                        continue
                    mat_change(mat,mt,most_freq_val)

        return mat

    def mat2poly(self):
        if self.mat is None:
            raise Exception('falta ejecutar img2mat')
        matrix = self.mat
        # Convertir NaN a 0
        binary_matrix = np.nan_to_num(matrix, nan=0).astype(int)

        all_polygons = []

        unique_values = np.unique(binary_matrix)

        for value in unique_values:
            binary_matrix = (matrix == value).astype(int)

            # Etiquetar las regiones conectadas
            label_matrix, num_labels = ndimage.label(binary_matrix)

            # Encontrar los contornos
            contours = measure.find_contours(binary_matrix, level=0.5)

            # Convertir contornos a polígonos
            polygons = []
            for contour in contours:
                if len(contour) > 3:  # Un polígono debe tener al menos 3 puntos
                    polygon = Polygon(contour)
                    if polygon.is_valid:
                        polygons.append(polygon)

            all_polygons += [polygons]
        self.poly = all_polygons
        return all_polygons

    def poly2gdf(self):
        if self.poly is None:
            raise Exception('falta ejecutar mat2poly')

        data = self.imgd.data[0]
        polygons = self.poly
        # Calculo el valor (ndvi) promedio
        mean_v = []
        for i in range(len(polygons)):
            mean_v += [np.nanmean(data[np.argwhere(self.mat==i)])]

        # Crear un DataFrame vacío con las columnas especificadas
        cols = {'Ambiente':'int', 'Valor':'float', 'Poly':'int','geometry': 'geometry'}
        poly_gdf = pd.DataFrame(columns=cols.keys()).astype(cols)

        # Convertir el DataFrame vacío en un GeoDataFrame
        poly_gdf = gpd.GeoDataFrame(poly_gdf, geometry='geometry', crs=32720)

    #    poly_gdf = poly_gdf.set_crs('epsg:4326')
    #    poly_gdf_crs = poly_gdf.to_crs(32720)

        npoly = 0
        for i in range(len(polygons)):
            for j in range(len(polygons[i])):
                lst=[]
                for y,x in polygons[i][j].boundary.coords:
                    lst += [(self.imgd.x[int(x)].item(),self.imgd.y[int(y)].item())]
                poly = Polygon(lst)

                # Definir una nueva fila con datos
                new_row = {'Ambiente': i, 'Valor' : mean_v[i], 'Poly': npoly, 'geometry': poly}
                new_gdf = gpd.GeoDataFrame([new_row], geometry='geometry', crs=poly_gdf.crs)

                # Agregar la nueva fila al GeoDataFrame
                poly_gdf = pd.concat([poly_gdf, new_gdf], ignore_index=True)
                npoly += 1

        # Verifico si hay problemas con los polígonos
        if len(poly_gdf[~poly_gdf.is_valid]) > 0:
            poly_gdf['geometry'] = poly_gdf['geometry'].buffer(0)

        return poly_gdf

    # ndvi_avg -- Promedia Imagens ndvi. Si lst es None, toma todas las imagenes. Sino, la lista pasada en lst
    
    def img_avg(self, lst=None):
        if lst is None:
            lst = range(len(self.data))

        n = len(lst)
        mat = self.data[lst[0]][1].copy()
        for i in lst[1:]:
            mat += self.data[i][1].values
        
        self.imgd = mat/n
        
        return self.imgd

    # Convierte un GeoDataFrame en un xarray compatible con las imagenes de NDVI

    def GDF2xarray(self, res=10, flip=True, yield_fld='Yield'):

        # Definir la resolución y la extensión del raster resultante
        #res = 10  # resolución en metros
        xmin, ymin, xmax, ymax = self.fld_crs.total_bounds
        width = int((xmax - xmin) / res)
        height = int((ymax - ymin) / res)
        transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, width, height)

        # Coordenadas de píxeles
        y_coords, x_coords = np.mgrid[0:height, 0:width]

        # Convertir coordenadas de píxeles a coordenadas en CRS
        in_proj = pyproj.Proj('epsg:32720')  # Proyección WGS84 (latitud/longitud)
        out_proj = pyproj.Proj(self.fld_crs.crs)  # CRS del GeoDataFrame
        transformer = pyproj.Transformer.from_proj(in_proj, out_proj)
        x_crs, y_crs = transformer.transform(x_coords * res + xmin, y_coords * res + ymin)

        # Rasterizar el GeoDataFrame
        mask = rasterize(
            [(geom, value) for geom, value in zip(self.fld_crs.geometry, self.fld_crs[yield_fld])],
            out_shape=(height, width),
            transform=transform,
            fill=np.nan,
            all_touched=True,
            dtype=rasterio.float64
        )

        if flip:
            # Invertir el eje Y de la imagen rasterizada
            mask = np.flipud(mask)

        # Crear un DataArray con coordenadas en CRS
        self.crop_yield = xarray.DataArray(
            mask,
            coords=[('y', y_crs[:,0]), ('x', x_crs[0,:])],
            dims=['y', 'x'],
            attrs={'crs': self.fld_crs.crs}
        )

        # Agregar la dimensión 'band' al DataArray
        self.crop_yield = self.crop_yield.expand_dims('band')

        # Ajustar las coordenadas de la dimensión 'band'
        self.crop_yield['band'] = [1]

        self.imgd = self.crop_yield.copy(deep=True)
        return self.imgd

    def save_shp(self, name, poly0=False, verbose=False):
        clt = []
        nc = len(self.clusters)
        # Calculo los promedios de cada polígono
        if poly0:  # Se genera el polígono exterior?
            poligonos = [self.fld.unary_union]
            ndvi_promedios=np.zeros(nc+1)
            ndvi_promedios[0] = np.nanmean(self.imgd.values) # NDVI promedio de todo el lote
            for i in range(1,nc+1):
                ndvi_promedios[i] = self.clusters[i-1][3][2]
                clt += [self.clusters[i-1][0]]
        else:
            poligonos = []
            ndvi_promedios=np.zeros(nc)
            for i in range(nc):
                ndvi_promedios[i] = self.clusters[i][3][2]
                clt += [self.clusters[i][0]]

        for i in range(nc):
            poligonos += [self.clusters[i][1]]
            if verbose:
                print('Poligono',i, 'Promedio', ndvi_promedios[i])

        # Genero el geodataframe
        datos = {'geometry': poligonos, 'ndvi': ndvi_promedios, 'cluster' : clt}
        gdf = gpd.GeoDataFrame(datos)
        gdf.to_file(name + '.shp')
        
        return datos

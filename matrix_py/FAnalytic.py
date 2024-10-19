# Funciones para hacer analítica de rendimientos

# Esta primera versión necesita variables para funcionar

# Agregar los paths correspondientes

#path_presc = 
#path_lote = 
#path_cosecha = 
#path_img = 

# Variables

#Rinde = 'Rendimient'       # Columna de rendimiento en datos de cosecha
#Dosis = 'SEM'              # Columna de dosis en prescripciones
#Superficie = 'Hect'        # Has de cada ambiente
#Altitud = 'Altitud'        # Altitud

# Archivos

#presc_file = 'PreMAL4.shp'
#rinde_file = 'Agroyunta_La Maria Ana - Grupo P-F_Mapa de rinde L4 LMA.shp'
#lote_file = 'Lote 4.shp'

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, shape, MultiPoint
from FieldLib import FieldLib as fl
#from ph import ph
import pandas as pd
import rasterio
from rasterio import features
import os
import matplotlib.animation as animation
from IPython.display import HTML
from scipy.stats import ttest_ind

__version__ = 'Ago 15 - Beta 1.4'

#
# Funciones
#


# presc_process no recibe parámetros y genera el plot de prescriciones

def presc_process(file,Dosis, Amb, path_img,shrink=1, cmap='RdYlGn', figsize=(6,5)):

    presc = gpd.read_file(file)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    unique_values = np.sort(presc[Dosis].unique())
    amb_dict = dict(zip(unique_values,Amb))
    presc['Ambiente'] = presc[Dosis].map(amb_dict)

    # Usar el colormap predefinido 'RdYlGn'
    cmap = plt.get_cmap(cmap)

    presc.plot(ax=ax, column=Dosis,cmap=cmap)
    ax.grid(True)
    plt.grid(True)
    # Crear una normalización para la colorbar
    #norm = BoundaryNorm([i for i in range(len(presc))], cmap.N)
    boundaries = np.arange(0, len(unique_values)+1)
    norm = mcolors.BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)

    # Añadir la colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=shrink)
    cbar.set_ticks([i+0.5 for i in range(len(unique_values))])#[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    #class_label = [str(int(d)) for d in unique_values]
    cbar.set_ticklabels(Amb)#class_label)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.savefig(path_img + 'presc.png')

    if presc.crs is None:
        presc = presc.set_crs('4326')
    presc = presc.to_crs('32720')

    return presc

# rinde_process() lee los rendimientos y genera el plot correspondiente

def rinde_process(file, Rinde, path_img, min_rinde=None, max_rinde=None, fm=0, cmap='RdYlGn'):
    rinde = gpd.read_file(file)

    # Filtro si hace falta
    if min_rinde is not None:
        if fm == 0:
            rinde = rinde[rinde[Rinde] >= min_rinde]
        else:
            rinde[Rinde] = np.where(rinde[Rinde] < min_rinde,min_rinde,rinde[Rinde])
    if max_rinde is not None:
        if fm == 0:
            rinde = rinde[rinde[Rinde] <= max_rinde]
        else:
            rinde[Rinde] = np.where(rinde[Rinde] > max_rinde,max_rinde,rinde[Rinde])

    # Creo la clase de FieldLib

    rinde_fl = fl(rinde)

    # Transformo los datos para que sean compatibles con las imágenes satelitales

    rinde_fl.GDF2xarray(yield_fld=Rinde)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)


    rinde_fl.imgd.plot(robust=True, cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Mapa de Rendimiento')
#    plt.savefig(path_img + 'rinde.png')

    plt.subplot(1, 2, 2)

    rinde[Rinde].hist(bins=20, density=True,edgecolor='black')
    plt.title('Distribución de Rendimiento')
    plt.xlabel('Tn/Ha')
    plt.savefig(path_img + 'rinde.png')

    return rinde, rinde_fl

# rend_por_ambientes_process genera el plot que compara diferencias entre dosis y rendimientos

def rend_por_ambientes_process(rinde, presc, Dosis, Rinde, path_img, figsize=(12,5)):
    # Analizo rendimientos por ambientes
    # CRS Común
    rinde = rinde.to_crs('32720')
    presc = presc.to_crs('32720')
    
    unique_values = np.sort(presc[Dosis].unique())

    gdf_union = gpd.sjoin(rinde, presc, how="inner", predicate="within")
    
    rend_dosis = gdf_union[[Dosis,Rinde]].groupby(Dosis).mean()
    rend_dosis.reset_index(inplace=True)
    unique_values = np.sort(gdf_union[Dosis].unique())

    dd=[]
    dr=[]
    for i in range(1,len(unique_values)):
        dd += [(rend_dosis.iloc[i,0]/rend_dosis.iloc[i-1,0]-1)*100]
        dr += [(rend_dosis.iloc[i,1]/rend_dosis.iloc[i-1,1]-1)*100]

    plt.figure(figsize=figsize)
    xticks = range(1,len(dd)+1)
    xlabels = ['Dif '+str(i)+'-'+str(i+1) for i in range(1,len(dd)+1)]
    plt.plot(xticks, dd,label='Dif dosis', marker='o', color='r')
    plt.bar(xticks, dr,label='Dif rendimiento')#, marker='o')
    plt.legend()
    plt.xlabel('Diferencias entre ambientes')
    plt.xticks(xticks, labels=xlabels)
    plt.ylabel('Diferencias en %')
    plt.grid(True)
    plt.savefig(path_img + 'dosis-rend.png')
    return gdf_union, rend_dosis

# boxplot_process genera el boxplot

def boxplot_process(Dosis, Rinde, gdf_union, path_img, labels=None):
    if labels == None:
        labels = [str(d) for d in gdf_union[Dosis].unique()]
    sns.boxplot(gdf_union[[Dosis,Rinde]], y=Rinde,x=Dosis)
    plt.xticks(ticks=range(len(labels)), labels=labels)
    plt.ylabel('Rinde')
    plt.xlabel('Ambiente')
    plt.savefig(path_img + 'boxplot.png')


# dist_por_ambientes_process es similar al boxplot pero con distribuciones

def dist_por_ambientes_process(gdf_union, rend_dosis, Dosis, Rinde, path_img, labels=None, figsize=(5,4)):
    # Creo plot con distribuciones de rendimientos por ambientes
    if labels == None:
        labels = [str(d) for d in gdf_union[Dosis].unique()]
    d_dict = {}
    n=0
    dosis_unique = np.sort(gdf_union[Dosis].unique())

    for dosis in dosis_unique:
        d_dict[str(n)] = gdf_union[gdf_union[Dosis] == dosis][Rinde]
        n += 1
    data = pd.DataFrame(d_dict)
    min_bin = gdf_union[Rinde].min().round(0)
    max_bin = gdf_union[Rinde].max().round(0)

    # Crea subgráficos apilados horizontalmente
    na = data.shape[1] # Nro de ambientes
    fig, axs = plt.subplots(1, na, figsize=figsize)
    tit = rend_dosis[Dosis].unique().astype(int)
    #bins = np.arange(min_bin,max_bin,1)
    bins = [min_bin+i*(max_bin-min_bin)/20 for i in range(20)]
    #
    # Llena cada subgráfico con datos y agrega títulos
    for i, ax in enumerate(axs):
        ax.hist(data[str(i)],bins=bins,orientation='horizontal',edgecolor='black', linewidth=1.2)
        ax.invert_xaxis()
        ax.set_title(labels[i])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticklabels('')
        ax.set_xticks([])

        if i < na - 1:
            ax.set_yticks([])
        else:
            ax.yaxis.set_ticks_position('right')
            ax.yaxis.set_label_position('right')
            ax.set_ylabel('Tn/Ha')

        ax.grid(False)
        if i == n//2:
            ax.set_xlabel('Ambiente')

    for ax in axs.flat:
        ax.label_outer()

    plt.savefig(path_img + 'dist-rend.png')

# Las siguientes funciones se usan para leer NDVI y generar el plot y el video

def ndvi_search(lote,rango_fecha,nubes=10):
    lote_fl = fl(lote)
    nitems, items = lote_fl.search_img(rango_fecha,nubes)
    
    # Muestro las imágenes disponibles que cumplen con los requisitos

    k=0
    item_list = []
    print('{:32s} {:s}\t{:s}'.format('Fecha y hora','% de Nubes','% de Vegetación'))
    for i in lote_fl.items:
        k += 1
    #    print('VA ---> ',pd.to_datetime(i.properties['datetime']))
    #    if pd.to_datetime(i.properties['datetime']).month not in [12,1,2,3]:
    #        continue
        print('{:d} - {:32s} {:5.2f}\t\t{:5.2f}'.format(k,i.properties['datetime'], \
                        i.properties['eo:cloud_cover'],i.properties['s2:vegetation_percentage']))
        item_list += [k-1]
    return lote_fl, item_list

def ndvi_read(lote_fl, item_list=None):
    ndvi = lote_fl.get_ndvi(items=item_list, verbose=True)
    return ndvi

def make_video(ndvi, path_img):
    
    # Función para generar las imagenes del video
    def update(frame):
        img.set_array(ndvi_array[frame])
        date_text.set_text(dates[frame])
        return img,date_text
    
    # Configurar la figura y el eje
    fig, ax = plt.subplots()
    ndvi_array = [ndvi[i][1][0] for i in range(len(ndvi))]
    dates = [ndvi[i][0]['datetime'][:10] for i in range(len(ndvi))]

    # Inicializar la imagen en el eje
    img = ax.imshow(ndvi_array[0], cmap='RdYlGn', animated=True)

    # Añadir texto para las fechas
    date_text = ax.text(0.70, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')

    # Crear la animación
    ani = animation.FuncAnimation(fig, update, frames=len(ndvi_array), blit=True, repeat=True)

    HTML(ani.to_jshtml())
    
    # Guardar la animación como un archivo mp4
    ani.save(path_img + 'ndvi.mp4', writer='ffmpeg')

def plot_ndvi(ndvi, path_img):
    plt.figure(figsize=(7, 5))
    ndvi_v = []
    fecha_v = []

    for i in range(len(ndvi)):
        ndvi_v += [ndvi[i][1].mean().values.item()]
        fecha_v += [ndvi[i][0]['datetime'][:10]]
    plt.plot(fecha_v,ndvi_v, marker='o',color='green')
    #plt.style.use({
    #    'figure.facecolor': 'lightblue',   # Color de fondo de la figura
    #    'axes.facecolor': 'lightyellow'    # Color de fondo de los ejes
    #})
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.title('NDVI')
    plt.tight_layout()
    plt.savefig(path_img + 'ndvi-plot.png')

# Las siguientes funciones generan una tabla (tab1.csv) con datos de rendimientos por ambiente

def point_in_pol(mat,poly, rinde_fl):
    # Extraer las coordenadas
    lon = mat.coords['x'].values
    lat = mat.coords['y'].values

    # Crear una lista de puntos
    points = [Point(lon[i], lat[j]) for i in range(len(lon)) for j in range(len(lat))]

    # Crear un GeoDataFrame con estos puntos
    points_gdf = gpd.GeoDataFrame(geometry=points, crs=mat.crs)

    joined = gpd.sjoin(points_gdf, poly, how='inner', predicate='within')

    # Extraer los índices de las celdas que están dentro de los polígonos
    indices = joined.index.values

    # Convertir los índices a coordenadas de la matriz original
    coords_dentro = [(index // len(lat), index % len(lat)) for index in indices]

    s=0
    for i in coords_dentro:
        if not np.isnan(rinde_fl.imgd.data[0][i[1],i[0]]):
            s += 1
    return s

def map_vs_presc(gdf_union, rinde_fl, presc, Rinde, Dosis, Superficie, path_img):
    h_dict = dict([(d,0) for d in np.sort(presc[Dosis].unique())])
    for i in range(len(presc)):
        h_dict[presc.iloc[i][Dosis]] += point_in_pol(rinde_fl.imgd, presc.iloc[i:i+1], rinde_fl)

    #df = gdf_union[[Dosis,Rinde, Superficie]].groupby(Dosis).agg({Rinde:'mean', Superficie:'mean'}).reset_index()
    df = gdf_union[[Dosis,Rinde]].groupby(Dosis).agg({Rinde:'mean' }).reset_index()

#    df['Superficie'] = gdf_union[[Dosis,Superficie]].groupby(Dosis).mean().values
    df['Superficie'] = presc[[Dosis,Superficie]].groupby(Dosis).sum().values
    df['Sup. Mapeada'] = df[Dosis].apply(lambda x: h_dict[x]/100)

    df['Rinde Total'] = df['Sup. Mapeada'] * df[Rinde]

    df.columns = ['Dosis', 'Rendimiento', 'Superficie', 'Sup. Mapeada','Rinde Total']

    print(df)

    df.to_csv(path_img + 'tab1.csv', index=False)

    return df

# plot_rxa - Plots de rendimientos ambiente por ambiente

def plot_rxa(presc, rinde_fl, Dosis, Amb, path_img, shrink=1, figsize=(9,12)):
    dosis_v = np.sort(presc[Dosis].unique())
    nrows = len(dosis_v)//2 if len(dosis_v)%2 == 0 else len(dosis_v)//2 + 1
    figure, axis = plt.subplots(nrows,2,figsize=figsize)

    for i in range(len(dosis_v)):
        fila = i // 2  # Calcular el número de fila (0, 1 o 2)
        columna = i % 2  # Calcular el número de columna (0 o 1)

        rinde_fl.imgd.plot(ax=axis[fila,columna], robust=True, cmap='RdYlGn', cbar_kwargs={'shrink': shrink})

        presc[presc[Dosis]==dosis_v[i]].boundary.plot(ax=axis[fila,columna], edgecolor='black', alpha=0.5, hatch='+')

        axis[fila,columna].set_title(Amb[i])#'Dosis '+str(int(dosis_v[i])))
        axis[fila,columna].set_xticks([])
        axis[fila,columna].set_xlabel('')
        axis[fila,columna].set_yticks([])
        axis[fila,columna].set_ylabel('')

        # Si tengo un cuadro de más, borro las líneas

    if len(dosis_v)%2 != 0:
        fila = nrows-1
        columna = 1
        axis[fila,columna].set_xticks([])
        axis[fila,columna].set_xlabel('')
        axis[fila,columna].set_yticks([])
        axis[fila,columna].set_ylabel('')
        axis[fila,columna].spines['top'].set_visible(False)
        axis[fila,columna].spines['right'].set_visible(False)
        axis[fila,columna].spines['bottom'].set_visible(False)
        axis[fila,columna].spines['left'].set_visible(False)

    plt.savefig(path_img + 'rinde-ambientes.png')


# Variabilidad y topografía

def mat_std(m, dx=5, dy=5):
    y, x = m.shape
    mat = np.empty((x, y))
    mat[:] = np.nan
    for i in range(dx, x - dx):
        for j in range(dy, y - dy):
            window = m[i - dx:i + dx, j - dy:j + dy]
            if np.count_nonzero(~np.isnan(window)) > 1:
                mat[i, j] = np.nanstd(window)
    return np.rot90(mat)

def plot_var_topografia(rinde, rinde_fl, Altitud, path_img, shrink=1, figsize=(12,6), dx=5, dy=5):

    if Altitud not in rinde.columns:
        print('{:s} no pertenece a los datos de rendimiento'.format(Altitud))
        return None

    mat_var = mat_std(rinde_fl.imgd.data[0],dx=dx, dy=dy)

    # Calculo la topografía
    altitud_fl = fl(rinde)
    altitud_fl.GDF2xarray(yield_fld=Altitud)

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    img1 = ax[0].imshow(np.rot90(mat_var,k=2))#, cmap='GnYlRd')
    fig.colorbar(img1, ax=ax[0], orientation='vertical', shrink=shrink)
    ax[0].set_title('Variabilidad de Rendimiento')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_xlabel('')
    ax[0].set_ylabel('')

    img2 = altitud_fl.imgd.plot(ax=ax[1], add_colorbar=False, robust=True)#, cmap='RdYlGn')
    fig.colorbar(img2, ax=ax[1], orientation='vertical')
    ax[1].set_title('Mapa de Topografía')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_xlabel('')
    ax[1].set_ylabel('')

    plt.savefig(path_img + 'var-altura.png')

    return altitud_fl

def plot_topografia(rinde, rinde_fl, Altitud, path_img, shrink=1, figsize=(12,6),cmap='viridis'):

    if Altitud not in rinde.columns:
        print('{:s} no pertenece a los datos de rendimiento'.format(Altitud))
        return None

    # Calculo la topografía
    altitud_fl = fl(rinde)
    altitud_fl.GDF2xarray(yield_fld=Altitud)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    img = altitud_fl.imgd.plot(ax=ax, add_colorbar=False, robust=True, cmap=cmap)
    fig.colorbar(img, ax=ax, orientation='vertical')
    ax.set_title('Mapa de Topografía')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    plt.savefig(path_img + 'topografia.png')

    return altitud_fl

# Diferencias significativas en los rendimientos por ambientes

def dif_rinde(presc,gdf_union, Dosis,Rinde):
    dosis_v = np.sort(presc[Dosis].unique())
    dosis_v = dosis_v.astype(int)
    rinde_dif = {}
    for i in range(1,len(dosis_v)):
        r1 = gdf_union[gdf_union[Dosis] == dosis_v[i-1]][Rinde]
        r2 = gdf_union[gdf_union[Dosis] == dosis_v[i]][Rinde]
        # Realizar la prueba t de Student para muestras independientes
        resultado = ttest_ind(r2, r1)
        t = resultado.statistic.round(4)
        pvalue = resultado.pvalue.round(4)
        rinde_dif[str(dosis_v[i-1])+'-'+str(dosis_v[i])] = {'t':t, 'pvalue':pvalue}
        if abs(t) < 2:
            print('Cuidado! podría existir una diferencia no significativa')
    return pd.DataFrame(rinde_dif)

def extremos(gdf_union,presc,Dosis, Rinde,path_img, qt=0.25, cmap='RdYlGn', figsize=(10,8)):
    fig, ax = plt.subplots(figsize=figsize)
    presc.plot(ax=ax, column=Dosis, edgecolor='black', alpha=0.1, cmap=cmap)
    for amb in gdf_union['Ambiente'].unique():
        if qt < 0.5:
            filt = (gdf_union[Rinde]<gdf_union[gdf_union['Ambiente']==amb][Rinde].quantile(qt))
        else:
            filt = (gdf_union[Rinde]>gdf_union[gdf_union['Ambiente']==amb][Rinde].quantile(qt))
        gdf_union[(gdf_union['Ambiente']==amb) & filt]['geometry'].plot(ax=ax, markersize=3)
        ax.set_xticklabels('')
        ax.set_yticklabels('')
        ax.set_xticks([])
        ax.set_yticks([])
        if qt < 0.5:
            plt.title('{:4.2f}% de valores mínimos de rendimiento en cada ambiente'.format(qt*100))
        else:
            plt.title('{:4.2f}% de valores maximos de rendimiento en cada ambiente'.format((1-qt)*100))
    if qt < 0.5:
        plt.savefig(path_img + 'xtr_inf.png')
    else:
        plt.savefig(path_img + 'xtr_sup.png')

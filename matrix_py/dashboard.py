import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import colors as mcolors
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import BoundaryNorm, Normalize

from FieldLib import FieldLib as fl

st.set_page_config(layout="centered")

def dashboard():
    st.markdown('<h2 style="font-size: 24px;">Dashboard</h2>', unsafe_allow_html=True)
    analysis_choice = st.selectbox("Seleccionar análisis", ["Análisis 1", "Análisis 2", "Análisis 3"], placeholder="Seleccione un análisis para visualizar")

    # Data
    data = {
        "Ambiente": ["Malo", "Regular", "Bueno", "Menos Bueno", "Mas Bueno", "Excelente", "Sup."],
        "Rend Medio": [2.89, 1.81, 1.50, 4.34, 3.44, 8.90, 8.22],
        "Superficie Mapeada": [28.31, 4.09, 4.01, 4.88, 19.47, 31.09, 66.67],
        "Rinde Total": [127.94, 18.29, 29.67, 64.91, 122.59, 74.85, 119.09]
    }

    # Convert data to DataFrame
    df = pd.DataFrame(data)

    # Set the index to 'Ambiente' to make the table more readable
    df.set_index('Ambiente', inplace=True)

    # Render the table
    st.write("### Tabla de valores promedios por ambiente y porcentajes de cambio")
    st.table(df)

    if analysis_choice == "Análisis 1":
        col1, col2, col3 = st.columns(3)

# fa.plot_topografia(rinde, rinde_fl, Altitud, path_img, figsize=(6,4), shrink=0.93)

def plot_topografia():
    rinde = gpd.read_file("shapes/cosecha_plot/cosecha_1.shp")
    rinde_fl = fl(rinde)
    figsize = (10, 7)
    shrink=0.93
    Altitud = 'Altitud'
    cmap = 'RdYlGn'

    if Altitud not in rinde.columns:
        print('{:s} no pertenece a los datos de rendimiento'.format(Altitud))
        return None

    # Calculo la topografía
    altitud_fl = fl(rinde)
    altitud_fl.GDF2xarray(yield_fld=Altitud)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)

  # Set figure background color
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    img = altitud_fl.imgd.plot(ax=ax, add_colorbar=False, robust=True, cmap=cmap)
    
    # Add colorbar with white ticks
    cbar = fig.colorbar(img, ax=ax, orientation='vertical', shrink=shrink)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    plt.grid(True, color='white', linestyle='dashed', linewidth=0.7)
    
    # img.boundary.plot(ax=ax, color='white', linewidth=3, alpha=0.2)
    # img.boundary.plot(ax=ax, marker='o', markersize=5, color='yellow', linestyle='None')

    # Set title and remove axis labels/ticks
    ax.set_title('Mapa de Topografía', color='white', fontsize=16, weight='bold', pad=20)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    plt.tight_layout()
    plt.savefig('imagenes/topografia.png', format='png', dpi=300, bbox_inches='tight', facecolor='#0e1117')

    st.pyplot(fig)

# fa.plot_rxa(presc, rinde_fl, Dosis, Amb, path_img, figsize=(15,17))

# def plot_rxa():
#     rinde = gpd.read_file("shapes/cosecha_plot/cosecha_1.shp")
#     rinde_fl = fl(rinde)
#     Dosis = 'Dosis Pres'
#     Amb = ['Malo', 'Regular', 'Bueno Menos', 'Bueno Mas', 'Excelente']
#     figsize = (15, 17)
#     presc = gpd.read_file("shapes/ambientacion_plot/ambientacion_1.shp")

#     dosis_v = np.sort(presc[Dosis].unique())
#     nrows = len(dosis_v)//2 if len(dosis_v)%2 == 0 else len(dosis_v)//2 + 1
#     figure, axis = plt.subplots(nrows,2,figsize=figsize)

#     for i in range(len(dosis_v)):
#         fila = i // 2  # Calcular el número de fila (0, 1 o 2)
#         columna = i % 2  # Calcular el número de columna (0 o 1)

#         rinde_fl.imgd.plot(ax=axis[fila,columna], robust=True, cmap='RdYlGn', cbar_kwargs={'shrink': 1})

#         presc[presc[Dosis]==dosis_v[i]].boundary.plot(ax=axis[fila,columna], edgecolor='black', alpha=0.5, hatch='+')

#         axis[fila,columna].set_title(Amb[i])#'Dosis '+str(int(dosis_v[i])))
#         axis[fila,columna].set_xticks([])
#         axis[fila,columna].set_xlabel('')
#         axis[fila,columna].set_yticks([])
#         axis[fila,columna].set_ylabel('')

#         # Si tengo un cuadro de más, borro las líneas

#     if len(dosis_v)%2 != 0:
#         fila = nrows-1
#         columna = 1
#         axis[fila,columna].set_xticks([])
#         axis[fila,columna].set_xlabel('')
#         axis[fila,columna].set_yticks([])
#         axis[fila,columna].set_ylabel('')
#         axis[fila,columna].spines['top'].set_visible(False)
#         axis[fila,columna].spines['right'].set_visible(False)
#         axis[fila,columna].spines['bottom'].set_visible(False)
#         axis[fila,columna].spines['left'].set_visible(False)

#     plt.savefig('imagenes/rinde-ambientes.png')

################### No se ve las lineas de cuadritos sobre el gráfico, no sé si me faltó algo

def plot_rxa():
    rinde = gpd.read_file("shapes/cosecha_plot/cosecha_1.shp")
    rinde_fl = fl(rinde)
    Dosis = 'Dosis Pres'
    Amb = ['Malo', 'Regular', 'Bueno Menos', 'Bueno Mas', 'Excelente']
    figsize = (15, 17)
    presc = gpd.read_file("shapes/ambientacion_plot/ambientacion_1.shp")

    # Ensure rinde_fl.imgd is correctly initialized
    rinde_fl.GDF2xarray(yield_fld='Rendimient')

    if rinde_fl.imgd is None:
        raise ValueError("The imgd attribute is None. Ensure that GDF2xarray() is processing the data correctly.")

    dosis_v = np.sort(presc[Dosis].unique())
    nrows = len(dosis_v)//2 if len(dosis_v)%2 == 0 else len(dosis_v)//2 + 1
    figure, axis = plt.subplots(nrows, 2, figsize=figsize)
    
    # Set figure background color
    figure.patch.set_facecolor('#0e1117')

    for i in range(len(dosis_v)):
        fila = i // 2# Calculate row number
        columna = i % 2# Calculate column number# Set axis background color
        axis[fila, columna].set_facecolor('#0e1117')

        # Plotting the data if imgd is available
        if rinde_fl.imgd is not None:
          img_plot = rinde_fl.imgd.plot(ax=axis[fila, columna], robust=True, cmap='RdYlGn', cbar_kwargs={'shrink': 1})
          cbar = img_plot.colorbar
          cbar.ax.yaxis.set_tick_params(color='white')
          plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        else:
          print(f"No image data available for plot {i}.")

        # Plotting prescription boundaries with yellow edge color
        presc[presc[Dosis] == dosis_v[i]].boundary.plot(
            ax=axis[fila, columna], edgecolor='yellow', alpha=0.5, hatch='+'
        )

        # Setting title with white text
        axis[fila, columna].set_title(Amb[i], color='white', fontsize=16, weight='bold', pad=20)
        axis[fila, columna].set_xticks([])
        axis[fila, columna].set_xlabel('')
        axis[fila, columna].set_yticks([])
        axis[fila, columna].set_ylabel('')

    # Handling the extra subplot (if there is one)
    if len(dosis_v) % 2 != 0:
        fila = nrows - 1
        columna = 1
        axis[fila, columna].set_xticks([])
        axis[fila, columna].set_xlabel('')
        axis[fila, columna].set_yticks([])
        axis[fila, columna].set_ylabel('')
        axis[fila, columna].spines['top'].set_visible(False)
        axis[fila, columna].spines['right'].set_visible(False)
        axis[fila, columna].spines['bottom'].set_visible(False)
        axis[fila, columna].spines['left'].set_visible(False)

        axis[fila, columna].set_facecolor('#0e1117')


    plt.savefig('imagenes/rinde-ambientes.png', format='png', dpi=300, bbox_inches='tight')
    st.pyplot(figure)
    
# def plot_rxa():
#     rinde = gpd.read_file("shapes/cosecha_plot/cosecha_1.shp")
#     rinde_fl = fl(rinde)
#     Dosis = 'Dosis Pres'
#     Amb = ['Malo', 'Regular', 'Bueno Menos', 'Bueno Mas', 'Excelente']
#     figsize = (15, 17)
#     presc = gpd.read_file("shapes/ambientacion_plot/ambientacion_1.shp")

#     dosis_v = np.sort(presc[Dosis].unique())
#     nrows = len(dosis_v)//2 if len(dosis_v)%2 == 0 else len(dosis_v)//2 + 1
#     figure, axis = plt.subplots(nrows, 2, figsize=figsize)
    
#     # Set figure background color
#     figure.patch.set_facecolor('#0e1117')

#     for i in range(len(dosis_v)):
#         fila = i // 2# Calculate row number
#         columna = i % 2# Calculate column number# Set axis background color
#         axis[fila, columna].set_facecolor('#0e1117')

#         # Plotting the data
#         rinde_fl.imgd.plot(ax=axis[fila, columna], robust=True, cmap='RdYlGn', cbar_kwargs={'shrink': 1})

#         # Plotting prescription boundaries with yellow edge color
#         presc[presc[Dosis] == dosis_v[i]].boundary.plot(
#             ax=axis[fila, columna], edgecolor='yellow', alpha=0.5, hatch='+'
#         )

#         # Setting title with white text
#         axis[fila, columna].set_title(Amb[i], color='white', fontsize=16, weight='bold', pad=20)
#         axis[fila, columna].set_xticks([])
#         axis[fila, columna].set_xlabel('')
#         axis[fila, columna].set_yticks([])
#         axis[fila, columna].set_ylabel('')

#     # Handling the extra subplot (if there is one)iflen(dosis_v) % 2 != 0:
#         fila = nrows - 1
#         columna = 1
#         axis[fila, columna].set_xticks([])
#         axis[fila, columna].set_xlabel('')
#         axis[fila, columna].set_yticks([])
#         axis[fila, columna].set_ylabel('')
#         axis[fila, columna].spines['top'].set_visible(False)
#         axis[fila, columna].spines['right'].set_visible(False)
#         axis[fila, columna].spines['bottom'].set_visible(False)
#         axis[fila, columna].spines['left'].set_visible(False)

#     plt.tight_layout()
#     plt.savefig('imagenes/rinde-ambientes.png', format='png', dpi=300, bbox_inches='tight')
#     st.pyplot(figure)

# rinde, rinde_fl = fa.rinde_process(path_cosecha + rinde_file, Rinde, path_img, max_rinde=7, min_rinde=1.5,cmap='viridis_r')#, min_rinde=1, max_rinde=7)

# def rinde_process():
    
#     max_rinde = 7
#     min_rinde = 1.5
#     cmap='RdYlGn'
#     fm=0

#     Rinde = 'Rendimient'
#     rinde = gpd.read_file("shapes/cosecha_plot/cosecha_1.shp")    

#     # Filtro si hace falta
#     if min_rinde is not None:
#         if fm == 0:
#             rinde = rinde[rinde[Rinde] >= min_rinde]
#         else:
#             rinde[Rinde] = np.where(rinde[Rinde] < min_rinde,min_rinde,rinde[Rinde])
#     if max_rinde is not None:
#         if fm == 0:
#             rinde = rinde[rinde[Rinde] <= max_rinde]
#         else:
#             rinde[Rinde] = np.where(rinde[Rinde] > max_rinde,max_rinde,rinde[Rinde])

#     # Creo la clase de FieldLib

#     rinde_fl = fl(rinde)

#     # Transformo los datos para que sean compatibles con las imágenes satelitales

#     rinde_fl.GDF2xarray(yield_fld=Rinde)

#     plt.figure(figsize=(12, 5))

#     plt.subplot(1, 2, 1)


#     rinde_fl.imgd.plot(robust=True, cmap=cmap)
#     plt.xticks([])
#     plt.yticks([])
#     plt.xlabel('')
#     plt.ylabel('')
#     plt.title('Mapa de Rendimiento')
# #    plt.savefig(path_img + 'rinde.png')

#     plt.subplot(1, 2, 2)

#     rinde[Rinde].hist(bins=20, density=True,edgecolor='black')
#     plt.title('Distribución de Rendimiento')
#     plt.xlabel('Tn/Ha')
#     plt.savefig('imagenes/rinde.png')

# def rinde_process():
#     max_rinde = 7
#     min_rinde = 1.5
#     cmap = 'RdYlGn'
#     fm = 0

#     Rinde = 'Rendimient'
#     rinde = gpd.read_file("shapes/cosecha_plot/cosecha_1.shp")    

#     # Apply filters if necessaryif min_rinde isnotNone:
#     if fm == 0:
#             rinde = rinde[rinde[Rinde] >= min_rinde]
#     else:
#             rinde[Rinde] = np.where(rinde[Rinde] < min_rinde, min_rinde, rinde[Rinde])
#     if max_rinde is not None:
#         if fm == 0:
#             rinde = rinde[rinde[Rinde] <= max_rinde]
#         else:
#             rinde[Rinde] = np.where(rinde[Rinde] > max_rinde, max_rinde, rinde[Rinde])

#     # Create FieldLib class
#     rinde_fl = fl(rinde)

#     # Transform data to be compatible with satellite images
#     rinde_fl.GDF2xarray(yield_fld=Rinde)

#     fig = plt.figure(figsize=(12, 5))
    
#     # Set background color
#     fig.patch.set_facecolor('#0e1117')

#     # Mapa de Rendimiento
#     ax1 = plt.subplot(1, 2, 1)
#     ax1.set_facecolor('#0e1117')
#     rinde_fl.imgd.plot(robust=True, cmap=cmap, ax=ax1)
#     ax1.set_xticks([])
#     ax1.set_yticks([])
#     ax1.set_xlabel('')
#     ax1.set_ylabel('')
#     ax1.set_title('Mapa de Rendimiento', color='white', fontsize=16, weight='bold', pad=20)

#     # Distribución de Rendimiento
#     ax2 = plt.subplot(1, 2, 2)
#     ax2.set_facecolor('#0e1117')
#     rinde[Rinde].hist(bins=20, density=True, edgecolor='yellow', linewidth=1.5, color='green', alpha=0.7, ax=ax2)
#     ax2.set_title('Distribución de Rendimiento', color='white', fontsize=16, weight='bold', pad=20)
#     ax2.set_xlabel('Tn/Ha', color='white', fontsize=12, weight='light', labelpad=15)
#     ax2.set_ylabel('Density', color='white', fontsize=12, weight='light', labelpad=15)
#     ax2.tick_params(axis='x', colors='white', labelsize=10)
#     ax2.tick_params(axis='y', colors='white', labelsize=10)
#     ax2.grid(False)

#     plt.tight_layout()
#     plt.savefig('imagenes/rinde.png', format='png', dpi=300, bbox_inches='tight')
#     st.pyplot(fig)

def rinde_process():
    max_rinde = 7
    min_rinde = 1.5
    cmap = 'RdYlGn'
    fm = 0

    Rinde = 'Rendimient'
    rinde = gpd.read_file("shapes/cosecha_plot/cosecha_1.shp")    

    # Apply filters if necessaryif min_rinde isnotNone:
    if fm == 0:
            rinde = rinde[rinde[Rinde] >= min_rinde]
    else:
            rinde[Rinde] = np.where(rinde[Rinde] < min_rinde, min_rinde, rinde[Rinde])
    if max_rinde is not None:
        if fm == 0:
            rinde = rinde[rinde[Rinde] <= max_rinde]
        else:
            rinde[Rinde] = np.where(rinde[Rinde] > max_rinde, max_rinde, rinde[Rinde])

    # Create figure for "Distribución de Rendimiento"
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Set background color
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    # Plot histogram
    rinde[Rinde].hist(bins=20, density=True, edgecolor='yellow', linewidth=1.5, color='green', alpha=0.7, ax=ax)
    ax.set_title('Distribución de Rendimiento', color='white', fontsize=16, weight='bold', pad=20)
    ax.set_xlabel('Tn/Ha', color='white', fontsize=12, weight='light', labelpad=15)
    ax.set_ylabel('Density', color='white', fontsize=12, weight='light', labelpad=15)
    ax.tick_params(axis='x', colors='white', labelsize=10)
    ax.tick_params(axis='y', colors='white', labelsize=10)
    ax.grid(False)

    plt.tight_layout()
    plt.savefig('imagenes/rinde-1.png', format='png', dpi=300, bbox_inches='tight')
    st.pyplot(fig)



# fa.extremos(gdf_union,presc,Dosis,Rinde,path_img, qt=0.05, figsize=(8,12))

####################### me tira error en "Ambiente" -> será qué tengo otro archivo?

# def extremos():
#     Rinde = 'Rendimient'
#     Dosis = 'Dosis Pres'
#     figsize = (8, 12)
#     qt=0.05
#     cmap='RdYlGn'

#     presc = gpd.read_file("shapes/ambientacion_plot/ambientacion_1.shp")
#     rinde = gpd.read_file("shapes/cosecha_plot/cosecha_1.shp")
#     gdf_union = gpd.sjoin(rinde, presc, how="inner", predicate="within")

#     fig, ax = plt.subplots(figsize=figsize)
#     presc.plot(ax=ax, column=Dosis, edgecolor='black', alpha=0.1, cmap=cmap)
#     for amb in gdf_union['Ambiente'].unique():
#         if qt < 0.5:
#             filt = (gdf_union[Rinde]<gdf_union[gdf_union['Ambiente']==amb][Rinde].quantile(qt))
#         else:
#             filt = (gdf_union[Rinde]>gdf_union[gdf_union['Ambiente']==amb][Rinde].quantile(qt))
#         gdf_union[(gdf_union['Ambiente']==amb) & filt]['geometry'].plot(ax=ax, markersize=3)
#         ax.set_xticklabels('')
#         ax.set_yticklabels('')
#         ax.set_xticks([])
#         ax.set_yticks([])
#         if qt < 0.5:
#             plt.title('{:4.2f}% de valores mínimos de rendimiento en cada ambiente'.format(qt*100))
#         else:
#             plt.title('{:4.2f}% de valores maximos de rendimiento en cada ambiente'.format((1-qt)*100))
#     if qt < 0.5:
#         plt.savefig('imagenes/xtr_inf.png')
#     else:
#         plt.savefig('imagenes/xtr_sup.png')
#     st.pyplot(fig)

# def extremos():
#     Rinde = 'Rendimient'
#     Dosis = 'Dosis Pres'
#     figsize = (8, 12)
#     qt = 0.05
#     cmap = 'RdYlGn'

#     presc = gpd.read_file("shapes/ambientacion_plot/ambientacion_1.shp")
#     rinde = gpd.read_file("shapes/cosecha_plot/cosecha_1.shp")
#     gdf_union = gpd.sjoin(rinde, presc, how="inner", predicate="within")

#     fig, ax = plt.subplots(figsize=figsize)
    
#     # Set background color
#     fig.patch.set_facecolor('#0e1117')
#     ax.set_facecolor('#0e1117')

#     # Plotting prescription map with transparency
#     presc.plot(ax=ax, column=Dosis, edgecolor='yellow', alpha=0.1, cmap=cmap)

#     for amb in gdf_union['Ambiente'].unique():
#         if qt < 0.5:
#             filt = (gdf_union[Rinde] < gdf_union[gdf_union['Ambiente'] == amb][Rinde].quantile(qt))
#         else:
#             filt = (gdf_union[Rinde] > gdf_union[gdf_union['Ambiente'] == amb][Rinde].quantile(qt))

#         gdf_union[(gdf_union['Ambiente'] == amb) & filt]['geometry'].plot(
#             ax=ax, markersize=3, color='green', alpha=0.7
#         )
    
#     ax.set_xticks([])
#     ax.set_yticks([])
    
#     title_color = 'white'
#     if qt < 0.5:
#         plt.title('{:4.2f}% de valores mínimos de rendimiento en cada ambiente'.format(qt * 100), color=title_color, fontsize=16, weight='bold', pad=20)
#     else:
#         plt.title('{:4.2f}% de valores máximos de rendimiento en cada ambiente'.format((1 - qt) * 100), color=title_color, fontsize=16, weight='bold', pad=20)

#     if qt < 0.5:
#         output = 'imagenes/xtr_inf.png'
#     else:
#         output = 'imagenes/xtr_sup.png'
    
#     plt.tight_layout()
#     plt.savefig(output, format='png', dpi=300, bbox_inches='tight')
#     st.pyplot(fig)

# def rinde_process():
#     max_rinde = 7
#     min_rinde = 1.5
#     cmap = 'RdYlGn'
#     fm = 0

#     Rinde = 'Rendimient'
#     rinde = gpd.read_file("shapes/cosecha_plot/cosecha_1.shp")    

#     # Filtro si hace faltaif min_rinde isnotNone:
#     if fm == 0:
#         rinde = rinde[rinde[Rinde] >= min_rinde]
#     else:
#         rinde[Rinde] = np.where(rinde[Rinde] < min_rinde, min_rinde, rinde[Rinde])
#     if max_rinde is not None:
#         if fm == 0:
#             rinde = rinde[rinde[Rinde] <= max_rinde]
#         else:
#             rinde[Rinde] = np.where(rinde[Rinde] > max_rinde, max_rinde, rinde[Rinde])

#     rinde_fl = fl(rinde)

#     rinde_fl.GDF2xarray(yield_fld=Rinde)

#     fig = plt.figure(figsize=(12, 5))
    
#     # Set background color
#     fig.patch.set_facecolor('#0e1117')

#     # Mapa de Rendimiento
#     ax1 = plt.subplot(1, 2, 1)
#     ax1.set_facecolor('#0e1117')
#     rinde_fl.imgd.plot(robust=True, cmap=cmap, ax=ax1)
#     ax1.set_xticks([])
#     ax1.set_yticks([])
#     ax1.set_xlabel('')
#     ax1.set_ylabel('')
#     ax1.set_title('Mapa de Rendimiento', color='white', fontsize=16, weight='bold', pad=20)
    
#     # Distribución de Rendimiento
#     ax2 = plt.subplot(1, 2, 2)
#     ax2.set_facecolor('#0e1117')
#     rinde[Rinde].hist(bins=20, density=True, edgecolor='yellow', linewidth=1.5, color='green', alpha=0.7, ax=ax2)
#     ax2.set_title('Distribución de Rendimiento', color='white', fontsize=16, weight='bold', pad=20)
#     ax2.set_xlabel('Tn/Ha', color='white', fontsize=12, weight='light', labelpad=15)
#     ax2.set_ylabel('Density', color='white', fontsize=12, weight='light', labelpad=15)
#     ax2.tick_params(axis='x', colors='white', labelsize=10)
#     ax2.tick_params(axis='y', colors='white', labelsize=10)
#     ax2.grid(False)

#     plt.tight_layout()
#     plt.savefig('imagenes/rinde.png', format='png', dpi=300, bbox_inches='tight')
#     st.pyplot(fig)


# df = fa.map_vs_presc(gdf_union, rinde_fl, presc, Rinde, Dosis, Superficie, path_img)

# def map_vs_presc():
    
#     Rinde = 'Rendimient'
#     Dosis = 'Dosis Pres'
#     Superficie = 'Area'
#     Altitud = 'Altitud'
#     labels = ['Malo', 'Regular', 'Bueno Menos', 'Bueno Mas', 'Excelente']
#     figsize = (7, 5)

#     presc = gpd.read_file("shapes/ambientacion_plot/ambientacion_1.shp")
#     rinde = gpd.read_file("shapes/cosecha_plot/cosecha_1.shp")
#     gdf_union = gpd.sjoin(rinde, presc, how="inner", predicate="within")

#     h_dict = dict([(d,0) for d in np.sort(presc[Dosis].unique())])
#     for i in range(len(presc)):
#         h_dict[presc.iloc[i][Dosis]] += point_in_pol(rinde_fl.imgd, presc.iloc[i:i+1], rinde_fl)

#     df = gdf_union[[Dosis,Rinde]].groupby(Dosis).agg({Rinde:'mean' }).reset_index()

#     df['Superficie'] = presc[[Dosis,Superficie]].groupby(Dosis).sum().values
#     df['Sup. Mapeada'] = df[Dosis].apply(lambda x: h_dict[x]/100)

#     df['Rinde Total'] = df['Sup. Mapeada'] * df[Rinde]

#     df.columns = ['Dosis', 'Rendimiento', 'Superficie', 'Sup. Mapeada','Rinde Total']

#     # print(df)

#     # df.to_csv('imagenes/tab1.csv', index=False)

#     return df

def dist_por_amb():
    Rinde = 'Rendimient'
    Dosis = 'Dosis Pres'
    Superficie = 'Area'
    Altitud = 'Altitud'
    labels = ['Malo', 'Regular', 'Bueno Menos', 'Bueno Mas', 'Excelente']
    figsize = (7, 5)

    presc = gpd.read_file("shapes/ambientacion_plot/ambientacion_1.shp")
    rinde = gpd.read_file("shapes/cosecha_plot/cosecha_1.shp")
    gdf_union = gpd.sjoin(rinde, presc, how="inner", predicate="within")

    rend_dosis = gdf_union[[Dosis, Rinde]].groupby(Dosis).mean()
    rend_dosis.reset_index(inplace=True)
    unique_values = np.sort(gdf_union[Dosis].unique())

    dd = []
    dr = []
    for i in range(1, len(unique_values)):
        dd += [(rend_dosis.iloc[i, 0] / rend_dosis.iloc[i-1, 0] - 1) * 100]
        dr += [(rend_dosis.iloc[i, 1] / rend_dosis.iloc[i-1, 1] - 1) * 100]

    if labels is None:
        labels = [str(d) for d in gdf_union[Dosis].unique()]
    d_dict = {}
    n = 0
    dosis_unique = np.sort(gdf_union[Dosis].unique())

    for dosis in dosis_unique:
        d_dict[str(n)] = gdf_union[gdf_union[Dosis] == dosis][Rinde]
        n += 1
    data = pd.DataFrame(d_dict)
    min_bin = gdf_union[Rinde].min().round(0)
    max_bin = gdf_union[Rinde].max().round(0)

    # Create horizontally stacked subplots
    na = data.shape[1]
    fig, axs = plt.subplots(1, na, figsize=figsize)
    
    # Set background color
    fig.patch.set_facecolor('#0e1117')
    for ax in axs:
        ax.set_facecolor('#0e1117')

    colors = sns.color_palette("RdYlGn", n_colors=na)
    bins = [min_bin + i * (max_bin - min_bin) / 20 for i in range(20)]

    for i, ax in enumerate(axs):
        ax.hist(
            data[str(i)], 
            bins=bins, 
            orientation='horizontal', 
            edgecolor='yellow', 
            linewidth=1.5, 
            color=colors[i], 
            alpha=0.7
        )
        ax.invert_xaxis()
        ax.set_title(labels[i], color='white', fontsize=12, weight='light', pad=10)
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
            ax.set_ylabel('Tn/Ha', color='white', fontsize=12, weight='light', labelpad=15)
            ax.tick_params(axis='y', colors='white')

        ax.grid(False)
        if i == n // 2:
            ax.set_xlabel('Ambiente', color='white', fontsize=12, weight='light', labelpad=15)

    for ax in axs.flat:
        ax.label_outer()

    plt.tight_layout()
    plt.savefig('imagenes/dist-rend.png', format='png', dpi=300, bbox_inches='tight')
    st.pyplot(fig)

def boxplot():
    Rinde = 'Rendimient'
    Dosis = 'Dosis Pres'
    Superficie = 'Area'
    Altitud = 'Altitud'
    labels = ['Malo', 'Regular', 'Bueno Menos', 'Bueno Mas', 'Excelente']

    presc = gpd.read_file("shapes/ambientacion_plot/ambientacion_1.shp")
    rinde = gpd.read_file("shapes/cosecha_plot/cosecha_1.shp")
    gdf_union = gpd.sjoin(rinde, presc, how="inner", predicate="within")

    if labels is None:
        labels = [str(d) for d in gdf_union[Dosis].unique()]

    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Set background colors
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    sns.boxplot(
        data=gdf_union, 
        y=Rinde, 
        x=Dosis, 
        ax=ax, 
        palette="RdYlGn",
        boxprops=dict(edgecolor='white', linewidth=1),
        medianprops=dict(color='white', linewidth=2),
        whiskerprops=dict(color='white', linewidth=2),
        capprops=dict(color='white', linewidth=2),
        flierprops=dict(marker='o', color='white', markersize=5, alpha=0.7)
    )
    
    ax.set_xticks(ticks=range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', color='white', fontsize=10)
    
    ax.set_ylabel('Rinde', color='white', fontsize=12, weight='light', labelpad=15)
    ax.set_xlabel('Ambiente', color='white', fontsize=12, weight='light', labelpad=15)
    
    ax.tick_params(axis='x', colors='white', labelsize=10)
    ax.tick_params(axis='y', colors='white', labelsize=10)
    
    ax.grid(True, color='white', linestyle='dashed', linewidth=0.7)
    
    # Set title (optional)
    ax.set_title('Boxplot de Rinde por Ambiente', color='white', fontsize=16, weight='bold', pad=20)
    
    # Save the figure
    plt.tight_layout()
    fig.savefig('imagenes/boxplot.png', format='png', dpi=300, bbox_inches='tight')
    
    # Show the plot in Streamlit
    st.pyplot(fig)



def lotes_plot():
    st.markdown('<h2 style="font-size: 15px;">Polígono del Lote</h2>', unsafe_allow_html=True)
    lotes = gpd.read_file("shapes/lote_plot/lote_2.shp")
    lote = lotes[lotes['Lote'] == '2bc AS']
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    
    lote.plot(color='g', ax=ax, edgecolor='white', linewidth=1.5, alpha=0.6)
    
    ax.set_title("Polígono correspondiente al Lote", color='white', fontsize=16, weight='bold', pad=20)
    ax.set_xlabel("Latitud", color='white', fontsize=12, weight='light', labelpad=15)
    ax.set_ylabel("Longitud", color='white', fontsize=12, weight='light', labelpad=15)
    
    ax.tick_params(axis='x', colors='white', labelsize=10)
    ax.tick_params(axis='y', colors='white', labelsize=10)

    plt.xticks(rotation=75, ha='right')
    
    plt.grid(True, color='white', linestyle='dashed', linewidth=0.7)
    
    lote.boundary.plot(ax=ax, color='white', linewidth=3, alpha=0.2)
    lote.boundary.plot(ax=ax, marker='o', markersize=5, color='yellow', linestyle='None')
    fig.savefig("imagenes/lote_plot.png", format="png", dpi=300)

    # Render the plot
    st.pyplot(fig)

def ambientacion_plot():
    st.markdown('<h2 style="font-size: 15px;">Ambientación</h2>', unsafe_allow_html=True)
    Rinde = 'Rendimient'
    Dosis = 'Dosis Pres'
    Superficie = 'Area'
    Altitud = 'Altitud'
    Amb = ['Malo', 'Regular', 'Bueno Menos', 'Bueno Mas', 'Excelente']
    figsize = (10, 7)
    cmap = 'RdYlGn'

    presc = gpd.read_file("shapes/ambientacion_plot/ambientacion_1.shp")

    fig, ax = plt.subplots(figsize=figsize)

    # Set background color
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    unique_values = np.sort(presc[Dosis].unique())
    amb_dict = dict(zip(unique_values, Amb))
    presc['Ambiente'] = presc[Dosis].map(amb_dict)

    cmap = plt.get_cmap(cmap)

    presc.plot(ax=ax, column=Dosis, cmap=cmap, edgecolor='white', linewidth=1, alpha=0.8)

    ax.set_title("Ambientación", color='white', fontsize=16, weight='bold', pad=20)
    ax.set_xlabel("Latitud", color='white', fontsize=12, weight='light', labelpad=15)
    ax.set_ylabel("Longitud", color='white', fontsize=12, weight='light', labelpad=15)

    plt.xticks(rotation=75, ha='right')

    ax.tick_params(axis='x', colors='white', labelsize=10)
    ax.tick_params(axis='y', colors='white', labelsize=10)

    plt.grid(True, color='white', linestyle='dashed', linewidth=0.7)

    presc.boundary.plot(ax=ax, color='white', linewidth=0.5, alpha=0.2)
    presc.boundary.plot(ax=ax, marker='o', markersize=5, color='yellow', linewidth=0.5, linestyle='None')

    boundaries = np.arange(0, len(unique_values) + 1)
    norm = mcolors.BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=1)
    cbar.set_ticks([i + 0.5 for i in range(len(unique_values))])
    cbar.set_ticklabels(Amb)
    cbar.ax.tick_params(axis='both', colors='white', labelsize=10)
    cbar.set_label('Ambiente', color='white', fontsize=12, weight='light', labelpad=10)

    if presc.crs is None:
        presc = presc.set_crs('4326')
    presc = presc.to_crs('32720')

    fig.savefig("imagenes/ambientacion_plot.png", format="png", dpi=300)

    # Render the plot
    st.pyplot(fig)

def mapa_de_rendimiento():
    st.markdown('<h2 style="font-size: 15px;">Mapa de Rendimiento</h2>', unsafe_allow_html=True)
    rinde = gpd.read_file("shapes/cosecha_plot/cosecha_1.shp")
    fig, ax = plt.subplots(figsize=(10, 7))
    
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    rinde['Rendimient'] = np.where(rinde['Rendimient'] > 7,7, rinde['Rendimient'])
    rinde['Rendimient'] = np.where(rinde['Rendimient'] < 2,2, rinde['Rendimient'])
    
    rinde.plot(column='Rendimient', cmap='RdYlGn', ax=ax)
    
    norm = Normalize(vmin=0, vmax=8)
    cax = plt.gca().inset_axes([1, 0, 0.05, 1])
    colorbar = ColorbarBase(cax, cmap='RdYlGn', norm=norm, orientation='vertical')

    for label in cax.get_yticklabels():
        label.set_color('white')
    
    ax.set_title("Mapa de Rendimiento", color='white', fontsize=16, weight='bold', pad=20)
    ax.set_xlabel("Longitud", color='white', fontsize=12, weight='light', labelpad=15)
    ax.set_ylabel("Latitud", color='white', fontsize=12, weight='light', labelpad=15)

    plt.xticks(rotation=75, ha='right')
    
    ax.tick_params(axis='x', colors='white', labelsize=10)
    ax.tick_params(axis='y', colors='white', labelsize=10)
    
    plt.grid(True, color='white', linestyle='dashed', linewidth=0.7)
    fig.savefig("imagenes/mapa_rendimiento_plot.png", format="png", dpi=300)

    # Render the plot
    st.pyplot(fig)

def rend_por_ambientes():
    presc = gpd.read_file("shapes/ambientacion_plot/ambientacion_1.shp")
    rinde = gpd.read_file("shapes/cosecha_plot/cosecha_1.shp")
    
    
    Rinde = 'Rendimient'
    Dosis = 'Dosis Pres'
    Superficie = 'Area'
    Altitud = 'Altitud'
    Amb = ['Malo', 'Regular', 'Bueno Menos', 'Bueno Mas', 'Excelente']
    figsize = (5, 3)
    cmap = 'RdYlGn'

    rinde = rinde.to_crs('32720')
    presc = presc.to_crs('32720')
    
    unique_values = np.sort(presc[Dosis].unique())

    gdf_union = gpd.sjoin(rinde, presc, how="inner", predicate="within")
    
    rend_dosis = gdf_union[[Dosis, Rinde]].groupby(Dosis).mean()
    rend_dosis.reset_index(inplace=True)
    unique_values = np.sort(gdf_union[Dosis].unique())

    dd = []
    dr = []
    for i in range(1, len(unique_values)):
        dd += [(rend_dosis.iloc[i, 0] / rend_dosis.iloc[i-1, 0] - 1) * 100]
        dr += [(rend_dosis.iloc[i, 1] / rend_dosis.iloc[i-1, 1] - 1) * 100]

    fig, ax = plt.subplots(figsize=figsize)
    
    # Set background colors
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    xticks = range(1, len(dd) + 1)
    xlabels = ['Dif ' + str(i) + '-' + str(i+1) for i in range(1, len(dd) + 1)]
    
    ax.plot(xticks, dd, label='Dif Dosis', marker='o', color='red')
    bars = ax.bar(xticks, dr, label='Dif Rendimiento', color='green', alpha=0.7, edgecolor='yellow', linewidth=1.5)

    # Move legend outside of the plot
    ax.legend(facecolor='#0e1117', edgecolor='white', labelcolor='white', loc='upper left', bbox_to_anchor=(1, 1))
    
    ax.set_xlabel('Diferencias entre ambientes', color='white', fontsize=12, weight='light', labelpad=15)
    ax.set_ylabel('Diferencias en %', color='white', fontsize=12, weight='light', labelpad=15)
    
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=45, ha='right', color='white', fontsize=10)
    
    ax.tick_params(axis='x', colors='white', labelsize=10)
    ax.tick_params(axis='y', colors='white', labelsize=10)
    
    ax.grid(True, color='white', linestyle='dashed', linewidth=0.7)
    
    fig.savefig('imagenes/dosis-rend.png', format='png', dpi=300, bbox_inches='tight')


    st.pyplot(fig)

dashboard()

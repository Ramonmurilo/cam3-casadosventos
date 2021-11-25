import pandas as pd
import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point
from cftime import DatetimeNoLeap
import pendulum
from dateutil.relativedelta import relativedelta

def addcbar(plot, ax, cblabel="", orientation="horizontal", rotation=0, ticks=None):
    """
    Adiciona uma barra de cores a um mapa automaticamente.

    A função adiciona um eixo independente à figura
    """
    # cria um eixo para a colorbar
    cbar = plt.colorbar(plot, orientation=orientation, pad=0.05, ax=ax, shrink = 1.0, aspect=40)
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(cblabel, fontsize = 17)
    
    return cbar

def geoaxes_format(ax, longitude_interval=2.5):
    """
    Formata eixos com referências geográficas para mapas cartopy.

    Essa função é usada pela função basemap.

    Parâmetros
    ----------
    ax: matplotlib.axes._subplots.AxesSubplot
        Eixo matplotlib a ser formatado como eixo geográfico

    Retorna
    -------
    gl: cartopy.mpl.geoaxes.GeoAxesSubplot
        Eixo geográfico formatado.
    """
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='black', alpha=0.3, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_left = True
    gl.ylabels_right = False
    gl.ylines = True
    gl.xlines = True
    gl.xlocator = mticker.FixedLocator(np.arange(-90,50,10)) #-70,50,20
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 15}
    gl.ylabel_style = {'size': 15}
    
    return gl


def get_lead(ano, mes, lead):
    """
    Retorna a média dos membros para um lead.
    OBS.: acho que deve lidar bem com viradas de ano, mas testei pouco.
    """

    # forma um objeto de tempo a partir dos parâmetros ano e mês
    data = datetime.strptime(f"{ano}-{mes}", "%Y-%m")

    # separa os valores da data para formar os caminhos até as pastas dos arqs
    mes_2_digitos = data.strftime("%m")
    ano_2_digitos = data.strftime("%y")
    ano_4_digitos = data.strftime("%Y")

    # se necessário, converte o tipo da variável lead para int
    lead = lead if isinstance(lead, int) else int(lead)

    # data do lead = data da rodada + valor do lead
    # se rodada = Jan/2020, lead 1 = Fev/2020
    data_lead = data + relativedelta(months=lead)

    # caso a data do lead exceda o horizonte a partir da data informada
    if data_lead > data + relativedelta(months=6):
        raise ValueError("Quer mais que 6 leads pra quê, jovem?")

    # separa os valores da data do lead para formar os caminhos
    mes_lead_2_digitos = data_lead.strftime("%m")
    ano_lead_4_digitos = data_lead.strftime("%Y")

    # caminho fixo até a pasta das rodadas
    caminho_base = "dados"

    # caminho para a pasta de uma rodada específica
    caminho_rodada = f"{caminho_base}"

    # lista para guardar os datasets Xarray de cada membro
    membros = list()

    print(f"Data da rodada: {data}. Data do lead {lead}: {data_lead}.")

    # coleta os membros (membros usados: 4 a 9)
    # dia de rodada = membro (._.")
    for dia_rodada in range(4, 10):

        # forma o caminho até o arquivo de um membro da rodada. Formato:
        # r(ano)(mes)0(dia da rodada).cam2.h0.(ano do lead)-(mes do lead)
        nome_arquivo = f"r{ano_2_digitos}{mes_2_digitos}0{dia_rodada}.cam2.h0.{ano_lead_4_digitos}-{mes_lead_2_digitos}.nc"
        caminho_completo = f'{caminho_rodada}/{nome_arquivo}'

        print(f"membro {dia_rodada}: {caminho_completo}")

        # lê o NC
        membro = xr.open_dataset(caminho_completo, decode_times=False)
        membro = xr.decode_cf(membro, use_cftime=True)

        # adiciona uma coordenada para indicar o membro do arquivo
        membro["membro"] = dia_rodada
        membros.append(membro)

    # concatena os membros do lead
    membros = xr.concat(membros, dim="membro")#.mean(dim=["membro"])

    return membros

def preparar_dados(ano, mes):
    """
    Checa/cria diretórios e subdiretórios para os plots.
    Retorna um dict contendo todos os leads para a data requisitada.
    """

    mes = int(mes)
    if mes < 10:                    
      mes_mkdir = '0'+ str(mes) 
    else:
      mes_mkdir = str(mes)   
    if os.path.exists (f'IMAGENS/CAM3/{ano}/r{ano_2digitos}{mes_mkdir}') == True:
      print (f'A pasta /IMAGENS/CAM3/{ano}/r{ano_2digitos}{mes_mkdir} já existe !') 
    else :
      os.mkdir(f'IMAGENS/CAM3/{ano}/r{ano_2digitos}{mes_mkdir}')
      os.mkdir(f'IMAGENS/CAM3/{ano}/r{ano_2digitos}{mes_mkdir}/ensemble')
      print (f'As pastas r{ano_2digitos}{mes_mkdir} e ensemble foram criadas com sucesso!')
    mes = str(mes) 

    leads = dict()
    for lead in range(0, 7):
        dataset = get_lead(ano=ano, mes=mes, lead=lead)
        leads.update({lead: dataset})

    print("Fim.")
    return leads

def vento(ax, dados_cam, clima, membro, lev):
  """
  Gera as linhas de corrente para o nível determinado.
  *** argumentos ***
  ax: eixo da figura
  dados_cam: arquivo contendo o lead para gerar anomalia
  clima: arquivo de clima para gerar anomalia
  membro: dia que a figura deve ser gerada ou 'ensemble' para a média dos membros
  lev: nível atmosférico a ser gerado
  """ 

  if membro == 'ensemble':
    dataset = dados_cam.mean(dim='membro')
  else:
    dataset = dados_cam.sel(membro=membro)      

  lev = lev
  vento_u = dataset.U.sel(lev=lev, method='nearest')                  # Escolhe o nível na dimensão 'lev'
  vento_v = dataset.V.sel(lev=lev, method='nearest')                  # Escolhe o nível na dimensão 'lev'

  clima_u = clima.U.sel(lev=lev, method='nearest')                    # Escolhe o nível na dimensão 'lev'
  clima_v = clima.V.sel(lev=lev, method='nearest')                    # Escolhe o nível na dimensão 'lev'

  anom_u = vento_u.groupby('time.month').mean() - clima_u.groupby('time.month').mean().values                       # Cria anomalia de U
  anom_v = vento_v.groupby('time.month').mean() - clima_v.groupby('time.month').mean().values                       # Cria anomalia de V

  anom_u_sel = anom_u.isel(month=0)
  anom_v_sel = anom_v.isel(month=0)

  uvel, lonu = add_cyclic_point(anom_u_sel, coord=anom_u_sel.lon)
  vvel, lonv = add_cyclic_point(anom_v_sel, coord=anom_u_sel.lon)

  lonu = np.where(lonu>=180., lonu-360.,lonu)

  magnitude = (uvel ** 2 + vvel ** 2) ** 0.5

  levels=[0,2,4,6,8,10,15,20,25]
  norm = mpl.colors.Normalize(vmin=levels[0], vmax=levels[-1])

  if lev == 1000 or lev == 850:
    sp = ax.quiver(lonu, anom_v_sel.lat,uvel,vvel,magnitude,scale=50,norm=norm,cmap='rainbow', transform=ccrs.PlateCarree(), pivot='middle')

  elif lev == 200 or lev ==500:
    sp = ax.streamplot(lonu, anom_v_sel.lat, uvel, vvel,
                      linewidth=2,
                      cmap="rainbow",
                      arrowsize=3,
                      density=5,
                      color=magnitude,
                      norm=norm,
                      transform=ccrs.PlateCarree())
  
  #cbar = addcbar(sp.lines, ax, "m/s", orientation= "vertical")

  crs_latlon = ccrs.PlateCarree()
  gl = geoaxes_format(ax)
  ax.coastlines("50m")
  ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
  estados_br = cfeature.NaturalEarthFeature(category='cultural', scale='50m', facecolor='none', name='admin_1_states_provinces')
  #estados_br = cfeature.NaturalEarthFeature(category='cultural', scale='50m', facecolor='none', name='admin_1_states_provinces_shp')
  ax.add_feature(estados_br, edgecolor='gray')
  ax.set_extent([-83,-30,-53, 7], crs=crs_latlon)
  add_contornos(ax=ax, shapefiles=['estados.shp','america_do_sul.shp'])

  if membro == 'ensemble':
    ax.set_title(f"Média dos 6 membros", fontdict={'fontsize': 18}, loc = 'center')
  else:
    ax.set_title(f"dia {membro}", fontdict={'fontsize': 18}, loc = 'center')
  return sp

def psl(ax, dados_cam, clima, membro):
  """
  Gera a anomalia de pressão.
  *** argumentos ***
  ax: eixo da figura
  dados_cam: arquivo contendo os leads
  clima: arquivo de clima para gerar anomalia
  membro: dia que a figura deve ser gerada ou 'ensemble' para a média dos membros
  """ 

  if membro == 'ensemble':
    dataset = dados_cam.mean(dim='membro')
  else:
    dataset = dados_cam.sel(membro=membro)
  
  pressao = dataset.PSL
  pressao_clima = clima.PSL

  pressao_anom = (pressao.groupby('time.month').mean() - pressao_clima.groupby('time.month').mean().values)/100
  pressao_sel = pressao_anom.isel(month=0)

  pressao_wrap, lon_wrap = add_cyclic_point(pressao_sel.values, coord=pressao_sel.lon)

  levels=np.arange(-6,7,1)
  norm = mpl.colors.Normalize(vmin=levels[0], vmax=levels[-1])
  
  #levelsc = [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
  #norm = mpl.colors.BoundaryNorm(levelsc,255)                                   
  #cmap = mpl.colors.LinearSegmentedColormap.from_list(eneva, _eneva_dec,N=255)
  #cmap=plt.cm.RdBu_r
  #ncores = cmap.N

  cont = ax.contourf(lon_wrap, pressao_sel.lat, pressao_wrap,
                    norm=normc,
                    levels=levelsc,
                    cmap= cmapc,
                    extend='both',
                    transform=ccrs.PlateCarree())

  gl = geoaxes_format(ax)
  ax.coastlines("50m")
  estados_br = cfeature.NaturalEarthFeature(category='cultural', scale='50m', facecolor='none', name='admin_1_states_provinces')
  #estados_br = cfeature.NaturalEarthFeature(category='cultural', scale='50m', facecolor='none', name='admin_1_states_provinces_shp')
  ax.add_feature(estados_br, edgecolor='gray')
  ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
  ax.set_extent([-83,-30,-53, 7])
  add_contornos(ax=ax, shapefiles=['bacias.shp'])
  
  if membro == 'ensemble':
    ax.set_title(f"Média dos 6 membros", fontdict={'fontsize': 18}, loc = 'center')
  else:
    ax.set_title(f"dia {membro}", fontdict={'fontsize': 18}, loc = 'center')
  return cont

def temp(ax, dados_cam, clima, membro):
  """
  Gera a anomalia de temperatura em 1000 hpa.
  *** argumentos ***
  ax: eixo da figura
  dados_cam: arquivo contendo os leads
  clima: arquivo de clima para gerar anomalia
  membro: dia que a figura deve ser gerada ou 'ensemble' para a média dos membros
  """ 

  if membro == 'ensemble':
    dataset = dados_cam.mean(dim='membro')
  else:
    dataset = dados_cam.sel(membro=membro)
  
  temp = dataset.T.sel(lev=1000, method='nearest')
  temp_clima = clima.T.sel(lev=1000, method='nearest')

  temp_anom = temp.groupby('time.month').mean() - temp_clima.groupby('time.month').mean().values
  temp_sel = temp_anom.isel(month=0)

  temp_wrap, lon_wrap = add_cyclic_point(temp_sel.values, coord=temp_sel.lon)

  #levels=np.arange(-6,7,1)
  #norm = mpl.colors.Normalize(vmin=levels[0], vmax=levels[-1])

  cont = ax.contourf(lon_wrap, temp_sel.lat, temp_wrap,
                    norm=norme,
                    levels=levelse,
                    cmap= cmape,
                    extend='both',
                    transform=ccrs.PlateCarree())

  gl = geoaxes_format(ax)
  ax.coastlines("50m")
  estados_br = cfeature.NaturalEarthFeature(category='cultural', scale='50m', facecolor='none', name='admin_1_states_provinces')
  #estados_br = cfeature.NaturalEarthFeature(category='cultural', scale='50m', facecolor='none', name='admin_1_states_provinces_shp')
  ax.add_feature(estados_br, edgecolor='gray')
  ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
  ax.set_extent([-83,-30,-53, 7])
  add_contornos(ax=ax, shapefiles=['estados.shp','america_do_sul.shp'])
  
  if membro == 'ensemble':
    ax.set_title(f"Média dos 6 membros", fontdict={'fontsize': 18}, loc = 'center')
  else:
    ax.set_title(f"dia {membro}", fontdict={'fontsize': 18}, loc = 'center')
  return cont

def fluxsol(ax, dados_cam, clima, membro):
  """
  Gera a anomalia de pressão.
  *** argumentos ***
  ax: eixo da figura
  dados_cam: arquivo contendo os leads
  clima: arquivo de clima para gerar anomalia
  membro: dia que a figura deve ser gerada ou 'ensemble' para a média dos membros
  """ 

  if membro == 'ensemble':
    dataset = dados_cam.mean(dim='membro')
  else:
    dataset = dados_cam.sel(membro=membro)
  
  flux_sol = dataset.FSNS
  flux_sol_clima = clima.FSNS

  sol_anom = flux_sol.groupby('time.month').mean() - flux_sol_clima.groupby('time.month').mean().values
  sol_sel = sol_anom.isel(month=0)

  sol_wrap, lon_wrap = add_cyclic_point(sol_sel.values, coord=sol_sel.lon)

  #levels=[-80,-65,-50,-35,-20,-5, 5,20,35,50,65,80]
  #norm = mpl.colors.Normalize(vmin=levels[0], vmax=levels[-1])

  cont = ax.contourf(lon_wrap, sol_sel.lat, sol_wrap,
                    norm=normd,
                    levels=levelsd,
                    cmap= cmapd,
                    extend='both',
                    transform=ccrs.PlateCarree())

  gl = geoaxes_format(ax)
  ax.coastlines("50m")
  estados_br = cfeature.NaturalEarthFeature(category='cultural', scale='50m', facecolor='none', name='admin_1_states_provinces')
  #estados_br = cfeature.NaturalEarthFeature(category='cultural', scale='50m', facecolor='none', name='admin_1_states_provinces_shp')
  ax.add_feature(estados_br, edgecolor='gray')
  ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
  ax.set_extent([-83,-30,-53, 7])
  add_contornos(ax=ax, shapefiles=['estados.shp','america_do_sul.shp'])
  
  if membro == 'ensemble':
    ax.set_title(f"Média dos 6 membros", fontdict={'fontsize': 18}, loc = 'center')
  else:
    ax.set_title(f"dia {membro}", fontdict={'fontsize': 18}, loc = 'center')
  return cont

def chuva(ax, dados_cam, clima, membro):
  """
  Gera a anomalia de precipitação.
  *** argumentos ***
  ax: eixo da figura
  dados_cam: arquivo contendo os leads
  clima: arquivo de clima para gerar anomalia
  membro: dia que a figura deve ser gerada ou 'ensemble' para a média dos membros
  """ 

  if membro == 'ensemble':
    dataset = dados_cam.mean(dim='membro')
  else:
    dataset = dados_cam.sel(membro=membro)

  precc_cam = dataset.PRECC
  precl_cam = dataset.PRECL
  precsh_cam = dataset.PRECSH
  prec_cam = (precc_cam + precl_cam + precsh_cam)*3600*24*30*1000

  precc_clima = clima.PRECC
  precl_clima = clima.PRECL
  precsh_clima = clima.PRECSH
  prec_clima = (precc_clima + precl_clima + precsh_clima)*3600*24*30*1000

  prec_anom = prec_cam.groupby('time.month').mean() - prec_clima.groupby('time.month').mean().values
  prec_sel = prec_anom.isel(month=0)

  prec_wrap, lon_wrap = add_cyclic_point(prec_sel.values, coord=prec_sel.lon)

  #levels=[-300,-200,-100,-50,-25,0,25,50,100,200,300]
  #norm = mpl.colors.Normalize(vmin=levels[0], vmax=levels[-1])

  cont = ax.contourf(lon_wrap, prec_sel.lat, prec_wrap,
                    norm=normb,
                    levels=levelsb,
                    cmap=cmapb,
                    extend='both',
                    transform=ccrs.PlateCarree())

  gl = geoaxes_format(ax)
  ax.coastlines("50m")
  estados_br = cfeature.NaturalEarthFeature(category='cultural', scale='50m', facecolor='none', name='admin_1_states_provinces')
  #estados_br = cfeature.NaturalEarthFeature(category='cultural', scale='50m', facecolor='none', name='admin_1_states_provinces_shp')
  ax.add_feature(estados_br, edgecolor='gray')
  ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
  add_contornos(ax=ax, shapefiles=['estados.shp','america_do_sul.shp'])


  ax.set_extent([-83,-30,-53, 7])
  if membro == 'ensemble':
    ax.set_title(f"Média dos 6 membros", fontdict={'fontsize': 18}, loc = 'center')
  else:
    ax.set_title(f"dia {membro}", fontdict={'fontsize': 18}, loc = 'center')
  return cont


############### interação com usuario
''' a versão com mais linhas é utilizado para facilitar o comando 'savefig' no bloco de comando 'Plots' guardando as variaveis ano e mês '''
ano = input('digite o ano da rodada com 4 dígitos:')
mes = input('digite o mês da rodada sem 0!\n(exs.: 1,2,3,4,5,6,7,8,9,10,11,12) :')
ano_2digitos = ano[-2:]           # pega dois ultimos dígitos do ano

geral = preparar_dados(ano, mes)

mes = int(mes)
if mes < 10:                   # bloco if corrige os meses sem zero apenas para utilizar no 'savefig'
  mes_savefig = '0' + str(mes) 
else:
  mes_savefig = mes

data_pendulum = pendulum.from_format(f'{ano}-{mes_savefig}', 'YYYY-MM')

''' Versão com 1 linha '''
#geral = preparar_dados(2020, 6)
#################################

count_time = datetime.now()                         # contador de tempo para gerar figuras        #ident0
for loop_lead in range(1,7):                        # loop operacional para gerar todos os leads  #ident0 #range(1,7)

  lead_escolhido = geral[loop_lead]                 # Escolhe lead |||| [0] É O LEAD 0! NÃO USAR
  data = lead_escolhido.time.values                 # Lê a data do dataset
  data_index = xr.CFTimeIndex.to_datetimeindex(data)# Transforma em datetime
  data_index = data_index - pd.Timedelta('1 day')   # Corrige a data de fechamento da previsão
  mes_lead = data_index.strftime('%m')[0]           # Transforma mês do lead em string
  ano_lead_4digitos = data_index.strftime('%Y')[0]  # Transforma ano do lead em string 4d
  ano_lead_2digitos = data_index.strftime('%y')[0]  # Transforma ano do lead em string 2d
  data_lead = data_index.strftime('%b/%Y')[0]       # Transforma da data do lead em string
  data_lead_pendulum = pendulum.from_format(data_index.strftime('%m/%Y')[0], 'MM/YYYY')

  clima = xr.open_dataset(f"/content/drive/My Drive/IC_CAM/CLIMA/{mes_lead}_climo.nc")
  
  ######## PLOT DO VENTO ########
  niveis = [1000, 850, 500, 200]
  for lev in niveis:
    fig, ax = plt.subplots(nrows=2,ncols=3,subplot_kw=dict(projection = ccrs.PlateCarree()), figsize=(32,16))
    plt.suptitle(f"LAMMOC - Anomalia de vento em {lev} hPa - {data_lead} (Cond.Inicial:{mes}/{ano_2digitos})", fontsize=24, fontweight='bold')

    vento(ax[0,0], lead_escolhido, clima, 4, lev)
    vento(ax[0,1], lead_escolhido, clima, 5, lev)
    vento(ax[0,2], lead_escolhido, clima, 6, lev)
    vento(ax[1,0], lead_escolhido, clima, 7, lev)
    vento(ax[1,1], lead_escolhido, clima, 8, lev)
    sp = vento(ax[1,2], lead_escolhido, clima, 9, lev)
    if lev == 1000 or lev == 850:
      cb = fig.colorbar(sp, ax=ax.ravel().tolist(), ticks=[0,2,4,6,8,10,15,20,25])
      cb.ax.tick_params(labelsize=20)
    elif lev == 500 or lev == 200:
      cb = fig.colorbar(sp.lines, ax=ax.ravel().tolist(), ticks=[0,2,4,6,8,10,15,20,25])
      cb.ax.tick_params(labelsize=20)
    fig.savefig(f'/content/drive/My Drive/IC_CAM/IMAGENS/CAM3/{ano}/r{ano_2digitos}{mes_savefig}/wind_anom_{lev}_membros_{mes_lead}_{ano_lead_4digitos}.png', bbox_inches='tight')

    fig, ax = plt.subplots( nrows=1,ncols=1,subplot_kw=dict(projection = ccrs.PlateCarree()), figsize=(32,16))
    plt.suptitle(f"LAMMOC - Anomalia de vento em {lev} hPa - \n{data_lead} (Cond.Inicial:{mes}/{ano_2digitos})", fontsize=24, fontweight='bold', x= 0.6)

    sp = vento(ax, lead_escolhido, clima, 'ensemble', lev)
    if lev == 1000 or lev == 850:
      cb = fig.colorbar(sp, ax=ax, ticks=[0,2,4,6,8,10,15,20,25])
      cb.ax.tick_params(labelsize=20)
    elif lev == 500 or lev == 200:
      cb = fig.colorbar(sp.lines, ax=ax, ticks=[0,2,4,6,8,10,15,20,25])
      cb.ax.tick_params(labelsize=20)
    fig.savefig(f'/content/drive/My Drive/IC_CAM/IMAGENS/CAM3/{ano}/r{ano_2digitos}{mes_savefig}/ensemble/wind_anom_{lev}_ensemble_{mes_lead}_{ano_lead_4digitos}.png', bbox_inches='tight')
  
  ######## PLOT DA PRESSÃO ########
  fig, ax = plt.subplots( nrows=2,ncols=3,subplot_kw=dict(projection = ccrs.PlateCarree()), figsize=(32,16))
  plt.suptitle(f"LAMMOC - Anomalia de Pressão ao nível do mar (hPa) - {data_lead} (Cond.Inicial:{mes}/{ano_2digitos})", fontsize=24, fontweight='bold')

  psl(ax[0,0], lead_escolhido, clima, 4)
  psl(ax[0,1], lead_escolhido, clima, 5)
  psl(ax[0,2], lead_escolhido, clima, 6)
  psl(ax[1,0], lead_escolhido, clima, 7)
  psl(ax[1,1], lead_escolhido, clima, 8)
  cont = psl(ax[1,2], lead_escolhido, clima, 9)
  cb = fig.colorbar(cont, ax=ax.ravel().tolist(), ticks=[-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6])
  cb.ax.tick_params(labelsize=20)
  #fig.colorbar(cont, ax=ax.ravel().tolist(), ticks=[-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6])
  fig.savefig(f'/content/drive/My Drive/IC_CAM/IMAGENS/CAM3/{ano}/r{ano_2digitos}{mes_savefig}/psl_anom_membros_{mes_lead}_{ano_lead_4digitos}.png', bbox_inches='tight')
  plt.close('all')

  fig, ax = plt.subplots( nrows=1,ncols=1,subplot_kw=dict(projection = ccrs.PlateCarree()), figsize=(32,16))
  plt.suptitle(f"LAMMOC - Anomalia de Pressão ao nível do mar (hPa) - \n{data_lead} (Cond.Inicial:{mes}/{ano_2digitos})", fontsize=24, fontweight='bold', x= 0.6)

  cont = psl(ax, lead_escolhido, clima, 'ensemble')
  cb = fig.colorbar(cont, ax=ax, ticks=[-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6])
  cb.ax.tick_params(labelsize=20)
  #fig.colorbar(cont, ax=ax, ticks=[-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6])
  fig.savefig(f'/content/drive/My Drive/IC_CAM/IMAGENS/CAM3/{ano}/r{ano_2digitos}{mes_savefig}/ensemble/psl_anom_ensemble_{mes_lead}_{ano_lead_4digitos}.png', bbox_inches='tight')
  plt.close('all')
  
  ######## PLOT DA TEMPERATURA ########

  fig, ax = plt.subplots( nrows=2,ncols=3,subplot_kw=dict(projection = ccrs.PlateCarree()), figsize=(32,16))
  plt.suptitle(f"LAMMOC - Anomalia de Temperatura (°C) - {data_lead} (Cond.Inicial:{mes}/{ano_2digitos})", fontsize=24, fontweight='bold')

  temp(ax[0,0], lead_escolhido, clima, 4)
  temp(ax[0,1], lead_escolhido, clima, 5)
  temp(ax[0,2], lead_escolhido, clima, 6)
  temp(ax[1,0], lead_escolhido, clima, 7)
  temp(ax[1,1], lead_escolhido, clima, 8)
  t = temp(ax[1,2], lead_escolhido, clima, 9)
  cb = fig.colorbar(t, ax=ax.ravel().tolist(), ticks=[-6,-5,-4,-3,-2,-1,1,2,3,4,5,6])
  cb.ax.tick_params(labelsize=20)
  fig.savefig(f'/content/drive/My Drive/IC_CAM/IMAGENS/CAM3/{ano}/r{ano_2digitos}{mes_savefig}/temp_anom_membros_{mes_lead}_{ano_lead_4digitos}.png', bbox_inches='tight')

  fig, ax = plt.subplots( nrows=1,ncols=1,subplot_kw=dict(projection = ccrs.PlateCarree()), figsize=(32,16))
  plt.suptitle(f"LAMMOC - Anomalia de Temperatura (°C) - \n{data_lead} (Cond.Inicial:{mes}/{ano_2digitos})", fontsize=24, fontweight='bold', x= 0.6)

  t = temp(ax, lead_escolhido, clima, 'ensemble')
  cb = fig.colorbar(t, ax=ax, ticks=[-6,-5,-4,-3,-2,-1,1,2,3,4,5,6])
  cb.ax.tick_params(labelsize=20)
  fig.savefig(f'/content/drive/My Drive/IC_CAM/IMAGENS/CAM3/{ano}/r{ano_2digitos}{mes_savefig}/ensemble/temp_anom_ensemble_{mes_lead}_{ano_lead_4digitos}.png', bbox_inches='tight')
  
  ######## PLOT DO FLUXO SOLAR ########

  fig, ax = plt.subplots( nrows=2,ncols=3,subplot_kw=dict(projection = ccrs.PlateCarree()), figsize=(32,16))
  plt.suptitle(f"LAMMOC - Anomalia de Fluxo Solar (W/m²) - {data_lead} (Cond.Inicial:{mes}/{ano_2digitos})", fontsize=24, fontweight='bold')

  fluxsol(ax[0,0], lead_escolhido, clima, 4)
  fluxsol(ax[0,1], lead_escolhido, clima, 5)
  fluxsol(ax[0,2], lead_escolhido, clima, 6)
  fluxsol(ax[1,0], lead_escolhido, clima, 7)
  fluxsol(ax[1,1], lead_escolhido, clima, 8)
  f = fluxsol(ax[1,2], lead_escolhido, clima, 9)
  cb = fig.colorbar(f, ax=ax.ravel().tolist(), ticks=[-80,-65,-50,-35,-20,-5,5,20,35,50,65,80])
  cb.ax.tick_params(labelsize=20)
  fig.savefig(f'/content/drive/My Drive/IC_CAM/IMAGENS/CAM3/{ano}/r{ano_2digitos}{mes_savefig}/solar_flux_anom_membros_{mes_lead}_{ano_lead_4digitos}.png', bbox_inches='tight')

  fig, ax = plt.subplots( nrows=1,ncols=1,subplot_kw=dict(projection = ccrs.PlateCarree()), figsize=(32,16))
  plt.suptitle(f"LAMMOC - Anomalia de Fluxo Solar (W/m²) - \n{data_lead} (Cond.Inicial:{mes}/{ano_2digitos})", fontsize=24, fontweight='bold', x= 0.6)

  f = fluxsol(ax, lead_escolhido, clima, 'ensemble')
  cb = fig.colorbar(f, ax=ax, ticks=[-80,-65,-50,-35,-20,-5,5,20,35,50,65,80])
  cb.ax.tick_params(labelsize=20)
  fig.savefig(f'/content/drive/My Drive/IC_CAM/IMAGENS/CAM3/{ano}/r{ano_2digitos}{mes_savefig}/ensemble/solar_flux_anom_ensemble_{mes_lead}_{ano_lead_4digitos}.png', bbox_inches='tight')
  
  ######## PLOT DA PRECIPITAÇÃO ########

  fig, ax = plt.subplots( nrows=2,ncols=3,subplot_kw=dict(projection = ccrs.PlateCarree()), figsize=(32,16))
  plt.suptitle(f"LAMMOC - Anomalia de Precipitação (mm) - {data_lead_pendulum.format('MMMM/YYYY', locale='pt-br')} (Cond.Inicial:{data_pendulum.format('MMMM/YYYY', locale='pt-br')})", fontsize=24, fontweight='bold')

  chuva(ax[0,0], lead_escolhido, clima, 4)
  chuva(ax[0,1], lead_escolhido, clima, 5)
  chuva(ax[0,2], lead_escolhido, clima, 6)
  chuva(ax[1,0], lead_escolhido, clima, 7)
  chuva(ax[1,1], lead_escolhido, clima, 8)
  f = chuva(ax[1,2], lead_escolhido, clima, 9)
  cb = fig.colorbar(f, ax=ax.ravel().tolist(), ticks=[-300,-200,-100,-50,-25,-10,0,10,25,50,100,200,300])
  cb.ax.tick_params(labelsize=20)
  fig.savefig(f'/content/drive/My Drive/IC_CAM/IMAGENS/CAM3/{ano}/r{ano_2digitos}{mes_savefig}/precip_anom_membros_{mes_lead}_{ano_lead_4digitos}.png', bbox_inches='tight')

  fig, ax = plt.subplots( nrows=1,ncols=1,subplot_kw=dict(projection = ccrs.PlateCarree()), figsize=(32,16))
  plt.suptitle(f"LAMMOC - Anomalia de Precipitação (mm) - \n{data_lead_pendulum.format('MMMM/YYYY', locale='pt-br')} (Cond.Inicial:{data_pendulum.format('MMMM/YYYY', locale='pt-br')})", fontsize=24, fontweight='bold', x= 0.6)

  p = chuva(ax, lead_escolhido, clima, 'ensemble')
  cb = fig.colorbar(p, ax=ax, ticks=[-300,-200,-100,-50,-25,-10,0,10,25,50,100,200,300])
  cb.ax.tick_params(labelsize=20)
  fig.savefig(f'/content/drive/My Drive/IC_CAM/IMAGENS/CAM3/{ano}/r{ano_2digitos}{mes_savefig}/ensemble/precip_anom_ensemble_{mes_lead}_{ano_lead_4digitos}.png', bbox_inches='tight')

count_time2 = datetime.now()                            #horário de término da rodada    #ident0
tempo_de_rodada = count_time2 - count_time              # diferença entre horario de inicio e termino...   #ident0
                                                        #(a diferença soluciona o problema quanto ao datetime não ser horario de brasilia)  #ident0
print(f'O tempo de rodada foi:    {tempo_de_rodada}')   # imprime na tela o tempo de rodada, antes das imagens   #ident0
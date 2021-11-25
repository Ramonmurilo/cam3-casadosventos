


cmapb = plt.cm.RdYlBu
ncores = cmapb.N

lista_cores = [cmapb(i) for i in range(ncores)]

for l in range (145,170): # azul --claro do 50 ao 100
  lista_cores[l] = (0.7072664359861593, 0.868973471741638, 0.9241061130334487, 1.0)
for l in range (140,145): # azul -claro do 25 ao 50
  lista_cores[l] = (0.821376393694733, 0.9249519415609382, 0.956401384083045, 1.0)
for m in range (135,141):  #azul claro do 10 ao 25
  lista_cores[m] = (0.9022683583237219, 0.962168396770473, 0.9287197231833906, 1.0)
for n in range (125, 135):  # branco do -10 ao 10
  lista_cores[n] = (0.0, 0.0, 0.0, 0.0)

levelsb = [-300,-200,-100,-50,-25,-10,0,10,25,50,100,200,300]
cmapb = mpl.colors.LinearSegmentedColormap.from_list('lammoc', lista_cores, ncores)
#normb = mpl.colors.BoundaryNorm(levelsb, ncores)
normb = mpl.colors.Normalize(vmin=levelsb[0], vmax=levelsb[-1])

######################################################################################
#Teste colobar "c" invertido psl

cmapc = plt.cm.RdYlBu_r
ncores = cmapc.N

lista_cores = [cmapc(i) for i in range(ncores)]
'''
for l in range (145,170): # azul --claro do 50 ao 100
  lista_cores[l] = (0.7072664359861593, 0.868973471741638, 0.9241061130334487, 1.0)
for l in range (140,145): # azul -claro do 25 ao 50
  lista_cores[l] = (0.821376393694733, 0.9249519415609382, 0.956401384083045, 1.0)
for m in range (135,141):  #azul claro do 10 ao 25
  lista_cores[m] = (0.9022683583237219, 0.962168396770473, 0.9287197231833906, 1.0)
  '''
for n in range (102, 153):  # branco do -10 ao 10
  lista_cores[n] = (0.0, 0.0, 0.0, 0.0)

levelsc = [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
cmapc = mpl.colors.LinearSegmentedColormap.from_list('lammoc', lista_cores, ncores)
#normc = mpl.colors.BoundaryNorm(levelsc, ncores)
normc = mpl.colors.Normalize(vmin=levelsc[0], vmax=levelsc[-1])

######################################################################################
#Teste colobar "d" invertido fluxo

cmapd = plt.cm.RdYlBu_r
ncores = cmapd.N

lista_cores = [cmapd(i) for i in range(ncores)]
'''
for l in range (145,170): # azul --claro do 50 ao 100
  lista_cores[l] = (0.7072664359861593, 0.868973471741638, 0.9241061130334487, 1.0)
for l in range (140,145): # azul -claro do 25 ao 50
  lista_cores[l] = (0.821376393694733, 0.9249519415609382, 0.956401384083045, 1.0)
for m in range (135,141):  #azul claro do 10 ao 25
  lista_cores[m] = (0.9022683583237219, 0.962168396770473, 0.9287197231833906, 1.0)
  '''
for n in range (120, 140):  # branco do -10 ao 10
  lista_cores[n] = (0.0, 0.0, 0.0, 0.0)

levelsd = [-80,-65,-50,-35,-20,-5, 5,20,35,50,65,80]
cmapd = mpl.colors.LinearSegmentedColormap.from_list('lammoc', lista_cores, ncores)
#normc = mpl.colors.BoundaryNorm(levelsd, ncores)
normd = mpl.colors.Normalize(vmin=levelsd[0], vmax=levelsd[-1])

######################################################################################
#Teste colobar "e" invertido temp

cmape = plt.cm.RdYlBu_r
ncores = cmape.N

lista_cores = [cmape(i) for i in range(ncores)]
'''
for l in range (145,170): # azul --claro do 50 ao 100
  lista_cores[l] = (0.7072664359861593, 0.868973471741638, 0.9241061130334487, 1.0)
for l in range (140,145): # azul -claro do 25 ao 50
  lista_cores[l] = (0.821376393694733, 0.9249519415609382, 0.956401384083045, 1.0)
for m in range (135,141):  #azul claro do 10 ao 25
  lista_cores[m] = (0.9022683583237219, 0.962168396770473, 0.9287197231833906, 1.0)
  '''
for n in range (102, 153):  # branco do -10 ao 10
  lista_cores[n] = (0.0, 0.0, 0.0, 0.0)

levelse = [-6,-5,-4,-3,-2,-1,1,2,3,4,5,6]
cmape= mpl.colors.LinearSegmentedColormap.from_list('lammoc', lista_cores, ncores)
#normc = mpl.colors.BoundaryNorm(levelse, ncores)
norme= mpl.colors.Normalize(vmin=levelse[0], vmax=levelse[-1])
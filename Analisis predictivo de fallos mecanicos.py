#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # CARGA DE DATOS
# 
# INFORMACION DEL DATASET
# 
# https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification
# 
# 1. Agrupe las maquinas por tipo y por tipo de fallo
# 2. Listado de las maquinas, ordenandolas por la cantidad de rotaciones por minuto(rpm)
# 3. Halle la cantidad de maquinas con fallas que tienen una temperatura en proceso mayor a la indicada por el usuario
# 4. Halle la cantidad de maquinas que presentan algun tipo de fallo
# 5. Halle el promedio de la temperatura del aire en las maquinas que tienen fallas por disipacion de calor
# 
# De manera grafica prsente los siguientes reportes
# 1. Temperatura en proceso promedio, minima y maxima por cada tipo de maquina
# 2. Porcentaje mde maquinas por tipo
# 3. Porcentaje de maquinas por fallos 
# 4. Correlacion entre las caracteriasticas de las maquinas 
# 

# In[2]:


#  PRESENTACION DE LOS DATOS

df = pd.read_csv("predictive_maintenance.csv")
df


# In[5]:


#1. AGRUPACION DE MAQUINAS POR TIPO Y TIPO DE FALLOS

df.groupby(['Type','Failure_Type']).nunique()


# In[6]:


#2. LISTADO DE MAQUINAS, ORDENADAS POR LA CANTIDAD DE ROTACIONES POR MINUTO RPM

 #ESTA ORGANIZADA DE MAYOR RPM A MENOR RPM

df[["Product_ID","Rotational_speed_[rpm]"]].sort_values("Rotational_speed_[rpm]",ascending =False)


# In[12]:


#3. CANTIDAD DE MAQUINAS CON FALLAS QUE TIENEN UNA TEMPERATURA EN PROCESO MAYOR A LA INDICADA POR EL USUARIO

temperatura_indicada = float(input('Ingrese la temperatura en K >>'))
df[['Product_ID']][(df['Process_temperature_[K]']>temperatura_indicada)&(df['Failure_Type'] != 'No Failure')]


# In[31]:


#4. CANTIDAD DE MAQUINAS POR TIPO Y CONTEO DE MAQUINAS QUE PRESENTAN FALLOS

df[df['Failure_Type'] != 'No Failure'].groupby(['Type','Failure_Type'])['UDI'].count().sum(level=0)


# In[18]:


#5. SE HALLO EL PROMEDIO DE LA TEMPERATURA DEL AIRE EN LAS MAQUINAS QUE TIENEN FALLO POR DISIPACION DE CALOR

df[(df['Failure_Type'] == 'Heat Dissipation Failure')]['Air_temperature_[K]'].mean()


# # REPORTES GRAFICOS  

# In[68]:


#TEMPERATURA EN PROCESO PROMEDIO, MINNIMA Y MAXIMA POR CADA TIPO DE MAQUINA
df.groupby('Type')['Process_temperature_[K]'].agg([np.mean, np.min, np.max])


# In[84]:


plt.rcParams["figure.figsize"] = (15, 15)
df.groupby('Type')['Process_temperature_[K]'].agg([np.mean, np.min, np.max]).plot.pie(explode = [0.03, 0.03, 0.03], colors = ['dodgerblue', 'yellow', 'orange'], autopct = '%1.3f%%',subplots= True, shadow = False)
plt.title('Temperatura en proceso promedio, minima y maxima por cada tipo de maquina')
plt.show()


# In[54]:


#PORCENTAJE DE MAQUINAS POR TIPO
plt.figure(figsize = (8,8))                              
df['Type'].value_counts().plot.pie(explode = [0.03, 0.03, 0.03], colors = ['dodgerblue', 'yellow', 'green'], autopct = '%1.1f%%', shadow = False)
plt.xlabel(''),plt.ylabel('')
plt.title('Porcentaje de maquinas por tipo')
plt.show()


# In[39]:


#PORCENTAJE DE MAQUINAS POR FALLO
plt.figure(figsize = (10,10))                                          
df['Failure_Type'].value_counts().plot.pie(explode = [0, 0.5, 0.6,0.7,0.8,1], colors = ['dodgerblue', 'yellow', 'brown','red','coral','black'], autopct = '%1.1f%%')
plt.xlabel(''),plt.ylabel('')
plt.title('Porcentaje de maquinas por fallo')
plt.show()


# In[40]:


#MATRIZ DE CORRELACION
df_modificado=df.drop(columns= ['UDI','Product_ID'])
df_modificado.corr()


# In[45]:


plt.figure(figsize=(10,10))
df_correlacion = df_modificado.corr()
sns.heatmap(df_correlacion, cmap = 'Blues', linewidths = 0.3, linecolor = 'dodgerblue', annot = True,
           vmin = -1, vmax = 1, cbar_kws = {'orientation':'vertical'}, square = True, cbar = True)
plt.title('Correlación entre las caracteristicas de las maquinas')
plt.show()


# In[ ]:





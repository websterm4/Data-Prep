Data-Prep
=========
import sys
sys.path.insert(0,'files/python')
import gdal
import numpy as np
import numpy.ma as ma
from raster_mask import *
from glob import glob
import os
from osgeo import ogr,osr
import pylab as plt
import datetime

year = 2009 or 2010
satellite = 'MOD10A1' or 'MYD10A1'

files = np.sort(glob.glob('files/data/MODIS_Snow_Data/%s.A%d*.*hdf'%(satellite,year)))

#for one file - day 1 of 2009

#def function may include all layers and filenames ????????
def get_snow(filename):
    g = gdal.Open(filename)
    subdatasets = g.GetSubDatasets()
    selected_layers = ["Fractional_Snow_Cover", "Snow_Spatial_QA"]
    data = {}
    file_template = 'HDF4_EOS:EOS_GRID:"%s":MOD_Grid_Snow_500m:%s'
    for i,layer in enumerate ( selected_layers ):
	this_file = file_template % ( filename, layer )
	g = gdal.Open( this_file )
	data[layer] = g.ReadAsArray()
    snow = data['Fractional_Snow_Cover']
    qc = data['Snow_Spatial_QA']
    qc = qc & 1
    snowm = np.ma.array ( snow, mask=qc )
    return snowm

import glob

year = 2009
satellite = 'MOD10A1'

files = np.sort(glob.glob('files/data/MODIS_Snow_Data/%s.A%d*.*hdf'%(satellite,year)))
#this can altered to just filter by terra and aqua satellites

#2009: days 238 and 250

snow = []
for f in files:
    snow.append(get_snow(f))

snow = np.ma.array(snow)

#plotting snow cover images for each day
snow_max = np.max(snow)
for i,f in enumerate(files):
    plt.imshow(snow[i],interpolation='none',vmin=0.,vmax=lai_max)
    file_id = f.split('/')[-1].split('.')[-5][1:]
#plt title, colorbar, savefig('files/images/snow_HUC2_%s.jpg'%file_id)


#build cloud mask
snow = data['Fractional_Snow_Cover']
#cloud = (snow == 250) - indicates where areas are cloud, visual purposes
valid_mask = (snow > 100)
plt.imshow(valid_mask)

#defining a loop for cloud mask
def read_snow(filename):
    layer = "Fractional_Snow_Cover"
    file_template = 'HDF4_EOS:EOS_GRID:"%s":MOD_Grid_Snow_500m:%s'
    this_file = file_template % ( filename, layer )
    g = gdal.Open( this_file )
    snow = g.ReadAsArray()
    valid_mask = (snow > 100)
    return ma.array(snow,mask=valid_mask)

snow = ma.array([read_snow(f) for f in files])
print snow.shape


#vector masking
import sys
sys.path.insert(0,'files/python')
from raster_mask import *
g = ogr.Open("files/data/Hydrologic_Units/HUC_Polygons.shp")
layer = g.GetLayer( 0 )
filename = 'files/data/MODIS_Snow_Data/MOD10A1.A2009001.h09v05.005.2009009120443.hdf'
layer = 'Fractional_Snow_Cover'
file_template = 'HDF4_EOS:EOS_GRID:"%s":MOD_Grid_Snow_500m:%s'
fname = file_template % ( filename, layer )

m = raster_mask2(fname,\
                target_vector_file="files/data/Hydrologic_Units/HUC_Polygons.shp",\
                attribute_filter=2)
if HUC2:
    this_file = file_template%(modis_files,'Fractional_Snow_Cover') #layer included for geometry
    mask = raster_mask2(this_file,\
                target_vector_file="files/data/Hydrologic_Units/HUC_Polygons.shp",\
                attribute_filter=2)
    rowpix,colpix = np.where(mask == False)
    mincol,maxcol = min(colpix),max(colpix)
    minrow,maxrow = min(rowpix),max(rowpix)
    ncol = maxcol - mincol + 1
    nrow = maxrow - minrow + 1
    area_mask = mask[minrow:minrow+nrow,mincol:mincol+ncol]
    print area_mask
else:
    mincol = 0
    maxcol = 0
    ncol = None
    nrow = None
#dictionary set up with empty list for data layer

data_field = {'Fractional_Snow_Cover':[]}
snow = {'filename':np.sort(files),\
           'minrow':minrow,'mincol':mincol,\
           'mask':area_mask}
snow.update(data_field)
for f in np.sort(snow['files']):
    this_snow = get_snow('files/data/%s'%f,			\mincol=mincol,ncol=ncol,
			\minrow=minrow,nrow=nrow)
    for layer in data_field.keys():
	new_mask = this_snow[layer].mask | area_mask
	this_snow[layer] = ma.array(this_snow[layer],mask=new_mask)
	snow[layer].append(this_snow[layer])

        return snow


#Temperature Data
import sys
sys.path.insert(0,'files/python')
import gdal
import numpy as np
import numpy.ma as ma
from raster_mask import *
from glob import glob
import glob
import os
from osgeo import ogr,osr
import pylab as plt
import datetime


file = 'files/data/delNorteT.dat'
fp = open(file, 'r')
tdata = fp.readlines()
fp.close()
 #chop off the header lines
required_data = tdata[3290:-1096] 
data = np.loadtxt(required_data,usecols=(0,1,2,3,4),unpack=True,dtype=str) #shape(5,363)
data = data.astype(int)

days = xrange(365)
year = np.empty(len(days))
doy = np.empty(len(days))
for i in days:
    year[i],doy[i] = datetime.datetime(data[0][i],data[1][i],data[2][i]).strftime('%Y %j').split()
a = []
a.append(year)
a.append(doy)
a.append(data[3])
a.append(data[4])
a = np.array(a)
a = a.T
a = a.astype(int)#this bit needs sorting out -erro: setting an array element within a sequence
print a
Temp2009 = a[3290:3655]
Temp2010 = a[3655:-1096]


#TypeError: cannot perform reduce with flexible type; dtype = string then this message occurs
#np.genfromtxt(filename....)

#Can only concatenate list (not str or np)
#.format for strings only
#len(data[first dimension


#Discharge Data
file = 'files/data/delnorte.dat'
fp = open(file, 'r')
data = fp.readlines()
required_data = data[3314:4044] #max lines in doc = 4045, first 35 are header lines
data = np.loadtxt(required_data,usecols=(2,3),unpack=True,dtype=str)
#plt.plot(data[1].astype(float))
data = data.T
#All of the above code taken directly from the course notes
#values needed are the last 730 values
#values from 3323-4018 (add 35 on to first value to account for header lines)

days = xrange(365)
year = np.empty(len(days))
doy = np.empty(len(days))
for i in days:
    ds = np.array(data[0][0].split('-')).astype(int)
    year[i],doy[i] = datetime.datetime(ds[0],ds[1],ds[2]).strftime('%Y %j').split()
d = []
d.append(year)
d.append(doy)
d.append(data[1])
print d
d = np.array(d)
d = d.astype(int)
Dis2009 = d[3314:3697]
Dis2010 = d[3697:4044]
data[  ].mean()


#interesting code
#z = np.sqrt(x**2 + y**2) # array set up how you want it defined
#ninside = len(np.where(z<1.)[0]) # new array showing how many values lie in a partciluar region using np.where

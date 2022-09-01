import numpy as np
import colorsys
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def define_ccolormap(): # custom colormap
    it = 55
    viridis = cm.get_cmap('viridis', 256)
    newcolors = viridis(np.linspace(0, 1, 4*it+1))
    k=0
    a = 4/6
    b = 1/4

    for i,j in [(3/6,1/3),(2/6,1/2),(1/6,1),(1/12,1)]:
        for h,v in zip(np.linspace(a,i,it,endpoint=False),np.linspace(b,j,it,endpoint=False)):
            col = list(colorsys.hsv_to_rgb(h,1.0,v))
            col.append(1)
            newcolors[k, :] = col
            k=k+1
            newcmp = ListedColormap(newcolors)
        a,b=i,j

    col = list(colorsys.hsv_to_rgb(a,1.0,b))
    col.append(1)
    newcolors[k, :] = col
    newcmp = ListedColormap(newcolors)
    return newcmp
    #        4./6.,1.0,0.25, // dark blue
    #        3./6.,1.0,0.33, // greenier (?)
    #        2./6.,1.0,0.5, // green
    #        1./6.,1.0,1.0, // light yellow
    #        1./12.,1.0,1.0 // lightwe(?) yellow

def fp(u,k): #function term
    a,b,c,d=k
    return -5*u**4 -3*a*u**2 -2*b*u -c
def reacterm(x,k):
    a,b,c,d=k
    return -x**5 -a*x**3 -b*x**2 -c*x -d
def potential(x,k): 
    a,b,c,d=k
    return -x**6/6- a*x**4/4- b*x**3/3- c*x**2/2 -d*x

def minima(a,b,c,d):
    coeffs = [-1.,0,-a,-b,-c,-d]
    allroots = np.roots(coeffs)
    realroots = [x.real for x in allroots if np.isreal(x)]
    stableroots = [x for x in realroots if fp(x,(a,b,c,d))<0]
    return np.sort(stableroots)

def mesh_hex(resol):
    res = int(resol*2/np.sqrt(3)),resol
    xx, yy = np.meshgrid(np.arange(res[1]), np.arange(res[0]), sparse=False, indexing='xy')
    yy = yy * np.sqrt(3)/2
    xx = xx*1.0
    xx[::2, :] = (xx[::2, :]+1/2)
    return res, xx, yy

def mesh_tri(resol):
    res = int(resol*2/np.sqrt(3)+1),resol*2//3+1
    xx, yy = np.meshgrid(np.arange(res[1]), np.arange(res[0]), sparse=False, indexing='xy')
    xx=xx*3.0/2
    xx[::2, ::2] = xx[::2, ::2]+1/2
    xx[1::2, 1::2] = xx[1::2, 1::2]+1/2
    yy=yy*np.sqrt(3)/2
    return res, xx, yy

def mesh_reg(resol):
    res = resol, resol
    xx, yy = np.meshgrid(np.arange(res[1]), np.arange(res[0]), sparse=False, indexing='xy')
    return res, xx, yy

def inioct(res,xx,yy,solutions,resol=100):#initial octagonal shape
    r1 = resol*3/8
    r2 = resol*2/8
    r3 = resol/8
    r4 = np.sqrt(2)

    z = (xx+yy)*0.0+solutions[0]
    for i in np.arange(res[0]):
        for j in np.arange(res[1]):
            if (np.abs(xx[i,j]-resol//2)<r3 and np.abs(yy[i,j]-resol//2)<r3) and np.abs(xx[i,j]-resol//2)+np.abs(yy[i,j]-resol//2)<r3*r4:
                z[i,j]= solutions[2]
            elif (np.abs(xx[i,j]-resol//2)<r2 and np.abs(yy[i,j]-resol//2)<r2) and np.abs(xx[i,j]-resol//2)+np.abs(yy[i,j]-resol//2)<r2*r4:
                z[i,j]= solutions[1]
            elif (np.abs(xx[i,j]-resol//2)<r1 and np.abs(yy[i,j]-resol//2)<r1) and np.abs(xx[i,j]-resol//2)+np.abs(yy[i,j]-resol//2)<r1*r4:
                z[i,j]= solutions[0]
            else:
                z[i,j]= solutions[2]
    return z

def inistripe(xx,yy,resol,phi,soli,solj): #initial angular front
    z = (xx+yy)*0.0
    m = np.tan(phi*np.pi/180)
    y = xx*m+(1-m)*resol/2    
    z[y<=yy]= soli
    z[y>yy]= solj
    return z

if __name__ == "__main__":
    pass
else:
    newcmp = define_ccolormap()
	
	
	

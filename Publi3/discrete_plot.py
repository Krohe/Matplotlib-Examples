import numpy as np
import colorsys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from discrete_basics import newcmp

def patches_reg(xx,yy,z1,rad=1):
    squares = [mpatches.RegularPolygon((xi,yi), 4, radius=np.sqrt(2)*0.5*rad, orientation=np.deg2rad(45), linewidth=0) for xi,yi in zip(xx.flatten(),yy.flatten())] #,radius=0.5
    collection = PatchCollection(squares, cmap=newcmp, norm=plt.Normalize(-3.5,3.5), rasterized=True)
    collection.set_array(z1.flatten())
    return collection

def patches_hex(xx,yy,z1,rad=1):
    hexagons = [mpatches.RegularPolygon((xi,yi), 6, radius=np.sqrt(1.1/3)*rad, orientation=np.deg2rad(0), linewidth=0) for xi,yi in zip(xx.flatten(),yy.flatten())]
    collection = PatchCollection(hexagons, cmap=newcmp, norm=plt.Normalize(-3.5,3.5), rasterized=True)
    collection.set_array(z1.flatten())
    return collection

def patches_tri(xx,yy,z1,rad=1):
    temp = np.hstack((xx[::2, ::2].flatten(),xx[1::2, 1::2].flatten())), np.hstack((yy[::2, ::2].flatten(),yy[1::2, 1::2].flatten()))
    triangles = [mpatches.RegularPolygon((xi,yi), 3, radius=1.075*rad, orientation=np.deg2rad(-30), linewidth=0) for xi,yi in zip(*temp)]
    temp = np.hstack((xx[1::2, ::2].flatten(),xx[::2, 1::2].flatten())), np.hstack((yy[1::2, ::2].flatten(),yy[::2, 1::2].flatten()))
    triangles = triangles + [mpatches.RegularPolygon((xi,yi), 3, radius=1.075*rad, orientation=np.deg2rad(30), linewidth=0) for xi,yi in zip(*temp)]
    collection = PatchCollection(triangles, cmap=newcmp, norm=plt.Normalize(-3.5,3.5), rasterized=True)
    collection.set_array(np.hstack((z1[::2, ::2].flatten(),z1[1::2, 1::2].flatten(),z1[1::2, ::2].flatten(),z1[::2, 1::2].flatten())))
    return collection

def plot_reg(xx,yy,z1,name): #standard-plots
    fig, ax = plt.subplots(1,1,figsize = (5,5), dpi=150)    
    ax.add_collection(patches_reg(xx,yy,z1))
    ax.axis('equal');
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(name+".pdf")
    plt.show()
    
def plot_hex(xx,yy,z1,name): #standard-plots
    fig, ax = plt.subplots(1,1,figsize = (5,5), dpi=150)
    ax.add_collection(patches_hex(xx,yy,z1))
    ax.axis('equal');
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(name+".pdf")
    plt.show()

def plot_tri(xx,yy,z1,name): #standard-plots
    fig, ax = plt.subplots(1,1,figsize = (5,5), dpi=150)
    ax.add_collection(patches_tri(xx,yy,z1))
    ax.axis('equal');
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(name+".pdf")
    plt.show() 

if __name__ == "__main__":
    pass
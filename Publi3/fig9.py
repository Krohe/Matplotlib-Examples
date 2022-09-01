import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import ImageGrid
from discrete_basics import *
from discrete_plot import *
#from discrete_sim import *

mpl.rcParams['backend'] = 'pdf'
mpl.rc('font',**{'family':'serif'})
mpl.rcParams['mathtext.fontset'] = 'cm'

def axin_setup(axins):
    axins.set_xlim(10, 50)
    axins.set_ylim(10, 50)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    axins.set_xticks([])
    axins.set_yticks([])
    return None

resolution = 100
a,b,c,d = -10,2,18.75,-5
print(a)
solutions = minima(a,b,c,d)
print(solutions)

# Set up figure and image grid
fig = plt.figure(figsize=(15, 5), dpi=150)

grid = ImageGrid(fig, 111, nrows_ncols=(1,3), axes_pad=0.15,
                 share_all=True, cbar_location="right",
                 cbar_mode="single", cbar_size="7%", cbar_pad=0.1)

res, xv, yv = mesh_hex(resolution)
z = inioct(res,xv,yv,solutions,resolution)
grid[1].add_collection(patches_hex(xv,yv,z))
axins = grid[1].inset_axes([0, 0, 0.5, 0.5])
axins.add_collection(patches_hex(xv,yv,z,0.6))
axin_setup(axins)

res, xv, yv = mesh_tri(resolution)
z = inioct(res,xv,yv,solutions,resolution)
grid[2].add_collection(patches_tri(xv,yv,z))
axins = grid[2].inset_axes([0, 0, 0.5, 0.5])
axins.add_collection(patches_tri(xv,yv,z,0.6))
axin_setup(axins)

res, xv, yv = mesh_reg(resolution)
z = inioct(res,xv,yv,solutions,resolution)
c = grid[0].add_collection(patches_reg(xv,yv,z))
axins = grid[0].inset_axes([0, 0, 0.5, 0.5])
axins.add_collection(patches_reg(xv,yv,z,0.6))
axin_setup(axins)

for ax in grid:
    ax.set_rasterization_zorder(0)
    ax.axis('equal');
    ax.axis('off')

ax.cax.cla()
cb = mpl.colorbar.Colorbar(ax.cax,c)
cb.ax.tick_params(labelsize=15)
cb.set_label(label=r'$u(x,y)$',weight='bold',fontsize=18)
ax.cax.toggle_label(True)

plt.savefig("9_initials.pdf")
plt.show()
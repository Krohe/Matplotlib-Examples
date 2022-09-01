import matplotlib.pyplot as plt
import matplotlib as mpl
from discrete_basics import *
from discrete_plot import *
from discrete_sim import *
from scipy.integrate import solve_ivp
mpl.rcParams['backend'] = 'pdf'
mpl.rc('font',**{'family':'serif'})
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amssymb,amsmath,wasysym}')
plt.rcParams['font.size'] = 10


a, b, c, d = -10, 2, 15, -3
resolution = 100
coup, dx = 5, 1.0
solutions = minima(a,b,c,d)
res, xv, yv = mesh_reg(resolution)
z = inioct(res,xv,yv,solutions,resolution)
diff = coup*dx**2
z1 = z.flatten()
timeplus=0

for tlimit in [(1,0.001),(3,0.01),(5,0.01),(10,0.01)]: #Looping to goal-values to print
    tstep = tlimit[1]
    t = 0
    while (t*tstep+timeplus)<tlimit[0]: #Looping prevents internal issues
        try:
            reso = solve_ivp(rhs_reg,[0,tstep],z1,'LSODA',args=((a,b,c,d,diff,dx,res),))
            z1 = reso.y[:,-1]
            t = t+1
        except(MemoryError): #Workaround against memory issue
            fig, ax = plt.subplots(1,1,figsize = (1,1))
            plt.close()
    plot_reg(xv,yv,z1,"10_reg{}_{}_{}_c{:.2f}_d{}_D{}_dx{}_t{:.3f}s".format(resolution,a,b,c,d,diff,dx,tlimit[0]))
    timeplus = timeplus+t*tstep


a, b, c, d = -10, 2, 18.75, -3
resolution = 100
coup, dx = 5, 1.0
solutions = minima(a,b,c,d)
res, xv, yv = mesh_reg(resolution)
z = inioct(res,xv,yv,solutions,resolution)
diff = coup*dx**2
z1 = z.flatten()
timeplus=0

for tlimit in [(1,0.001),(3,0.01),(5,0.01),(10,0.01)]: #Looping to goal-values to print
    tstep = tlimit[1]
    t = 0
    while (t*tstep+timeplus)<tlimit[0]: #Looping prevents internal issues
        try:
            reso = solve_ivp(rhs_reg,[0,tstep],z1,'LSODA',args=((a,b,c,d,diff,dx,res),))
            z1 = reso.y[:,-1]
            t = t+1
        except(MemoryError): #Workaround against memory issue
            fig, ax = plt.subplots(1,1,figsize = (1,1))
            plt.close()
    plot_reg(xv,yv,z1,"10_reg{}_{}_{}_c{:.2f}_d{}_D{}_dx{}_t{:.3f}s".format(resolution,a,b,c,d,diff,dx,tlimit[0]))
    timeplus = timeplus+t*tstep


a, b, c, d = -10, 2, 15, -3
resolution = 100
coup, dx = 2.5, 1.0
solutions = minima(a,b,c,d)
res, xv, yv = mesh_hex(resolution)
z = inioct(res,xv,yv,solutions,resolution)
diff = coup*dx**2
z1 = z.flatten()
timeplus=0

for tlimit in [(1,0.001),(5,0.01),(10,0.01)]: #Looping to goal-values to print
    tstep = tlimit[1]
    t = 0
    while (t*tstep+timeplus)<tlimit[0]: #Looping prevents internal issues
        try:
            reso = solve_ivp(rhs_hex,[0,tstep],z1,'LSODA',args=((a,b,c,d,diff,dx,res),))
            z1 = reso.y[:,-1]
            t = t+1
        except(MemoryError): #Workaround against memory issue
            fig, ax = plt.subplots(1,1,figsize = (1,1))
            plt.close()
    plot_hex(xv,yv,z1,"11_hex{}_{}_{}_c{:.2f}_d{}_D{}_dx{}_t{:.3f}s".format(resolution,a,b,c,d,diff,dx,tlimit[0]))
    timeplus = timeplus+t*tstep


a, b, c, d = -10, 2, 15, -3
resolution = 100
coup, dx = 5, 1.0
solutions = minima(a,b,c,d)
res, xv, yv = mesh_tri(resolution)
z = inioct(res,xv,yv,solutions,resolution)
diff = coup*dx**2
z1 = z.flatten()
timeplus=0

for tlimit in [(1,0.001),(5,0.01),(10,0.01)]: #Looping to goal-values to print
    tstep = tlimit[1]
    t = 0
    while (t*tstep+timeplus)<tlimit[0]: #Looping prevents internal issues
        try:
            reso = solve_ivp(rhs_tri,[0,tstep],z1,'LSODA',args=((a,b,c,d,diff,dx,res),))
            z1 = reso.y[:,-1]
            t = t+1
        except(MemoryError): #Workaround against memory issue
            fig, ax = plt.subplots(1,1,figsize = (1,1))
            plt.close()
    plot_tri(xv,yv,z1.reshape(res),"11_tri{}_{}_{}_c{:.2f}_d{}_D{}_dx{}_t{:.3f}s".format(resolution,a,b,c,d,diff,dx,tlimit[0]))
    timeplus = timeplus+t*tstep
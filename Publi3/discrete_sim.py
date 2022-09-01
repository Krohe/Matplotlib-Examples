import numpy as np
from numba import jit
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

@jit(nopython=True)
def rhs_reg(t,z,k):
    a,b,c,d,diff,dx,res = k
    m = z.reshape(res)
    dz = np.zeros_like(m)
    i = 0
    j = res[0]-1
    dz[0,0] = m[1,0]+m[0,1]-2*m[0,0]
    dz[0,j] = m[i+1,j]+m[i,j-1]-2*m[0,j]
    dz[j,0] = m[j-1,0]+m[j,1]-2*m[j,0]
    dz[j,j] = m[j-1,j]+m[j,j-1]-2*m[j,j]
    lim1 = 1
    for i in range(lim1,res[0]-lim1):
        for j in range(1,res[1]-1):
            dz[i,j] = m[i+1,j]+m[i-1,j]+m[i,j+1]+m[i,j-1]-4*m[i,j]
        j = 0
        dz[i,j] = m[i+1,j]+m[i-1,j]+m[i,j+1]-3*m[i,j]
        dz[j,i] = m[j+1,i]+m[j,i+1]+m[j,i-1]-3*m[j,i]
        j = res[1]-1
        dz[i,j] = m[i+1,j]+m[i-1,j]+m[i,j-1]-3*m[i,j]
        dz[j,i] = m[j-1,i]+m[j,i+1]+m[j,i-1]-3*m[j,i]

    m = dz*diff/dx**2-m**5 - a*m**3 - b*m**2 - c*m - d
    return m.flatten()

@jit(nopython=True)
def rhs_hex(t,z,k):
    a,b,c,d,diff,dx,res = k
    m = z.reshape(res)
    dz = np.zeros_like(m)
    i = 0
    j1 = res[0]-1
    j2 = res[1]-1
    dz[0,0] = m[0,1]+m[1,0]+(m[1,1])-3*m[0,0]
    dz[0,j2] = m[0,j2-1]+m[1,j2]-2*m[0,j2] 
    dz[j1,0] = m[j1,1]+m[j1-1,0]-2*m[j1,0]
    dz[j1,j2] = m[j1,j2-1]+m[j1-1,j2]+m[j1-1,j2-1]-3*m[j1,j2]

    for i in range(1,res[0]-1):
        for j in range(1,res[1]-1):
            dz[i,j] = m[i,j+1]+m[i,j-1]+m[i+1,j]+m[i-1,j]+(m[i-1,j-1]+m[i+1,j-1])*(i%2)+(m[i+1,j+1]+m[i-1,j+1])*((i+1)%2)-6*m[i,j]
        j = 0
        dz[i,j] = m[i,j+1]+m[i+1,j]+m[i-1,j]+(m[i+1,j+1]+m[i-1,j+1]-2*m[i,j])*((i+1)%2)-3*m[i,j]
        j = res[1]-1
        dz[i,j] = m[i,j-1]+m[i+1,j]+m[i-1,j]+  (m[i-1,j-1]+m[i+1,j-1]-2*m[i,j])*(i%2)-3*m[i,j]

    for j in range(1,res[1]-1):
        dz[0,j] = m[0,j+1]+m[0,j-1]+m[1,j]+m[1,j+1]-4*m[0,j]
        i = res[0]-1
        dz[i,j] = m[i,j+1]+m[i,j-1]+m[i-1,j]+(m[i-1,j-1])*(i%2)+(m[i-1,j+1])*((i+1)%2)-4*m[i,j]

        
    m = dz*diff*2/(3*dx**2)-m**5 - a*m**3 - b*m**2 - c*m - d
    return m.flatten()

@jit(nopython=True)
def rhs_tri(t,z,k):
    a,b,c,d,diff,dx,res = k
    m = z.reshape(res)
    dz = np.zeros_like(m)
    i = 0
    j1 = res[0]-1
    j2 = res[1]-1

    dz[0,0] = m[1,0]+m[0,1] -2*m[0,0]
    dz[0,j2] = m[1,j2]  +(j2%2)*((m[0,j2-1]-m[0,j2])) -m[0,j2]
    dz[j1,0] = m[j1-1,0]  +((j1+1)%2)*(m[j1,1]-m[j1,0]) -m[j1,0]
    dz[j1,j2] = m[j1-1,j2]  +((j2+1)%2)*((j1%2)*(m[j1,j2-1]-m[j1,j2])) +(j2%2)*(((j1+1)%2)*(m[j1,j2-1]-m[j1,j2])) -m[j1,j2]

    for i in range(1,res[0]-1):
        for j in range(1,res[1]-1):
            dz[i,j] = m[i+1,j]+m[i-1,j]  +((j+1)%2)*((i%2)*(m[i,j-1])+((i+1)%2)*(m[i,j+1])) +(j%2)*(((i+1)%2)*(m[i,j-1])+(i%2)*(m[i,j+1])) -3*m[i,j]
        j = 0
        dz[i,j] = m[i+1,j]+m[i-1,j]  +((i+1)%2)*(m[i,j+1]-m[i,j]) -2*m[i,j]
        j = res[1]-1
        dz[i,j] = m[i+1,j]+m[i-1,j]  +((j+1)%2)*(i%2)*(m[i,j-1]-m[i,j]) +(j%2)*((i+1)%2)*(m[i,j-1]-m[i,j]) -2*m[i,j]
    for j in range(1,res[1]-1):
        i = 0
        dz[i,j] = m[i+1,j]  +((j+1)%2)*((m[i,j+1])) +(j%2)*((+m[i,j-1])) -2*m[i,j]
        i = res[0]-1
        dz[i,j] = m[i-1,j]  +((j+1)%2)*((i%2)*(+m[i,j-1])+((i+1)%2)*(+m[i,j+1])) +(j%2)*(((i+1)%2)*(+m[i,j-1])+(i%2)*(+m[i,j+1])) -2*m[i,j]
        
    m = dz*diff*4/(3*dx**2) -m**5 - a*m**3 - b*m**2 - c*m - d
    return m.flatten()

def vecfit(dev,xx,yy,phi,resol):
    a = np.array([resol/2+dev[0]*np.cos(np.deg2rad(phi+90)), resol/2+dev[0]*np.sin(np.deg2rad(phi+90))]) #point on fitline, dev=0==ini
    b = np.array([np.cos(np.deg2rad(phi)),np.sin(np.deg2rad(phi))]) #directional vector
    p = np.column_stack((xx, yy)) #points in 'front'
    diff =np.cross(p-a,b)/np.linalg.norm(b) #distance between the fitline a+b and the front data
    return np.sum(np.abs(diff))

def vecfitter(xx,yy,phi,resol=100):
    res = minimize(vecfit, 0 ,args=(xx,yy,phi,resol), method='SLSQP',bounds=[((-resol*0.45,resol*0.45))])
    return (res.fun, res.x[0])


if __name__ == "__main__":
    pass
	
	
	

from scipy import special
from numba import jit
import numpy as np

eps_0 = 8.854187812813e-12
mu_0 = 1.25666370621219e-6

def cart_cyn(x,y,z):
    r= np.sqrt(np.power(x,2)+np.power(y,2))
    theta = np.arctan2(y,x)
    
    return r,theta,z

def cyn_cart(r,theta,z):
    x = r*np.cos(theta)
    y = r*np.sin(theta)

    return x,y,z

def occilating_field(a:np.ndarray,omega:float,phase:float=np.pi/2):
    '''return a lambda function describing how a field occilated over time
    '''
    return lambda t:a*np.sin((omega*t)+phase)

def constant_field(a:np.ndarray):
    '''return a lambda function decribing a constant field

    This exists for completeness and for use with RK4
    '''
    return lambda t:a

def mag_bottle(x,y,z,z0,a = 1,i0 = 1000) -> np.ndarray:
    
    '''Function for determining the magnetic field produced by two parallel current loops in close proximity to each other 
      resulting in the formation of a magnetic bottle
      
      This function takes the position of a particle between two current loops as arguements and returns the overall magnetic field 
      produced by the two current loops at that given point in space in x,y and z directions'''
    
    #radius of current loops
    u_0 = 1.257E-6                        #Permeability of free space
    #current of both loops (has to be the same to produce a magnetic bottle)
    #positions of the two current loops in the z plane

    theta = np.arctan2(y,x)
    r = np.sqrt((x**2 + y**2))
      
    K_c =  np.sqrt((4*a*(r))/((r +a)**2 + (z-z0)**2))

    A = (u_0*i0)/(2*np.pi)

    B = (z-z0)/np.sqrt(r*(((r + a)**2) + ((z-z0)**2)))    #converting B1 to cartesian for x and y components
    C = -special.ellipk(K_c) + ((((r**2) + (a**2) + ((z-z0)**2))/(((r-a)**2) + ((z-z0)**2)))*special.ellipe(K_c))

    D = 1/(np.sqrt(((r + a)**2)+((z-z0)**2)))
    E = special.ellipk(K_c) - ((((r**2)-(a**2) + ((z-z0)**2))/(((r-a)**2) + ((z-z0)**2)))*special.ellipe(K_c))

    bfield_r = A*B*C                  #eqn B1 from Schill2003 in cartesian
    bfield_z = A*D*E                  #eqn B2 from Schill2003

    bfield_x = bfield_r*np.cos(theta)
    bfield_y = bfield_r*np.sin(theta)

    b_loop = np.array([bfield_x,bfield_y,bfield_z])    #the magnetic field components produced by each loop

    #b_loop[1] *= -1
      
    b_bot = np.sum(b_loop, axis=1)            #the combined magnetic field components from each loop
 
    return b_bot

def electric_field(phi):
    e_field = np.gradient(phi)
    return e_field

@jit
def electric_potential(rho,*,tol=1e-3,max_iter=1e9):
    # (phi[i-1][j][k]-2*phi[i][j][k]+phi[i+1][j][k])/(dx*dx) + 
    # (phi[i][j-1][k]-2*phi[i][j][k]+phi[i][j+1][k])/(dy*dy) + 
    # (phi[i][j][k-1]-2*phi[i][j][k]+phi[i][j][k-1])/(dz*dz) = -rho[i][j]/eps0

    phi = (-rho/eps_0)*1e-10
    
    rel_diff = tol*100
    space = phi == 0
    ran = range(phi.shape[0]-1)
    for n in range(int(max_iter)):
        new_phi = phi.copy()
        for i in ran:
            for j in ran:
                for k in ran:
                    if space[i,j,k]==True:
                        new_phi[i,j,k] = (phi[i+1,j,k]+phi[i-1,j,k]+phi[i,j+1,k]+phi[i,j-1,k]+phi[i,j,k+1]+phi[i,j,k-1])*(1/6)
        rel_diff=np.nanmean((new_phi-phi)/phi)
        phi=new_phi

        if rel_diff > tol:
            break

    return phi

def charge_density(parti,xx,yy,zz,ds):
    #add in_region to only consider particles in the given region
    minimum = np.min(xx)
    max_inx = len(xx)
    rho = np.zeros_like(xx*yy*zz)
    pos = parti[:,0:3]
    q = parti[:,7]
    ref_node = np.floor((pos-minimum)/ds).astype('int')
    

    x_in = (ref_node[:,0]<max_inx)&(ref_node[:,0]>=0)
    y_in = (ref_node[:,1]<max_inx)&(ref_node[:,1]>=0)
    z_in = (ref_node[:,2]<max_inx)&(ref_node[:,2]>=0)

    out = ~x_in
    
    x = xx[ref_node[x_in,0],ref_node[y_in,1],ref_node[z_in,2]]
    y = yy[ref_node[x_in,0],ref_node[y_in,1],ref_node[z_in,2]]
    z = zz[ref_node[x_in,0],ref_node[y_in,1],ref_node[z_in,2]]

    parti[out,0:6] = [0.0,0.0,0.0,0.0,0.0,0.0]

    #principle volume. The volume between the refernce node and the particle
    dv = np.array([pos[x_in,0]-x[:],pos[y_in,1]-y[:],pos[z_in,2]-z[:]]).T

    ran = range(2)
    for i in ran:
        for j in ran:
            for k in ran:
                offset = np.array([i,j,k])
                node = ref_node+offset
                rho[node[:,0],node[:,1],node[:,2]] += np.product(np.absolute(offset-dv),axis=1)*q

    return rho

def create_region(min,max,n):
    '''creates a square region in which calculations are preformed
    '''
    x,ds = np.linspace(min,max,n,retstep=True)
    y = np.linspace(min,max,n)
    z = np.linspace(min,max,n)

    xx,yy,zz = np.meshgrid(x,y,z,indexing='ij')

    return xx,yy,zz,ds

def parti_to_electric(parti,min,max,n,*,tol=1e-3,max_iter=1e9):
    xx,yy,zz,ds = create_region(min,max,n)
    rho = charge_density(parti,xx,yy,zz,ds)
    phi = electric_potential(rho,tol=1e-3,max_iter=1e9)
    e_field = electric_field(phi)

    return e_field

def electric_to_parti(parti,e,upper,lower,n):
    
    xx,yy,zz,ds = create_region(lower,upper,n)
    ex,ey,ez = e
    minimum = np.min(xx)
    max_inx = len(xx)
    pos = parti[:,0:3]
    ref_node = np.floor((pos-minimum)/ds).astype('int')
    
    try:
        x = xx[ref_node[:,0],ref_node[:,1],ref_node[:,2]]
        y = yy[ref_node[:,0],ref_node[:,1],ref_node[:,2]]
        z = zz[ref_node[:,0],ref_node[:,1],ref_node[:,2]]
    except IndexError:
        x_in = ref_node[:,0]<max_inx&ref_node[:,0]<0
        y_in = ref_node[:,1]<max_inx&ref_node[:,1]<0
        z_in = ref_node[:,2]<max_inx&ref_node[:,2]<0
        
        x = xx[ref_node[x_in,0],ref_node[y_in,1],ref_node[z_in,2]]
        y = yy[ref_node[x_in,0],ref_node[y_in,1],ref_node[z_in,2]]
        z = zz[ref_node[x_in,0],ref_node[y_in,1],ref_node[z_in,2]]

    dv = np.array([pos[:,0]-x[:],pos[:,1]-y[:],pos[:,2]-z[:]]).T

    v_tot = ds**3

    total_x = 0
    total_y = 0
    total_z = 0

    ran = range(2)
    for i in ran:
        for j in ran:
            for k in ran:
                offset = np.array([i,j,k])
                node = ref_node+offset

                weight = ((offset-dv)/v_tot).T

                total_x += weight[0]*ex[node[:,0],node[:,1],node[:,2]]
                total_y += weight[1]*ey[node[:,0],node[:,1],node[:,2]]
                total_z += weight[2]*ez[node[:,0],node[:,1],node[:,2]]
    
    e_x = total_x/8
    e_y = total_y/8
    e_z = total_z/8

    parti_e = np.array([e_x,e_y,e_z]).T

    return parti_e
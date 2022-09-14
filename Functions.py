import numpy as np
from scipy import special
import matplotlib.pyplot as plt


def B_bot(x,y,z) -> np.ndarray:
    
    '''Function for determining the magnetic field produced by two parallel current loops in close proximity to each other 
      resulting in the formation of a magnetic bottle
      
      This function takes the position of a particle between two current loops as arguements and returns the overall magnetic field 
      produced by the two current loops at that given point in space in x,y and z directions'''
    
    
    u_0 = 1.257E-6                      #Permeability of free space
    a = 10                               #radius of current loops
    I_0 = 3                             #current of both loops (has to be the same magnitude to produce a magnetic bottle)           
    z0 = np.array([-15,15])             #planes of the two current loops
    theta = np.arctan2(y,x)
    r = np.sqrt((x**2 + y**2))
   
    K_c =  np.sqrt((4*a*r)/((r +a)**2 + (z-z0)**2))
    A = (u_0*I_0)/(2*np.pi)
    Bx = (z-z0)/(np.sqrt((r*(r + a)**2 + (z-z0)**2)))*np.cos(theta)    #converting B1 to cartesian for x and y components
    By = (z-z0)/(np.sqrt((r*(r + a)**2 + (z-z0)**2)))*np.sin(theta)
    C = -special.ellipk(K_c) + (special.ellipe(K_c)*(r**2 + a**2 + (z-z0)**2)/((r-a)**2 + (z-z0)**2))                  
    D = 1/(np.sqrt(((r + a)**2+(z-z0)**2)))
    E = special.ellipk(K_c) - (((r**2-a**2 + (z-z0)**2)/((r-a)**2 + (z-z0)**2))*special.ellipe(K_c)) 
    bfield_x = A*Bx*C                  #eqn B1 from Schill2003 in cartesian
    bfield_y = A*By*C
    bfield_z = A*D*E                   #eqn B2 from Schill2003
    b_loop = np.array([bfield_x,bfield_y,bfield_z])      #the magnetic field components produced by each loop 
      
    b_bot = np.sum(b_loop, axis=1)            #the combined magnetic field components from each loop

      
    
    return b_bot




def boris(x0,y0,z0,) -> np.ndarray: 
    'function used for boris method implementatation which can also be made to simplify down to the euler method'
    
    
    #intial properties
    #can have these as arguements instead
    E = np.array([0,0,0])  #x,y,z
    B = np.array([0,1,0])
    m = 1 # 9.11E-31
    q = 1   #1.60E-19
    dt = 0.005
    
    #time duration
    time = np.arange(0,30,dt)
    x,y,z = np.zeros(3)     #line needed to begin iteration else x,y,z are unbound
    v_intial = np.array([1,0,0]) #v**n  in x,y and z directions
    

    #iterating over the full time duration
    for t in time:
        
        #calulating the new velocity via the boris method  
        
        v_minus = v_intial + (q*E/m)*dt/2    #from eqn 1 particleincell.com
        t1 = (q*B/m)*dt/2
        v_prime = v_minus + np.cross(v_minus,t1)  #eqn 3 particleincell.com 
        s = 2*t1/(1 + (np.linalg.norm(t1))**2)
        v_plus = v_minus + np.cross(v_prime,s)   #eqn 4 particleincell.com
        v_final = v_plus + (q*E/m)*dt/2          #eqn 2 particleincell.com
        v_intial = v_final        #v**n  in x,y and z directions
        
        x += v_intial[0]*dt
        y += v_intial[1]*dt
        z += v_intial[2]*dt   
                    
        #arrays of updated positions
        x = np.append(x0,x)
        y = np.append(y0,y)
        z = np.append(z0,z)
        
    
    ax = plt.subplot(111,projection = '3d')
    #a = plt.plot()
    ax = ax.plot(x,y,z)
    #a = plt.plot(x,y)   
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()
    
    return(x,y,z)

#initial positions
x0 = np.array([0])      
y0 = np.array([0])
z0 = np.array([0])


def euler(x0,y0,z0):
    "Function for showing the motion of a charged particle in an electric and magnetic filed field"
    
    #intial properties
    #can have these as arguements instead
   
    B = np.array([0,1,0])
    E = np.array([0,0,0])
    m = 1 #9.11E-31
    q = 1  #1.60E-19
    dt = 0.05

    
    #time duration
    time = np.arange(0,30,dt)
    #E = E0*np.sin(w*time)
    
    x,y,z = np.zeros(3)  #line needed to begin iteration else x,y,z are unbound
    
    v_intial = np.array([1,0,0]) #v**n  in x,y and z directions
    
    for t in time:

        #calulate the force on the particle
        F = q*(E + np.cross(v_intial,B))

        #from this force calulate the accleration via newtons second law 
        a = F/m

        #calulate the particles velocity based off this accleration
        v_intial = v_intial + a*dt
    
       
        #calulate the new postion of the particle 
        x += v_intial[0]*dt
        y += v_intial[1]*dt
        z += v_intial[2]*dt


        #arrays of updated positions
        x = np.append(x0,x)
        y = np.append(y0,y)
        z = np.append(z0,z)
        
   
    fig = plt.figure()
    ax = plt.subplot(111,projection = '3d')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    
    return(x,y,z)

#initial positions
x0 = np.array([0])      
y0 = np.array([0])
z0 = np.array([0])

A = euler(x0,y0,z0)
print(A)
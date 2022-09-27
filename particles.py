import numpy as np

def motion_euler(e_field:np.ndarray,m_field:np.ndarray,parti:np.ndarray,dt:float,/,walls:list=[0,0,0],*,collisional:bool=False,thresh:float=0.5) -> np.ndarray:
    '''Move a charged paricle in repose to an electric field using the Euler Method

    Takes in an instantanious electric field (e_field) and a list of charged particles (parti) and calculates their motion for a small time step.

    e_field: A vector describing an electric field or a matrix describing the field strength calculated from a potential
    m_field: A vector describing a magnetic field
    parti: A list of particles in the electric field, contains infomation about current position, velocity, charge and mass
    dt: The time step by which the simulation has been advanced

    OPTIONAL
    walls: The edges of the particles' enviroment. If zero, that dimention is unbound.
    '''

    #Calculating the force acting on the particle due to the electric field and magnetic field
    f = parti[:,7,np.newaxis]*(e_field+np.cross(parti[:,3:6],m_field))

    #Calculating the acceleration on the particle due to that force
    a = f/parti[:,8,np.newaxis]
    #Determining the particle's new velocity vector
    parti[:,3:6] += a*dt
    #Determining the particle's new position vector
    parti[:,0:3] += parti[:,3:6]*dt

    #calling collisions to handle any colliding particles
    if collisional == True:
        parti = collisions(parti,thresh)
    
    return parti

def motion_range_kutta(parti:np.ndarray,dt:float,t:float,func_E:callable,func_B:callable,*,collisional:bool=False,thresh:float=0.5) -> np.ndarray:
    '''Move a charged particle in an electric field and magnetic field

    Takes in an instantanious electric field (e_field) and a list of charged particles (parti) and calculates their motion for a small time step.

    parti: A list of particles in the electric field, contains infomation about current position, velocity, charge and mass
    dt: The time step by which the simulation has been advanced
    t: The current time of the system
    func_E: The function that changes the e-field over time
    func_B: The function that changes the m-field over time
    '''

    #Calculating the acceleration on the particle due to the electric field
    a_init = (func_E(t)+np.cross(parti[:,3:6],func_B(t)))*parti[:,7,np.newaxis]/parti[:,8,np.newaxis]
    a_mid = (func_E(t+(dt/2))+np.cross(parti[:,3:6],func_B(t+(dt/2))))*parti[:,7,np.newaxis]/parti[:,8,np.newaxis]
    a_fin = (func_E(t+dt)+np.cross(parti[:,3:6],func_B(t+dt)))*parti[:,7,np.newaxis]/parti[:,8,np.newaxis]

    #calculate 'k' values for the new velocity
    k1_v = a_init*dt
    k2_v = (a_mid+((k1_v*dt)/2))*dt
    k3_v = (a_mid+((k2_v*dt)/2))*dt
    k4_v = (a_fin+((k3_v*dt)))*dt

    #calculate the new velocity
    parti[:,3:6] += ((k1_v/6)+(k2_v/3)+(k3_v/3)+(k4_v/6))#+dt**5

    #calculate the 'k values for the position
    k1_y = parti[:,3:6]*dt
    k2_y = (parti[:,3:6]+((k1_y*dt)/2))*dt
    k3_y = (parti[:,3:6]+((k2_y*dt)/2))*dt
    k4_y = (parti[:,3:6]+((k3_y*dt)))*dt

    #calculate the new positon
    parti[:,0:3] += ((k1_y/6)+(k2_y/3)+(k3_y/3)+(k4_y/6))#+dt**5

    #calling collisions to handle any colliding particles
    if collisional == True:
        parti = collisions(parti,thresh)

    return parti

def motion_boris(parti:np.ndarray,dt:float,e_field:np.ndarray,m_field:np.ndarray,*,collisional:bool=False,thresh:float=0.5):

    #calculate v minus
    v_minus = parti[:,3:6]+((parti[:,7,np.newaxis]/parti[:,8,np.newaxis])*e_field*(dt/2))

    #preform the rotation and find v plus
    t = (parti[:,7,np.newaxis]/parti[:,8,np.newaxis])*m_field*(dt/2)
    v_prime = v_minus + np.cross(v_minus,t)
    s = (2*t)/(1+t**2)
    v_plus = v_minus + np.cross(v_prime,s)

    #find the final velocity
    parti[:,3:6] = v_plus+((parti[:,7,np.newaxis]/parti[:,8,np.newaxis])*e_field*(dt/2))

    #finc the final position
    parti[:,0:3] += parti[:,3:6]*dt

    #calling collisions to handle any colliding particles
    if collisional == True:
        parti = collisions(parti,thresh)

    return parti

def create_particles(n:int,w:float|tuple,h:float|tuple,/,d:float|tuple=(0,0),q:list=[-1],m:list=[1.0],r:list=[1.0],*,v_max=0,v=None,exact_pos=False) -> np.ndarray:
    '''Creates an array of particles.

    Generated n number of charged particles with a random position and initial velocity

    n: The number of particles to generate
    w: width of the simulation/area that particles are allowed to generate in
    h: height of the simulation/area that particles are allowed to generate in

    OPTIONAL
    d: depth of the simulation/area that particles are allowed to generate in. Defaults to 0 (a 2D simulation)
    q: charge on the particles. Follows the distribution defined by the array. Defaults to all having q = -1
    m: mass of the particles. Follows the distribution defined by the array. Defaults to all having m =  1
    r: raduis of the particles. Follows the distribution defined by the array. Defaults to all having r =  1

    v_max: the maximum initial velocity. If int, specifies max velocity for all components. If list, specifies max velocity for each component
    '''
    #set up the random number generator
    rng = np.random.default_rng()

    #determine v_max's type and handle accordingly
    if isinstance(v_max,int):
        v_max = np.array([v_max,v_max,v_max])
    elif isinstance(v_max,list):
        if len(v_max) == 2:
            v_max.append(0)
        v_max = np.array(v_max)
    else:
        raise TypeError

    #define an empty list of particles
    particles = []
    #iterate for as many particles as n
    for i in range(n):
        #place the particle
        if isinstance(w,tuple) and isinstance(h,tuple) and isinstance(d,tuple):
            #create a random position for the particle from min -> max
            x,y,z = (rng.random()*(w[1]-w[0]))+w[0], (rng.random()*(h[1]-h[0]))+h[0], (rng.random()*(d[1]-d[0]))+d[0]
        elif isinstance(w,float) and isinstance(h,float) and isinstance(d,float):
            if exact_pos:
                x,y,z = w,h,d
            else:
                #create a random position for the particle assuming 0 -> x
                x,y,z = (rng.random()*w), (rng.random()*h), (rng.random()*d)
        else:
            raise TypeError

        if v:
            #set the velocity componets to given in v
            vx,vy,vz = v
        else:
            #create a random velocity for the particle
            vx,vy,vz = rng.random()*v_max

        #get the new particles values for r,q and m
        parti_r = r[i%len(r)]
        parti_q = q[i%len(q)]
        parti_m = m[i%len(m)]

        #add the new particle to the list of particles
        particles.append([x,y,z,vx,vy,vz,parti_r,parti_q,parti_m])

    return np.array(particles,dtype=float)

def collisions(parti:np.ndarray,threshold:float) -> np.ndarray:
    #set up the random number generator
    rng = np.random.default_rng()

    roll = rng.random(len(parti))

    hit = (roll[:]>threshold)&(np.abs(parti[:,2])>8)
    parti[hit,3:6] = [0,0,0]    
    
    return parti

def init_leap_frog(parti,e_field,m_field,dt):
    parti[:,3:6] -= (parti[:,7,np.newaxis]/parti[:,8,np.newaxis])*e_field*(dt/2)

    #parti[:,3:6] -= ((parti[:,7,np.newaxis]*(e_field+np.cross(parti[:,3:6],m_field)))/parti[:,8,np.newaxis])*(dt/2)

    return parti
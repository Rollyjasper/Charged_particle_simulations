import numpy as np
import time as tim

#import user-defined modules
import particles,visulization,enviroment

#create a list of particles
#parti = particles.create_particles(100,(-10,10),(-10,10),(-10,10),q=[-1],v=[0,0,0])

#define the timestep and the duration of the simulation
fps = 60
dt = 1/fps
time = np.arange(0,120,dt)

pos = []

x_euler = []
x_rk4 = []

e0 = np.array([1,0,0])
e = enviroment.constant_field(e0)

b0 = np.array([0,1,0])
b = enviroment.constant_field(b0)

parti = particles.create_particles(1,0.5,0.5,0.5,q=[-1],v=[1,0,0],exact_pos=True)

start = tim.time()
particles.init_leap_frog(parti,e(-dt/2),b(-dt/2),dt)
for t in time:
    pos.append(np.copy(parti[:,:3]))
    #particles.motion_euler(e(t),b(t),parti,dt)
    #particles.motion_range_kutta(parti,dt,t,e,b)
    b = enviroment.mag_bottle(parti[:,0],parti[:,1],parti[:,2],[-10,10],10.5,1e9)
    particles.motion_boris(parti,dt,e(t),b)

end = tim.time()

print(end-start)

pos = np.array(pos)
#visulization.plot_3d_ani(pos,fps,x_label='Electric Field',y_label='Magnetic Field')
#e_field = enviroment.parti_to_electric(parti,-10,10,200)

#x = np.linspace(-10,10,200)

#visulization.plot_2d_contor(x,x,e_field[2][95])
# e,b = np.array([0,0,0]),np.array([0,0,0])
# upper,lower,n = -10,10,250

# labels = [
#     ['x','y','z'],
#     ['y','z'],
#     ['y','z'],
#     ['y','z'],
# ]

# es = {
#     'range':(lower,upper,n)
# }

# bs = {

# }


# #start the simulation loop for the euler method
# for i,t in enumerate(time):
#     #this is for getting the position of particles
#     # pos.append(np.copy(parti[:,:3]))
#     e = enviroment.parti_to_electric(parti,upper,lower,n)
#     visulization.plot_frame(parti,e,b,es,bs,'test2','test',i,labels,plt_b=False)

    # e = enviroment.electric_to_parti(parti,e,upper,lower,n)
    # #move the particles
    # particles.motion_boris(parti,dt,e,b)

n = 20
xx,yy,zz,ds = enviroment.create_region(-10,10,n)
b = np.zeros((n,n,n,3)) #set the points to 0

for i in range(n): #iterate through the grid, setting each point equal to the magnetic field value there
    for j in range(n):
        for k in range(n):
            b[i,j,k] = enviroment.mag_bottle(xx[i,j,k],yy[i,j,k],zz[i,j,k],[-10,10],10.5,1e9) 
        
x = np.linspace(-10,10,n)

visulization.plot_2d_vector_path(x,x,b[10,:,:,1],b[10,:,:,2],pos,'x',x_label='y Axis',y_label='z Axis')
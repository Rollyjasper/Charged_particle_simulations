from matplotlib.animation import ArtistAnimation,PillowWriter
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import tkinter as tk
import numpy as np
import time as T
import os

def plot_3d_ani(pos,fps:float,*,x_label:str='x',y_label:str='y',z_label:str='z') -> None:
    '''Produces an animated, 3D plot for input data

    Takes in a lists of x,y,z co-ordinates

    x_label: label on the x axis
    y_label: label on the y axis
    z_label: label on the z axis
    ''' 
    fig = plt.figure()
    ax = plt.subplot(111,projection = '3d')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    plots = []
    for i in range(len(pos)):
        artist, = plt.plot(xs=pos[i,:,0],ys=pos[i,:,1],zs=pos[i,:,2], ls = '', color = '#000000', marker = '.')
        plots.append([artist])

    ani = ArtistAnimation(fig, plots, interval = 1000/fps, blit = True)

    #plt.show()

    f = os.getcwd()+'\\figures\\'+'25_exb_with_hits_10'+'.gif'
    writergif = PillowWriter(fps=fps) 
    ani.save(f, writer=writergif)



def plot_3d_path(pos,*,x_label:str='x',y_label:str='y',z_label:str='z') -> None:
    '''Produces an 3D plot for input data

    Takes in a lists of x,y,z co-ordinates

    x_label: label on the x axis
    y_label: label on the y axis
    z_label: label on the z axis
    ''' 
    fig = plt.figure()
    ax = plt.subplot(111,projection = '3d')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    for i in range(1,len(pos)):
        plt.plot(xs=[pos[i-1,0,0],pos[i,0,0]],ys=[pos[i-1,0,1],pos[i,0,1]],zs=[pos[i-1,0,2],pos[i,0,2]],color = '#000000')

    plt.show()

def plot_1d(data:list,fps:int,label:list,title:str = '',axis:str = '',legend:bool = False) -> None:
    duration = len(data[0])/fps
    t = np.arange(0,duration,1/fps)
    for i,part in enumerate(data):
        plt.plot(t,part,label = label[i])
    
    plt.title(title)
    plt.xlabel('time/s')
    plt.ylabel(axis)
    if legend == True:
        plt.legend()
    
    plt.show()

def plot_2d_vector(x,y,u,v,density:float=0.6) -> None:
    '''plots a 2D vector field

    x: x co-ordinates
    y: y co-ordiantes
    u: x component of the field
    v: y component of the field
    density: how dence the arrows are
    '''
    fig = plt.figure()
    ax = plt.subplot(111)

    ax.streamplot(x,y,u,v,density=density)

    ax.plot([-10,-10],[-5,5])
    ax.plot([10,10],[-5,5])

    ax.set_xlabel('y Axis')
    ax.set_ylabel('z Axis')

    ax.set_title('Particle Motion in a Magnetic Bottle')

    plt.show()

def plot_2d_path(path_data,suppressed:str='z',*,x_label:str='x',y_label:str='y') -> None:
    '''plots a 2D path for particles

    path_data: list of x,y,z co-ordinates over time
    suppressed: the dimention to be 'hidden' by the plot

    x_label: label on x axis
    y_label: label on y axis
    '''
    fig = plt.figure()
    ax = plt.subplot(111)

    match suppressed:
        case 'x':
            i,j = 2,1
        case 'y':
            i,j = 0,2
        case 'z':
            i,j = 0,1

    ax.plot(path_data[:,:,i],path_data[:,:,j])

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title('Motion in a Uniform Magnetic Field: Boris')

    plt.show()

def plot_2d_vector_path(x,y,u,v,path_data,suppressed:str='z',*,x_label:str='x',y_label:str='y',density:float=0.6) -> None:
    fig = plt.figure()
    ax = plt.subplot(111)

    match suppressed:
        case 'x':
            i,j = 1,2
        case 'y':
            i,j = 0,2
        case 'z':
            i,j = 0,1

    ax.streamplot(x,y,u,v,density=density,color = '#000000')

    ax.plot(path_data[:,:,i],path_data[:,:,j])

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.set_xlim([-10,10])
    ax.set_ylim([-10,10])

    ax.set_title('Particle Motion in a Magnetic Bottle')

    plt.show()

def plot_2d_contor(x,y,z):
    h = plt.contourf(x, y, z)

    plt.axis('scaled')

    plt.colorbar()

    plt.show()
    
def plot_2d_rt(particles:np.ndarray,fps:float,func,*,scale=10,collisional=False,width = 500, height = 500) -> None:
    '''
    '''
    root = tk.Tk()
    root.geometry(str(width)+'x'+str(height))

    frame = tk.Frame(root)
    frame.grid()

    ani = real_time_ani(frame,particles,fps,func,scale=scale,collisional=collisional,width = width,height = height)
    root.mainloop()

def plot_frame(particles,e_field,m_field,e_field_spec,m_field_spec,folder,file_name,frame,labels,*,plt_e=True,plt_b=True):

    es = e_field_spec
    bs = m_field_spec
    
    ex,ey,ez = e_field

    path = os.getcwd()+'\\animation\\'+folder+'\\'+file_name+'_'+str(frame)+'.png'

    if plt_e and plt_b:
        pass
    elif not plt_e and plt_b:
        pass
    elif plt_e and not plt_b:
        fig = plt.figure(figsize=(14.0,7.0))

        gs = GridSpec(2,4,figure=fig)

        ax1 = fig.add_subplot(gs[0:2,0:2],projection = '3d')
        ax2 = fig.add_subplot(gs[0,2])
        ax3 = fig.add_subplot(gs[1,2])
        ax4 = fig.add_subplot(gs[0,3])

        ax1.plot(xs=particles[:,0],ys=particles[:,1],zs=particles[:,2],ls='',marker='.',color='black')
        ax1.set_xlabel(labels[0][0])
        ax1.set_ylabel(labels[0][1])
        ax1.set_zlabel(labels[0][2])
        ax1.set_title('Particle Motion from Particle Generated Electric Field t={:0.2f}s'.format(frame/30))

        layer = len(ex)//2
        ticks = np.linspace(es['range'][0],es['range'][1],es['range'][2])

        
        ax2.imshow(ex[layer],origin='lower')
        ax2.set_xlabel(labels[1][0])
        ax2.set_ylabel(labels[1][1])
        ax2.set_title('Ex/x=0')
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        
        ax3.imshow(ey[layer],origin='lower')
        ax3.set_xlabel(labels[2][0])
        ax3.set_ylabel(labels[2][1])
        ax3.set_title('Ey/x=0')
        ax3.set_xticklabels([])
        ax3.set_yticklabels([])
        
        ax4.imshow(ez[layer],origin='lower')
        ax4.set_xlabel(labels[3][0])
        ax4.set_ylabel(labels[3][1])
        ax4.set_title('Ez/x=0')
        ax4.set_xticklabels([])
        ax4.set_yticklabels([])


    else:
        pass

    plt.savefig(path)
    plt.close(fig)
class real_time_ani(object):
    def __init__(self,master,/,particles:np.ndarray,fps:float,func,*,scale=1,collisional=False,width = 100,height = 100) -> None:
        self.canvas = tk.Canvas(master,background='#88ff88',width=width,height=height)
        self.canvas.grid(row=0,column=0,rowspan=2)

        self.master = master
        self.particles = particles
        self.fps = fps
        self.scale = scale
        self.collisional = collisional
        self.func = func

        self.width = width
        self.height = height
        self.horizon = 100

        self.dt = 1/fps
        self.colours = {-1:'#ff0000',1:'#0000ff',0:'#000000'}

        self.particle_list = []

        self.init_state()

        self.loop()
    
    def init_state(self):
        for particle in self.particles:
            x,y = self.coo_to_px(particle[0],particle[1])
            r = self.z_axis(particle[6]*self.scale,0,particle[2])
            self.particle_list.append(self.canvas.create_oval(x-r,y-r,x+r,y+r,fill=self.colours[particle[7]]))
        
        self.particle_list = np.array(self.particle_list)
    
    def coo_to_px(self,x,y):
        pxx = (self.width/2)+(x*self.scale)
        pxy = (self.height/2)+(y*self.scale)
        return pxx,pxy
    
    def z_axis(self,r1,z1,z2):
        a1 = z1-self.horizon
        a2 = z2-self.horizon
        r2 = (a2/a1)*r1
        return r2
    
    def loop(self):
        for i in range(10*self.fps):
            self.particles = self.func(self.particles,self.dt,self.collisional)
            self.update_pos()
            self.master.update()
            T.sleep(self.dt)
        print('done')

    
    def update_pos(self):
        for i,particle in enumerate(self.particles):
            x,y = self.coo_to_px(particle[0],particle[1])
            r = self.z_axis(particle[6]*self.scale,0,particle[2])
            self.canvas.coords(self.particle_list[i],x-r,y-r,x+r,y+r)
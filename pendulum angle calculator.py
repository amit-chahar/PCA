
from numpy import sin, cos
from math import radians
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
from scipy import *
class DoublePendulum:
    """Double Pendulum Class

    init_state is [theta1, omega1, theta2, omega2] in degrees,
    where theta1, omega1 is the angular position and velocity of the first
    pendulum arm, and theta2, omega2 is that of the second pendulum arm
    """
    def __init__(self,
                 origin=(0, 0)):
        self.theta = 90
        self.dtheta = 0
        self.origin = origin
        self.time_elapsed = 0
    
    def position(self):
	scaling = 300.0/(SWINGLENGTH**2)
 
        firstDDtheta = -sin(radians(self.theta))*scaling
        midDtheta = self.dtheta + firstDDtheta
        midtheta = self.theta + (self.dtheta + midDtheta)/2.0
 
        midDDtheta = -sin(radians(midtheta))*scaling
        midDtheta = self.dtheta + (firstDDtheta + midDDtheta)/2
        midtheta = self.theta + (self.dtheta + midDtheta)/2
 
        midDDtheta = -sin(radians(midtheta)) * scaling
        lastDtheta = midDtheta + midDDtheta
        lasttheta = midtheta + (midDtheta + lastDtheta)/2.0
 
        lastDDtheta = -sin(radians(lasttheta)) * scaling
        lastDtheta = midDtheta + (midDDtheta + lastDDtheta)/2.0
        lasttheta = midtheta + (midDtheta + lastDtheta)/2.0
 
        self.dtheta = lastDtheta
        self.theta = lasttheta
	print(str(self.theta)),
	x = np.cumsum([self.origin[0],-SWINGLENGTH*sin(radians(self.theta))])
	y = np.cumsum([self.origin[1],-SWINGLENGTH*cos(radians(self.theta))])
        return (x, y)

    def step(self, dt):
        """execute one time step of length dt and update state"""
        #self.state = integrate.odeint(self.dstate_dt, self.state, [0, dt])[1]
        self.time_elapsed += dt

#------------------------------------------------------------
# set up initial state and global variables
pendulum = DoublePendulum()
dt = 1./30 # 30 fps
PIVOT = (12.5, 2.5)
SWINGLENGTH = PIVOT[1]*7
 

#------------------------------------------------------------
# set up figure and animation
WIDTH = 51
HEIGHT = 51

dpi = plt.rcParams['figure.dpi']
plt.rcParams['savefig.dpi'] = dpi
plt.rcParams["figure.figsize"] = (1.0 * WIDTH / dpi, 1.0 * HEIGHT / dpi)

fig = plt.figure()
#fig.set_size_inches(2,3,True)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-25, 25), ylim=(-30, 20),axisbg='black')
ax.grid()
#For removing the x and y-axis numbering
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
fig.tight_layout(pad=0.0) 
#plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
line, = ax.plot([], [],  '-',color='white',linewidth=3)  #markersize = 10

def init():
    """initialize animation"""
    line.set_data([], [])
    return line,

def animate(i):
    """perform animation step"""
    global pendulum, dt,SWINGLENGTH,PIVOT
    #pendulum.step(dt)
    line.set_data(*pendulum.position())
    return line,

# choose the interval based on dt and the time to animate one step
from time import time
t0 = time()
animate(0)
t1 = time()
interval = 1000 * dt - (t1 - t0)

ani = animation.FuncAnimation(fig, animate, frames=500,
                              interval=30, blit=True, init_func=init)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html

#ani.save('double_pendulum.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()

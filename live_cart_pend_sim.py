## Simple simulation for a swinging pendulum on a moving cart.
#  Use python3, not conda
#  The equations of motion are solved in real time between the frames of the 
#  animation. Run using sudo to control the cart with an applied force using 
#  the right and left arrow keys.

import math
from time import time
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import keyboard

class pendulum_model():
    
    def __init__(self):

        # Define system parameters:
        self.L = 3 
        self.mc = 10
        self.mp = .1
        self.b = 10 # Viscous damping coefficient
        self.g = 9.81

        self.state = np.array([0.1, 0, 0, 0]) # Initial State [rad, rad/s]
        self.elapsed_time = 0


    def position(self):
        # Calculate x,y position
        x = self.state[2]
        xp = x + self.L*math.sin(np.pi-self.state[0])
        yp = self.L*math.cos(np.pi-self.state[0])
        return x,xp,yp


    def dydt(self,t,y):
        # Define dynamics of pendulum model
        # y[0] = theta
        # y[1] = theta_dot
        # y[2] = x
        # y[3] = x_dot

        if keyboard.is_pressed('right'):
            F = 4
        elif keyboard.is_pressed('left'):
            F = -4
        else: 
            F = 0

        F = F*10

        theta_dot = y[1]
        theta_ddot = (-F*math.cos(y[0]) - self.mp*self.L*y[1]**2*math.cos(y[0])*math.sin(y[0]) - (self.mc+self.mp)*self.g*math.sin(y[0]))/(self.L*(self.mc + self.mp*math.sin(y[0])**2))
        x_dot = y[3]
        x_ddot = (F + self.mp*math.sin(y[0])*(self.L*y[1]**2 + self.g*math.cos(y[0])) - self.b*y[3])/(self.mc + self.mp*math.sin(y[0])**2)

        return np.r_[theta_dot,theta_ddot,x_dot,x_ddot]


    def step(self, dt):
        # Calculate one time step
        self.state = solve_ivp(self.dydt, t_span=[0,dt], y0=self.state).y[:,-1]
        self.elapsed_time += dt


my_model = pendulum_model()
dt = 0.05

fig = plt.figure(figsize=(6,6))
plt.subplots_adjust(top=1,bottom=0,right=1,left=0)
ax = plt.axes()
line, = ax.plot([],[])
circle = ax.scatter([],[],s=200)
#cart = ax.scatter([],[],s=500)
width = 1
height = 0.5
cart = Rectangle([-width/2,0],width,height,fill=False)
ax.add_patch(cart)
time_text = ax.text(0.01, 0.01, '', transform=ax.transAxes)
plt.title('Cart-Pendulum Simulator')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.xlim([-4,4])
plt.ylim([-4,4])
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
#plt.axes().get_xaxis().set_visible(False)
#ax.get_yaxes().set_visible(True)
plt.grid(True)


def animate(i):
    my_model.step(dt)

    x,xp,yp = my_model.position()

    line.set_data([x,xp],[0+height/2,yp+height/2])
    circle.set_offsets([xp,yp+height/2])
    cart.set_xy((x-width/2,0))
    time_text.set_text('t={}s'.format(round(my_model.elapsed_time)))
    ax.set_xlim(x-4,x+4)
    return line,circle,cart,time_text,ax


t0 = time()
animate(0)
t1 = time()
interval = 1000 * (dt - (t1 - t0))
print('Interval: {} s'.format(interval/1000))

pend_ani = animation.FuncAnimation(fig, animate, frames=1000, interval=interval, blit=True)
plt.show()
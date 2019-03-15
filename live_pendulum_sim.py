## Simple simulation for a swinging pendulum.
#  Use python3, not conda

import math
from time import time
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import keyboard

class pendulum_model():
    
    def __init__(self):

        # Define system parameters:
        self.L = 1 
        self.m = 1
        self.b = 0.2 # Viscous damping coefficient
        self.g = 9.81

        self.state = np.array([0.1, 0]) # Initial State [rad, rad/s]
        self.elapsed_time = 0


    def position(self):
        # Calculate x,y position
        x = self.L*math.sin(self.state[0])
        y = self.L*math.cos(self.state[0])
        return x,y


    def dydt(self,t,y):
        # Define dynamics of pendulum model
        # y[0] = theta
        # y[1] = theta_dot

        if keyboard.is_pressed('right'):
            tau = 1
        elif keyboard.is_pressed('left'):
            tau = -1
        else: 
            tau = 0

        theta_dot = y[1]
        #theta_ddot = self.g/self.L*math.sin(y[0]) - self.b/self.m/self.L**2*theta_dot + self.m*self.L**2*tau
        theta_ddot = self.m*self.L**2*tau
        #theta_ddot = - self.b/self.m/self.L**2*theta_dot + self.m*self.L**2*tau

        return np.r_[theta_dot,theta_ddot]


    def step(self, dt):
        # Calculate one time step
        self.state = solve_ivp(self.dydt, t_span=[0,dt], y0=self.state).y[:,-1]
        self.elapsed_time += dt


my_model = pendulum_model()
dt = 0.03

fig = plt.figure(figsize=(6,6))
ax = plt.axes()
line, = ax.plot([],[])
circle = ax.scatter([],[],s=200)
time_text = ax.text(0.01, 0.01, '', transform=ax.transAxes)
plt.title('Pendulum Test')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.xlim([-1.2,1.2])
plt.ylim([-1.2,1.2])
plt.grid(True)

def animate(i):
    my_model.step(dt)

    x,y = my_model.position()

    line.set_data([0,x],[0,y])
    circle.set_offsets([x,y])
    time_text.set_text('t={}s'.format(round(my_model.elapsed_time)))
    return line,circle,time_text


t0 = time()
animate(0)
t1 = time()
interval = 1000 * dt - (t1 - t0)

pend_ani = animation.FuncAnimation(fig, animate, frames=1000, interval=interval, blit=True)
plt.show()
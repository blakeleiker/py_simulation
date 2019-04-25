## Simple simulation for a 2 body orbit. Solved in real time.
#  Use python3, not conda

import math
from time import time
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#import keyboard

class orbit_model():
    
    def __init__(self):

        # Define system parameters:
        self.m_earth = 1000000000000
        self.m_moon = .001

        self.r_earth = np.array([0, 0])
        self.r_moon = np.array([10, 0])

        self.G = 6.664*10**-11

        self.state = np.r_[1,2,self.r_moon] # Initial State [m/s,m/s,m,m]
        self.elapsed_time = 0


    def position(self):
        # Calculate x,y position

        return self.state[2], self.state[3]


    def dydt(self,t,y):
        # Define dynamics of pendulum model
        # y[0] = rx_dot
        # y[1] = ry_dot
        # y[2] = rx
        # y[3] = ry

        r_hat = (self.r_earth-y[2:])/np.linalg.norm(self.r_earth-y[2:])
        F = self.G*self.m_earth*self.m_moon/(np.linalg.norm(y[2:]-self.r_earth)**2) * r_hat
        r_moon_ddot = F/self.m_moon

        return np.r_[r_moon_ddot,y[0],y[1]]


    def step(self, dt):
        # Calculate one time step
        self.state = solve_ivp(self.dydt, t_span=[0,dt], y0=self.state).y[:,-1]
        print(self.state)
        self.elapsed_time += dt


my_model = orbit_model()
dt = .1

fig = plt.figure(figsize=(6,6))
ax = plt.axes()
earth = ax.scatter([my_model.r_earth[0]],[my_model.r_earth[1]],s=200)
moon = ax.scatter([my_model.r_moon[0]],[my_model.r_moon[1]],s=50)
time_text = ax.text(0.01, 0.01, '', transform=ax.transAxes)
plt.title('2 Body Orbit')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.xlim([-20,20])
plt.ylim([-20,20])
plt.grid(True)

def animate(i):
    my_model.step(dt)

    x,y = my_model.position()

    moon.set_offsets([x,y])
    time_text.set_text('t={}s'.format(round(my_model.elapsed_time)))
    return moon,time_text


t0 = time()
animate(0)
t1 = time()
interval = 1000 * dt - (t1 - t0)

pend_ani = animation.FuncAnimation(fig, animate, frames=1000, interval=interval, blit=True)
plt.show()
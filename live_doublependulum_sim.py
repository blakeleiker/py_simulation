## Simple simulation for a swinging double pendulum.
#  Use python3, not conda
#  The equations of motion of the pendulums are solved in real time between the 
#  frames of the animation. Run using sudo to control the pendulum with an applied 
#  torque at the center pivot using the right and left arrow keys.

import math
from time import time
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import keyboard

# Redefine useful math functions
sin = math.sin
cos = math.cos

class pendulum_model():
    
    def __init__(self):

        # Define system parameters:
        self.L1 = .5
        self.L2 = .5
        self.m1 = 1
        self.m2 = .5
        self.b = 0.2 # Viscous damping coefficient
        self.g = 9.81

        self.state = np.array([3.14, 0, 3.14, 0]) # Initial State [rad, rad/s]
        self.elapsed_time = 0


    def position(self):
        # Calculate x,y position
        x1 = self.L1*math.sin(self.state[0])
        y1 = -self.L1*math.cos(self.state[0])
        x2 = x1 + self.L2*math.sin(self.state[2])
        y2 = y1 - self.L2*math.cos(self.state[2])

        return x1,y1,x2,y2


    def dydt(self,t,y):
        # Define dynamics of double pendulum model.
        # y[0] = theta1
        # y[1] = theta1_dot
        # y[2] = theta2
        # y[3] = theta2_dot

        if keyboard.is_pressed('right'):
            tau = 1
        elif keyboard.is_pressed('left'):
            tau = -1
        else: 
            tau = 0

        tau = tau*10

        theta1_dot = y[1]
        theta1_ddot = (-self.g*(2*self.m1 + self.m2)*sin(y[0]) - self.m2*self.g*sin(y[0]-2*y[2]) - 2*sin(y[0]-y[2])*self.m2*(y[3]**2*self.L2 + y[1]**2*self.L1*cos(y[0]-y[2])))/( self.L1*(2*self.m1 + self.m2 - self.m2*cos(2*y[0]-2*y[2])) ) - self.b*y[1] + tau
        theta2_dot = y[3]
        theta2_ddot = (2*sin(y[0]-y[2])*( y[1]**2*self.L1*(self.m1+self.m2) + self.g*(self.m1+self.m2)*cos(y[0]) + y[3]**2*self.L2*self.m2*cos(y[0]-y[2]) ))/(self.L2*( 2*self.m1 + self.m2 - self.m2*cos(2*y[0]-2*y[2]) )) - self.b*(y[3]-y[1])

        return np.r_[theta1_dot,theta1_ddot,theta2_dot,theta2_ddot]


    def step(self, dt):
        # Calculate one time step
        self.state = solve_ivp(self.dydt, t_span=[0,dt], y0=self.state).y[:,-1]
        self.elapsed_time += dt


my_model = pendulum_model()
dt = 0.03

fig = plt.figure(figsize=(6,6))
ax = plt.axes()
line, = ax.plot([],[])
circle1 = ax.scatter([],[],s=200)
circle2 = ax.scatter([],[],s=200)
time_text = ax.text(0.01, 0.01, '', transform=ax.transAxes)
plt.title('Pendulum Test')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.xlim([-1.2,1.2])
plt.ylim([-1.2,1.2])
plt.grid(True)

def animate(i):
    my_model.step(dt)

    x1,y1,x2,y2 = my_model.position()

    line.set_data([0,x1,x2],[0,y1,y2])
    circle1.set_offsets([x1,y1])
    circle2.set_offsets([x2,y2])
    time_text.set_text('t={}s'.format(round(my_model.elapsed_time)))
    return line,circle1,circle2,time_text


t0 = time()
animate(0)
t1 = time()
interval = 1000 * dt - (t1 - t0)

pend_ani = animation.FuncAnimation(fig, animate, frames=1000, interval=interval, blit=True)
plt.show()
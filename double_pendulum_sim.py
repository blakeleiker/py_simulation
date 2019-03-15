## Simple simulation for a swinging double pendulum.

import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define system parameters:
L1 = 1
L2 = 1  
m1 = 1
m2 = 1
b = 0.2 # Viscous damping coefficient
g = 9.81

sin = math.sin
cos = math.cos

# Define pendulum model:
def model(t,y):
    # y[0] = theta1
    # y[1] = theta1_dot
    # y[2] = theta2 
    # y[3] = theta2_dot

    theta1_dot = y[1]
    theta1_ddot = (-g*(2*m1 + m2)*sin(y[0]) - m2*g*sin(y[0]-2*y[2]) - 2*sin(y[0]-y[2])*m2*(y[3]**2*L2 + y[1]**2*L1*cos(y[0]-y[2])))/( L1*(2*m1 + m2 - m2*cos(2*y[0]-2*y[2])) ) - b*y[1]
    theta2_dot = y[3]
    theta2_ddot = (2*sin(y[0]-y[2])*( y[1]**2*L1*(m1+m2) + g*(m1+m2)*cos(y[0]) + y[3]**2*L2*m2*cos(y[0]-y[2]) ))/(L2*( 2*m1 + m2 - m2*cos(2*y[0]-2*y[2]) )) - b*(y[3]-y[1])

    return np.r_[theta1_dot,theta1_ddot,theta2_dot,theta2_ddot]

tspan = [0,30]
y0 = [3.14, 0, 3.14, 0] # [rad, rad/s, rad, rad/s]
dt = 1/60
teval = np.arange(tspan[0],tspan[1],dt)
sol = solve_ivp(model, tspan, y0, t_eval=teval, method='RK45', max_step=dt)

'''
plt.figure(figsize=(8,5))
plt.plot(sol.t, sol.y[0], label='theta')
plt.title('Pendulum')
plt.xlabel('t')
plt.legend()
plt.grid(True)
plt.show()
'''

x1 = L1*np.sin(sol.y[0])
y1 = -L1*np.cos(sol.y[0])
x2 = x1 + L2*np.sin(sol.y[2])
y2 = y1 - L2*np.cos(sol.y[2])

fig = plt.figure(figsize=(6,6))
ax = plt.axes()
line, = ax.plot([],[])
circle1 = ax.scatter([],[],s=200)
circle2 = ax.scatter([],[],s=200)
time_text = ax.text(0.01, 0.01, '', transform=ax.transAxes)
plt.title('Double Pendulum Test')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.xlim([-(L1+L2)-.2,L1+L2+.2])
plt.ylim([-(L1+L2)-.2,L1+L2+.2])
plt.grid(True)

def animate(i):
    line.set_data([0,x1[i],x2[i]],[0,y1[i],y2[i]])
    circle1.set_offsets([x1[i],y1[i]])
    circle2.set_offsets([x2[i],y2[i]])
    time_text.set_text('t={}s'.format(round(sol.t[i],1)))
    return line,circle1,circle2,time_text

pend_ani = animation.FuncAnimation(fig, animate, frames=len(x1), interval=dt*1000, blit=True)
plt.show()
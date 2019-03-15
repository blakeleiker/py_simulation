## Simple simulation for a swinging pendulum.

import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define system parameters:
L = 1 
b = 0.2 # Viscous damping coefficient
g = 9.81

# Define pendulum model:
def model(t,y):
    # y[0] = theta
    # y[1] = theta_dot

    theta_dot = y[1]
    theta_ddot = g/L*math.sin(y[0]) - b*theta_dot

    return np.r_[theta_dot,theta_ddot]

tspan = [0,30]
y0 = [0.1, 0] # [rad, rad/s]
dt = 0.03
sol = solve_ivp(model, tspan, y0, method='RK45', max_step=dt)
print(np.diff(sol.t))
input()

'''
plt.figure(figsize=(8,5))
plt.plot(sol.t, sol.y[0], label='theta')
plt.title('Pendulum')
plt.xlabel('t')
plt.legend()
plt.grid(True)
plt.show()
'''

x = L*np.sin(sol.y[0])
y = L*np.cos(sol.y[0])

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
    line.set_data([0,x[i]],[0,y[i]])
    circle.set_offsets([x[i],y[i]])
    time_text.set_text('t={}s'.format(round(sol.t[i],1)))
    return line,circle,time_text

pend_ani = animation.FuncAnimation(fig, animate, frames=len(x), interval=dt*1000, blit=True)
plt.show()
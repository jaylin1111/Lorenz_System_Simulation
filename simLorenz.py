import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.axes3d import Axes3D

beta = [10,28,8/3]  # chaotic values
x0 = [0,1,20]       # intial conditions
dt = 0.01          # time step
t_start = 0         # start time
t_stop = 15         # stop time

t_span = np.arange(t_start, t_stop, dt)
numDataPoints = len(t_span)

def lorenz(t, x, beta):
    dx = np.zeros_like(x)
    dx = [
        beta[0]*(x[1]-x[0]),
        x[0]*(beta[1]-x[2])-x[1],
        x[0]*x[1] - beta[2]*x[2]
    ]
    return dx

sol = integrate.odeint(lorenz, x0, t_span, args=(beta,), tfirst=True)
dataSet = np.array([sol[:,0], sol[:,1], sol[:,2], t_span])

plt.style.use('dark_background')
fig = plt.figure(figsize=(7, 7))
ax = Axes3D(fig)
fig.add_axes(ax)

line = plt.plot(sol[:,0], sol[:,1], sol[:,2], lw=1.5, c='c')[0]

def func(num, dataSet, line):
    line.set_data(dataSet[0:2, :num])
    line.set_3d_properties(dataSet[2, :num])
    return line

line_ani = animation.FuncAnimation(fig, func, frames=numDataPoints, fargs=(dataSet, line), interval=1, blit=False)

# plt.show()

Writer = animation.writers['ffmpeg']
writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1e7)
line_ani.save('lorenzSin.mp4', writer=writer)

# ALL THE LIBRARY IMPORTS
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3


# DEFINING THE CONSTANTS AND PARAMETERS
beta = [10,28,8/3]                  # chaotic values
err = 1e-2                          # error amount
e_ar = [err, err, err]              # error array
x0_1 = [0,1,20]                     # initial conditions 1
x0_2 = np.add(x0_1, e_ar)           # initial conditions 2
x0_3 = np.subtract(x0_1, e_ar)      # initial conditions 3
dt = 0.01                           # time step
t_start = 0                         # start time
t_stop = 30                         # stop time

t_span = np.arange(t_start, t_stop, dt)
numDataPoints = len(t_span)


# LORENZ SYSTEM
def lorenz(t, x, beta):
    dx = np.zeros_like(x)
    dx = [
        beta[0]*(x[1]-x[0]),
        x[0]*(beta[1]-x[2])-x[1],
        x[0]*x[1] - beta[2]*x[2]
    ]
    return dx


# SOLVING AND SAVING DATA
sol1 = integrate.odeint(lorenz, x0_1, t_span, args=(beta,), tfirst=True)
sol2 = integrate.odeint(lorenz, x0_2, t_span, args=(beta,), tfirst=True)
sol3 = integrate.odeint(lorenz, x0_3, t_span, args=(beta,), tfirst=True)

data1 = np.array([sol1[:,0], sol1[:,1], sol1[:,2]])
data2 = np.array([sol2[:,0], sol2[:,1], sol2[:,2]])
data3 = np.array([sol3[:,0], sol3[:,1], sol3[:,2]])
  # There is definitely a way to make this better, but I only need 3 lines
  # And it's not worth my time to figure it out cause that's not my job


# UPDATE LINES FUNCTION
def update_lines(num, data1, data2, data3, line1, line2, line3):
    line1.set_data(data1[0:2, :num])
    line1.set_3d_properties(data1[2, :num])
    
    line2.set_data(data2[0:2, :num])
    line2.set_3d_properties(data2[2, :num])
    
    line3.set_data(data3[0:2, :num])
    line3.set_3d_properties(data3[2, :num])
    
    return line1, line2, line3


# PLOTTING DETAILS
plt.style.use('dark_background')
  # For some reason, if you move this line below ax = p3.Axes3D(fig)
  # Them it'll make the background white. Interesting to say the least
  # Once again, I'm not gonna bother asking why and figuring it out
fig = plt.figure(figsize=(9, 8))
ax = p3.Axes3D(fig)

ax.set_xlim([-19,19])
ax.set_ylim([-29,29])
ax.set_zlim([1,49])
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('grey')
ax.yaxis.pane.set_edgecolor('grey')
ax.zaxis.pane.set_edgecolor('grey')


# DEFINE LINES
line1, = [ax.plot(data1[0, 0:1], data1[1, 0:1], data1[2, 0:1], color="r", lw=2)[0]]
line2, = [ax.plot(data2[0, 0:1], data2[1, 0:1], data2[2, 0:1], color="g", lw=2)[0]]
line3, = [ax.plot(data3[0, 0:1], data3[1, 0:1], data3[2, 0:1], color="b", lw=2)[0]]


# ANIMATION FUNCTION
line_ani = animation.FuncAnimation(fig, update_lines, numDataPoints, fargs=(data1, data2, data3, line1, line2, line3), interval=50, blit=False)


# SAVING
# I used ffmpeg cause it was the easiest to download, use whatever you want
Writer = animation.writers['ffmpeg']
writer = Writer(fps=60, metadata=dict(artist='jlin'), bitrate=1e7)
line_ani.save('lorenzSim.mp4', writer=writer)


# SHOW PLOT
plt.show()


# And I'm fucking done with this project
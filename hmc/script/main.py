import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
# mpl.rc('font', size = 16)
# mpl.rc('axes', titlesize = 'large', labelsize = 'large')
# mpl.rc('xtick', labelsize = 'large')
# mpl.rc('ytick', labelsize = 'large')
from matplotlib.animation import FuncAnimation
import torch
from torch.distributions import MultivariateNormal
from sys import exit

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = ax.plot([], [], 'ro')

def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    return ln,

def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return ln,

ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                    init_func=init, blit=True)
ani.save('./output/mc.mp4', writer = 'ffmpeg')
exit()

print('here')


#### define the distribution
loc = torch.zeros(2)
cov = torch.tensor([[1.0, 0.98], [0.98, 1.0]])
dist = MultivariateNormal(loc, cov)

#### Metropolis MC
x_record = []
x_current = torch.tensor([-2., -2.])
x_record.append(x_current)

cov_proposal = torch.tensor([[0.18, 0.0], [0.0, 0.18]])
accept_flag = []
for _ in range(20):
    for _ in range(20):
        x_proposal = MultivariateNormal(
            loc = x_current,
            covariance_matrix = cov_proposal).sample()
        logp = dist.log_prob(x_current)
        logq = dist.log_prob(x_proposal)
        alpha = torch.exp(logq - logp)
        if np.random.random() < alpha:
            accept_flag.append(True)
            x_current = x_proposal
        else:
            accept_flag.append(False)
    x_record.append(x_current)
    
n = 30
x = torch.linspace(-2.5, 2.5, n)
y = torch.linspace(-2.5, 2.5, n)

xv, yv = torch.meshgrid(x, y, indexing = 'xy')
points = torch.vstack([xv.reshape(-1), yv.reshape(-1)]).t()
logpdf = dist.log_prob(points)
logpdf = logpdf.reshape(n, n)
# plt.contour(xv, yv, logpdf, levels = torch.linspace(-5,logpdf.max(),7))
# plt.colorbar()
# plt.savefig(f'./output/potential_contour.pdf')

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = ax.plot([], [], 'ro')

def init():
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    return ln,

def update(data):
    xdata.append(data[0])
    ydata.append(data[1])
    ln.set_data(xdata, ydata)
    return ln,

ani = FuncAnimation(fig, update, frames=[x.tolist() for x in x_record],
                    init_func=init, blit=True)

ani.save('./output/mc.mp4', writer = 'ffmpeg')

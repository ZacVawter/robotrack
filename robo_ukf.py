from time import sleep
import numpy as np
from numpy import ma
import matplotlib.pyplot as plt

from robo_motion import Robot
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise

rnd = np.random.RandomState(0)

step = 0.01
r = Robot((0.0, 0.0), [(0.1, 1.0, 0.0),(1, 1.0, np.pi),(1.0, 1.0, 0.0)])

class Filter:
  def __init__(self):
    n_dim_state = 4 # x = (x, y, vx, vy)

    def fx(x, dt):
      # state transition function - predict next state based
      # on constant velocity model x = vt + x_0
      F = np.array([[1, 0, dt,  0],
                    [0, 1,  0,  0],
                    [0, 0,  1, dt],
                    [0, 0,  0,  1]], dtype=float)
      return np.dot(F, x)

    def hx(x):
      return x[0:2]

    points = MerweScaledSigmaPoints(n_dim_state, alpha=.1, beta=2., kappa=-1)
    self.kf = UnscentedKalmanFilter(n_dim_state, 2, step, hx, fx, points)
    self.kf.x = np.array([0.0, 0.0, 0.0, 0.0])
    self.kf.P *= 0.1
    self.kf.R = np.diag([0.01**2, 0.01**2])
    self.kf.Q = Q_discrete_white_noise(dim=2, dt=step, var=0.2**2, block_size=2)

if __name__ == "__main__":
  plt.ion()
  fig = plt.figure(figsize=(16, 16))
  main_axes = fig.add_subplot(111)
  main_axes.set_ylim([-1,1])
  main_axes.set_xlim([-1,1])

  truth_line, = plt.plot([],[], marker='_', color='g', label="Truth")
  obs_line, = plt.plot([], [], marker='x', color='b',
                         label='observations')
  position_line, = plt.plot([], [],
                        linestyle='-', marker='.', color='r',
                        label='position est.')
  velocity_line, = plt.plot([], [],
                        linestyle='-', marker='o', color='g',
                        label='10 * velocity est.')

  plt.legend(loc='lower right')
  plt.xlabel('time')
  plt.grid(True)


  def add_xy(mplot, x, y):
    mplot.set_xdata(np.append(mplot.get_xdata(), x))
    mplot.set_ydata(np.append(mplot.get_ydata(), y))

  filter = Filter()
  rnd = np.random.RandomState(0)
  t = 0.0
  while True:
    t += step
    r.step(step)

    pos = r.pos
    noise = rnd.normal(0.0, 0.01, 4)

    sim_measurements = [r.pos[0], r.pos[1], 0.0, 0.0]

    x,y,lin_v,rot_v = tuple(sim_measurements)  

    
    measurements = ma.asarray(sim_measurements + noise)
    measurements[2] = ma.masked
    measurements[3] = ma.masked

    obsv_x, obsv_y, obsv_lin_v, obsv_rot_v = tuple(measurements)

    filter.kf.predict()
    filter.kf.update([obsv_x, obsv_y])

    means = filter.kf.x 
    print("pos = ", pos, " means = ", means)

    add_xy(truth_line, x, y)
    add_xy(obs_line, obsv_x, obsv_y)
    add_xy(position_line, means[0], means[1])
    #add_xy(velocity_line, t, means[1] * 10)

    fig.canvas.draw()
    fig.canvas.flush_events()

    sleep(0.02)




import numpy as np
from math import cos, sin

def Normalize(radians):
  return (radians + np.pi) % (2 * np.pi) - np.pi

class Robot:
  # commands is a list of tuples (commands), each command tuple
  # should be 3 elements (time duration in seconds, linear velocity, angular velocity)
  # example: [(1.0, 1.0, 0.0), (0.5, 0.0, 3.14)] == drive forward for 1 second, then
  # turn right for 0.5 seconds.
  def __init__(self, initial_pos, commands):
    self.commands = commands
    self.pos = np.array(initial_pos)
    self.heading = 0.0
    self.velocity = 0.0
    self.angular_velocity = 0.0

  def commands_remaining(self):
    return len(self.commands)

  def step(self, dt):
    cmd = self.commands.pop(0)

    velocity_and_angular_v = np.array([cmd[1], cmd[2]])
    motion = np.array([[cos(self.heading), 0.0],
                       [sin(self.heading), 0.0],
                       [0                , 1.0]])

    delta_pos = motion.dot(velocity_and_angular_v) * dt
    
    self.velocity = cmd[1]
    self.angular_velocity = cmd[2]
    self.pos = self.pos + delta_pos[0:2]
    self.heading = Normalize(self.heading + delta_pos[2])

    if cmd[0] > dt*0.5:
      cmd = (cmd[0] - dt, cmd[1], cmd[2])
      self.commands.insert(0,cmd)
    else:
      print("finished cmd( ", cmd,")")

if __name__ == "__main__":
  #r = Robot([(1.0, 1.0, 0.0),(1.0, 0.0, -np.pi)])
  r = Robot(np.zeros((2)), [(1.0, 1.0, -np.pi)])

  time = 0.0
  step = 0.1
  while r.commands_remaining() > 0:
    print("time = %5.3f, pos = %s, h = %5.3f" % (time, str(r.pos), r.heading))
    r.step(step)
    time += step


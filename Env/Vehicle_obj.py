import numpy as np

class Vehicle:

    def __init__(self):
        self.speed = 0
        self.location = 100
        self.time_to_reach = 0
        self.max_speed = 10

    def step(self, acceleration):
        done = False
        reward = 0
        info = {'successful':False}

        self.speed += acceleration

        if self.speed < 0:
            self.speed = 0
        if self.speed > self.max_speed:
            self.speed = self.max_speed

        self.location -= self.speed
        self.time_to_reach -= 1

        if (self.location <= 0 or self.time_to_reach == 0):
            done = True

        if done:
            if (self.location < 2 and self.time_to_reach < 2):
                reward = 10 + 0.8*self.speed
                info = {'successful':True}
            else:
                reward = -10
        else:
            reward = -(self.location/100)


        return [self.speed, self.time_to_reach, self.location], reward, done, info

    def reset(self):
        self.time_to_reach = np.random.randint(20,30)
        self.speed = 0
        self.location = 100
        return [self.speed, self.time_to_reach, self.location]


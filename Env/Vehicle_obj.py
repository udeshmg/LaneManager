import numpy as np

class Vehicle:

    def __init__(self):
        self.speed = 0
        self.location = 400
        self.time_to_reach = 0
        self.max_speed = 22
        self.max_acc = 2



    def step(self, acceleration):
        done = False
        reward = 0
        info = {'is_success':False}

        self.speed += acceleration*self.max_acc*(1 - (self.speed/self.max_speed)**4)

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
                reward = 0.5*self.speed
                info = {'is_success':True}
            else:
                reward = -10
        else:
            reward = -(self.location/400)


        return [int(self.speed), self.time_to_reach, self.location], reward, done, info

    def reset(self):
        self.time_to_reach = np.random.randint(30,45)
        self.speed = 0
        self.location = 400
        return [self.speed, self.time_to_reach, self.location]


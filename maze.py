import gym
import numpy as np
from os import system
from time import sleep
from gym import error, spaces, utils
from gym.utils import seeding
from matplotlib import pyplot as plt

def build_Maze(obs):
    a=np.zeros((10,10))
    for j in obs:
        a[j]=1
    return a
        
class Maze(gym.Env):
    def __init__(self):
        super(Maze, self).__init__()
        self.actions=[(-1,0),(1,0),(0,1),(0,-1)]
        self.mean=['U','D','R','L']
        self.action_space=spaces.Discrete(len(self.actions))
        self.observation_space=spaces.Box(low=0,high=4,shape=(5,5))
        self.obs=[(0,2),(0,7),(0,8),(1,1),(1,2),(1,6),(2,1),(2,6),(2,7),
                  (3,1),(3,6),(4,1),(4,5),(4,8),(5,6),(6,1),(6,4),(6,8),(6,9),
                  (7,2),(7,3),(7,5),(8,2),(8,3),(8,5),(8,7),(8,8),(8,9),(9,5)]
        self.start=(9,0)
        self.goal=(0,9)
        self.maze=build_Maze(self.obs)
        
        self.state=None
        self.steps_beyond_me=None
        self.reset()
        
    
    def reset(self):
        self.maze=build_Maze(self.obs)
        self.state = (9,0)
        self.steps_beyond_done = None
        self.done = False
        return self.state
    
    
    def step(self, action):
        x=self.state[0]+self.actions[action][0]
        y=self.state[1]+self.actions[action][1]
        
        if x<0 or x>9 or y<0 or y>9:
            reward=-1
            done=False
        elif (x,y) in self.obs:
            reward=-1
            done=False
        elif (x,y)==self.goal:
            self.state=(x,y)
            reward=0
            done=True
        else:
            self.state=(x,y)
            reward=-1
            done=False
            
        return self.state, reward, done, {}
    
    def render(self,episode,t):

        _=system('clear')
        x,y=self.state
        a=np.array(self.maze)
        print('-----')
        for i in range(10):
            for j in range(10):
                if (x,y)==(i,j):
                    print('X', end='')
                else:
                    print(int(a[i,j]), end='')
            print()
        print('-----')
        a[self.start] = 4
        a[x,y]=8
        a[self.goal]=4
        plt.clf()
        plt.title('Episode: '+str(episode)+' Step: '+str(t))
        plt.ion()
        plt.draw()
        plt.imshow(a)
        plt.pause(0.001)


        
if __name__=='__main__':
    env=Maze()
    for i_episode in range(20):
        state = env.reset()
        print('Restart')
        env.render()
        for t in range(100):
            action = env.action_space.sample()
            print(env.mean[action])
            state, reward, done, info = env.step(action)
            env.render()
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

    
    


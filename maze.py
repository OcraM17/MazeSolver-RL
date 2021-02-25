import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

def build_Maze(obs):
    a=np.zeros((5,5))
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
        self.obs=[(0,2),(1,4),(2,2),(3,1),(3,3),(4,2)]
        self.start=(0,0)
        self.goal=(4,4)
        self.maze=build_Maze(self.obs)
        
        self.state=None
        self.steps_beyond_me=None
        self.reset()
        
    
    def reset(self):
        self.maze=build_Maze(self.obs)
        self.state = (0,0)
        self.steps_beyond_done = None
        self.done = False
        return self.state
    
    
    def step(self, action):
        x=self.state[0]+self.actions[action][0]
        y=self.state[1]+self.actions[action][1]
        
        if x<0 or x>4 or y<0 or y>4:
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
    
    def render(self):
        x,y=self.state
        a=np.array(self.maze)
        print('-----')
        for i in range(5):
            for j in range(5):
                if (x,y)==(i,j):
                    print('X', end='')
                else:
                    print(int(a[i,j]), end='')
            print()
        print('-----')


        
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

    
    


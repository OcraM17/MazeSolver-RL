import numpy as np
from maze import Maze
from tqdm import tqdm
from QAgent import QAgent
from ExpectedSarsa import ExpectedSarsaAgent
from matplotlib import pyplot as plt
def training():
    agent_info = {"num_actions": 4, "num_states": 25, "epsilon": 0.1, "step_size": 0.5, "discount": 1.0,"seed":5}
    env = Maze()
    agents=[QAgent(agent_info),ExpectedSarsaAgent(agent_info)]
    rew_all_agents=[]
    for c,agent in enumerate(agents):
        rew_agents = []
        for run in tqdm(range(100)):
            rew_history = []
            for i_episode in range(500):
                state = env.reset()
                env.render()
                action = agent.agent_start(state)
                ep_rew=0
                for t in range(100):
                    state, reward, done, info = env.step(action)
                    ep_rew+=reward
                    if done:
                        env.render()
                        #print("Episode finished after {} timesteps".format(t + 1))
                        agent.agent_end(reward)
                        break
                    action=agent.agent_step(reward, state)
                    env.render()
                rew_history.append(ep_rew)
            rew_agents.append(rew_history)
        rew_all_agents.append(rew_agents)

    plt.figure(1)
    for c,j in enumerate(['Q-Learning','ExpectedSarsa']):
        plt.plot(np.mean(rew_all_agents[c],axis=0),label=j)
    plt.xlabel("Episodes")
    plt.ylabel("Sum of\n rewards\n during\n episode", rotation=0, labelpad=40)
    plt.xlim(0, 500)
    plt.ylim(-9.5, -7.5)
    plt.legend()
    plt.show()

if __name__=='__main__':
    training()



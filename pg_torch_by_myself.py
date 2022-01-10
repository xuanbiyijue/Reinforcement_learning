import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time

#构建神经网络，此神经网络输出的是在s状态下执行动作的概率
class PGNet(nn.Module):
    def __init__(self, state_dim, action_dim):         #state_dim是输入的维度，表示状态；action_dim是输出，表示动作
        super(PGNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24,action_dim)
        )
        #self.mls = nn.MSELoss()
        self.opt = torch.optim.Adam(self.parameters(), lr = 0.01)

    def forward(self, inputs):
        return self.fc(inputs)

GAMMA = 0.95      #衰减率

#策略梯度算法
class PG():
    def __init__(self, env):
        # 状态空间和动作空间的维度
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        #存储库
        self.states = []
        self.actions = []
        self.rewards = []

        #搭建网络
        self.net = PGNet(self.state_dim, self.action_dim)

    def choose_actions(self, state):
        state = torch.FloatTensor(state)
        net_out = self.net.forward(state)
        with torch.no_grad():
            prob_weights = F.softmax(net_out, dim=0).data.numpy()
        #依概率从[0,1]选择
        action = np.random.choice(range(prob_weights.shape[0]), p=prob_weights)
        return action

    # 将状态，动作，奖励保存三个列表中
    def store_transition(self, s, a, r):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)

    def learn(self):
        #第一步：计算未来收益
        discounted_rewards = np.zeros_like(self.rewards)
        running_add = 0
        for t in reversed(range(0, len(self.rewards))):
            running_add = running_add * GAMMA + self.rewards[t]
            discounted_rewards[t] = running_add

        #标准化数据
        discounted_rewards -= np.mean(discounted_rewards)  # 减均值
        discounted_rewards /= np.std(discounted_rewards)  # 除以标准差
        discounted_rewards = torch.FloatTensor(discounted_rewards)

        #第二步：前向传播
        softmax_input = self.net.forward(torch.FloatTensor(self.states))
        print("input: %s"%str(softmax_input))
        print("actions: %s"%(self.actions))
        neg_log_prob = F.cross_entropy(input=softmax_input, target=torch.LongTensor(self.actions), reduction='none')
        print("prob: %s"%neg_log_prob)

        #第三步：反向传播
        loss = torch.mean(neg_log_prob * discounted_rewards)
        self.net.opt.zero_grad()
        loss.backward()
        self.net.opt.step()

        # 每次学习完后清空数组
        self.states, self.actions, self.rewards = [], [], []



#____________________________________
EPISODE = 3000
STEP = 300
TEST = 10
ENV_NAME = 'CartPole-v0'
env = gym.make(ENV_NAME)      #环境
agent = PG(env)              #智能体
for episode in range(EPISODE):
    state = env.reset()
    for step in range(STEP):
            action = agent.choose_actions(state)  # softmax概率选择action
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward)   # 新函数 存取这个transition
            state = next_state
            if done:
                # print("stick for ",step, " steps")
                agent.learn()   # 更新策略网络
                break

    # Test every 100 episodes
    if episode % 100 == 0:
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                env.render()
                action = agent.choose_actions(state)  # direct action for test
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward/TEST
        print ('episode: ', episode, 'Evaluation Average Reward:', ave_reward)

















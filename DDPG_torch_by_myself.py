'''
环境介绍：
1、连续状态域
n       Observation min         max
0       cos         -1.0        1.0
1       sin         -1.0        1.0
2       角速度      -8.0        8.0

2、连续动作域
n       action      min         max
0       角加速度    -2.0        2.0
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import gym


#ActorNet
class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, action_dim),
            nn.Tanh()
        )
        self.opt = torch.optim.Adam(self.parameters(), lr = 0.01)

    def forward(self, inputs):
        return self.fc(inputs) * 2    # for the game "Pendulum-v0", action range is [-2, 2]


#CriticNet
class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNet, self).__init__()
        self.linear1 = nn.Linear(state_dim+1, 24)
        self.linear2 = nn.Linear(24, 24)
        self.linear3 = nn.Linear(24, action_dim)

        self.opt = torch.optim.Adam(self.parameters(), lr = 0.01)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


STORE_SIZE = 3000        #存储库大小
BATCH_SIZE = 1000        #小批量大小
GAMMA = 0.9              #收益衰减率
TAU = 0.02

#DDPG算法
class DDPG():
    def __init__(self, env):
        #状态空间和动作空间的维度
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        #创建存储库
        self.store = np.zeros((STORE_SIZE, (self.state_dim * 2 + 2)))  # 初始化buffer 列中储存 s, a, s_, r
        self.store_count = 0

        #创建4个网络
        self.actor = ActorNet(self.state_dim, self.action_dim)
        self.actor_target = ActorNet(self.state_dim, self.action_dim)
        self.critic = CriticNet(self.state_dim, self.action_dim)
        self.critic_target = CriticNet(self.state_dim, self.action_dim)
        # 创建两个优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.003)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.003)
        #定义损失函数
        self.loss_func = nn.MSELoss()

    #存储数据，将状态，动作，奖励保存三个列表中
    def store_data(self, s, a, s_, r):
        #超过3000后覆盖
        self.store[self.store_count % STORE_SIZE][0:3] = s
        self.store[self.store_count % STORE_SIZE][3:4] = a
        self.store[self.store_count % STORE_SIZE][4:7] = s_
        self.store[self.store_count % STORE_SIZE][7:8] = r
        self.store_count += 1

    def get_action(self, state):
        s = torch.unsqueeze(torch.FloatTensor(state), 0)
        a = self.actor(s).detach()
        return a

    def learn(self):
        #当存够了数据
        if(self.store_count > STORE_SIZE):
            index = random.randint(0, STORE_SIZE - BATCH_SIZE -1)      #从0到999随机取一个
            #取出训练数据
            batch_s  = torch.Tensor(self.store[index:index + BATCH_SIZE, 0:3])
            batch_a  = torch.Tensor(self.store[index:index + BATCH_SIZE, 3:4])
            batch_s_ = torch.Tensor(self.store[index:index + BATCH_SIZE, 4:7])
            batch_r  = torch.Tensor(self.store[index:index + BATCH_SIZE, 7:8]).view(BATCH_SIZE,-1)

            def actor_learn():
                loss = -torch.mean( self.critic(batch_s, self.actor(batch_s)) )
                self.actor_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()

            def critic_learn():
                a_ = self.actor_target(batch_s_).detach()
                y_true = batch_r + GAMMA * self.critic_target(batch_s_, a_).detach()

                y_pred = self.critic(batch_s, batch_a)

                loss_fn = nn.MSELoss()
                loss = loss_fn(y_pred, y_true)
                self.critic_optimizer.zero_grad()
                loss.backward()
                self.critic_optimizer.step()

            def soft_update(net_target, net, tau):
                for target_param, param  in zip(net_target.parameters(), net.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

            critic_learn()
            actor_learn()
            soft_update(self.critic_target, self.critic, TAU)
            soft_update(self.actor_target, self.actor, TAU)


env = gym.make('Pendulum-v0')
agent = DDPG(env)

for episode in range(1000):
    s = env.reset()
    episode_reward = 0

    for step in range(500):
        env.render()
        a = agent.get_action(s)
        s_, r, done, _ = env.step(a)
        agent.store_data(s, a, s_, r)

        episode_reward += r
        s = s_

        agent.learn()

    print(episode, ': ', episode_reward)






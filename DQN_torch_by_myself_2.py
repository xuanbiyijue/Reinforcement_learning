import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

#搭建QNet，此神经网络的输出是在s状态下各动作的Q值
class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, action_dim)
        )

    def forward(self, inputs):
        return self.fc(inputs)

STORE_SIZE = 2000        #存储库大小
BATCH_SIZE = 1000        #小批量大小
GAMMA = 0.9              #收益衰减率
EPSILON = 0.9

#DQN算法
class DQN():
    def __init__(self, env):
        #状态空间和动作空间的维度
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        #创建存储库
        self.store = np.zeros((STORE_SIZE, (self.state_dim * 2 + 2)))  # 初始化buffer 列中储存 s, a, s_, r
        self.store_count = 0

        #创建网络
        self.net = QNet(self.state_dim, self.action_dim)
        self.target_net = QNet(self.state_dim, self.action_dim)
        self.optim = torch.optim.Adam(self.net.parameters(), 0.01)

        self.target_net.load_state_dict(self.net.state_dict())  #target网络加载net网络的模型参数
        self.loss_func = nn.MSELoss()                           #损失函数

    #存储数据，将状态，动作，奖励保存三个列表中
    def store_data(self, s, a, s_, r):
        #超过2000后覆盖
        self.store[self.store_count % STORE_SIZE][0:4] = s
        self.store[self.store_count % STORE_SIZE][4:5] = a
        self.store[self.store_count % STORE_SIZE][5:9] = s_
        self.store[self.store_count % STORE_SIZE][9:10] = r

    #选择动作
    def choose_actions(self, state):
        state = torch.FloatTensor(state)
        net_out = self.net(state).detach()
        with torch.no_grad():
            prob_weights = F.softmax(net_out, dim=0).data.numpy()
        #依概率从[0,1]选择
        action = np.random.choice(range(prob_weights.shape[0]), p=prob_weights)
        return action

    def choose_actions2(self, state):
        eps_threshold = random.random()
        action = self.net(torch.Tensor(state))
        #print(action)
        if eps_threshold < EPSILON:
            choice = torch.argmax(action).numpy()
        else:
            choice = np.random.randint(0, action.shape[0])  # 随机[0, action.shape[0]]之间的数
        return choice


    def learn(self):
        #当存够了2000条数据
        if self.store_count > STORE_SIZE:
            index = random.randint(0, STORE_SIZE - BATCH_SIZE -1)      #从0到999随机取一个
            #取出训练数据
            batch_s  = torch.Tensor(self.store[index:index + BATCH_SIZE, 0:4])
            batch_a  = torch.Tensor(self.store[index:index + BATCH_SIZE, 4:5]).long()
            batch_s_ = torch.Tensor(self.store[index:index + BATCH_SIZE, 5:9])
            batch_r  = torch.Tensor(self.store[index:index + BATCH_SIZE, 9:10])

            #正向传播
            q_next = self.target_net(batch_s_).detach().max(1)[0].reshape(BATCH_SIZE, 1)
            tq = batch_r + GAMMA * q_next                    #现实收益
            q = self.net(batch_s).gather(1, batch_a)         #预测收益

            #反向传播
            loss = self.loss_func(q, tq)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

#————————————————————————————
EPISODE = 3000         #回合数
TARGET_UPDATE = 20

env = gym.make('CartPole-v0')
env = env.unwrapped
agent = DQN(env)

avg_r = 0

for episode in range(EPISODE):
    s = env.reset()
    # print("s:",s)
    total_r = 0  # 每个episode的总reward
    while True:
        env.render()
        a = agent.choose_actions2(s)
        s_, r, done, _ = env.step(a)

        # modify the reward
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2
        #r = ( env.theta_threshold_radians - abs(s_[2]) ) / env.theta_threshold_radians * 0.7  +  ( env.x_threshold - abs(s_[0]) ) / env.x_threshold * 0.3   #r是即时奖励，神经网络需要学习持续奖励
        total_r += r  # 计算当前episode的总reward

        #保存得到的数据
        agent.store_data(s, a, s_, r)
        agent.store_count += 1

        s = s_
        if(agent.store_count > STORE_SIZE):
            #print("start study")
            if episode % TARGET_UPDATE == 0:
                agent.target_net.load_state_dict(agent.net.state_dict())
            agent.learn()
        if done:
            avg_r = avg_r + 1 / (episode + 1) * (total_r - avg_r)
            print('Episode ', episode, ' tot_reward: ', total_r, ' average_reward: ',avg_r)
            break

    if episode % TARGET_UPDATE == 0:
        agent.target_net.load_state_dict(agent.net.state_dict())

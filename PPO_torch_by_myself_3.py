'''
PPO算法也是Actor-Critic架构，但是与DDPG不同，PPO为on-policy算法，所以不需要设计target网络，也不需要ReplayBuffer，
并且Actor和Critic的网络参数可以共享以便加快学习。
PPO引入了重要度采样，使得每个episode的数据可以被多训练几次（实际的情况中，采样可能非常耗时）从而节省时间，
clip保证的更新的幅度不会太大。
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
import random
import gym

#定义网络。此网络为actor和critic的结合
class Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Net, self).__init__()
        self.linear_1 = nn.Linear(state_dim, 64)
        self.linear_actor = nn.Linear(64, action_dim)   #输出动作
        self.linear_critic = nn.Linear(64, 1)           #输出Q值

    def actor_forward(self, state, softmax_dim):
        s = F.relu( self.linear_1(state) )
        prob = F.softmax( self.linear_actor(s), dim=softmax_dim )    #输出动作的概率
        return prob

    def critic_forward(self, state):
        s = F.relu( self.linear_1(state) )
        return self.linear_critic(s)                 #输出Q值

GAMMA = 0.95
LAMBDA = 0.95
EPS_CLIP = 0.1

#PPO算法
class PPO():
    def __init__(self, env):
        #状态空间和动作空间的维度
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        #构建网络
        self.net = Net(self.state_dim, self.action_dim)
        #优化器
        self.optim = optim.Adam(self.net.parameters(), 0.003)

        #创建buffer
        self.data = []

    #把交互数据存入buffer
    def put_data(self, transition):
        self.data.append(transition)

    #将数据形成batch
    def make_batch(self):
        list_s = []
        list_a = []
        list_r = []
        list_s_ = []
        list_prob_a = []
        list_done = []

        for transition in self.data:
            s, a, r, s_, prob_a, done = transition
            list_s.append(s)
            list_a.append([a])
            list_r.append([r])
            list_s_.append(s_)
            list_prob_a.append([prob_a])
            done_mask = 0 if done else 1
            list_done.append([done_mask])

        s = torch.tensor(list_s, dtype=torch.float)
        a = torch.tensor(list_a)
        r = torch.tensor(list_r)
        s_ = torch.tensor(list_s_, dtype=torch.float)
        done_mask = torch.tensor(list_done, dtype=torch.float)
        prob_a = torch.tensor(list_prob_a)

        self.data = []            #清空数组
        return s, a, r, s_, done_mask, prob_a

    def learn(self):
        s, a, r, s_, done_mask, prob_a = self.make_batch()
        for i in range(3):
            #计算td_error误差，模型目标就是减少td_error
            td_target = r + GAMMA * self.net.critic_forward(s_) * done_mask
            delta = td_target - self.net.critic_forward(s)
            delta = delta.detach().numpy()

            #计算advantage，即当前策略比一般策略（baseline）要好多少
            #policy的优化目标就是让当前策略比baseline尽量好，但每次更新时又不能偏离太多，所以后面会有个clip
            list_advantage = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = GAMMA * LAMBDA * advantage + delta_t[0]
                list_advantage.append([advantage])
            list_advantage.reverse()
            advantage = torch.tensor(list_advantage, dtype=torch.float)

            #计算ratio，防止单词更新偏离太多
            pi = self.net.actor_forward(s, softmax_dim=1)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))    # a/b = exp( log(a) - log(b) )

            #计算clip，保证ratio在（1-eps_clip, 1+eps_clip)范围内
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-EPS_CLIP, 1+EPS_CLIP) * advantage
            #这里简化ppo，把policy loss和value loss放在一起计算
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.net.critic_forward(s), td_target.detach())

            self.optim.zero_grad()
            loss.mean().backward()
            self.optim.step()

T_horizon = 20

#主函数：简化ppo 这里先交互T_horizon个回合然后停下来学习训练，再交互，这样循环10000次
def main():
    #创建倒立摆环境
    env = gym.make('CartPole-v1')
    agent = PPO(env)
    average_reward = 0

    #主循环
    for episode in range(10000):
        s = env.reset()
        tot_reward = 0
        while True:
            env.render()
            prob = agent.net.actor_forward(torch.from_numpy(s).float(), 0)
            a = int(prob.multinomial(1))
            s_, r, done, _ = env.step(a)

            # modify the reward
            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2
            r = float(r)

            rate = prob[a].item()
            #保存数据
            agent.put_data( (s, a, r, s_, rate, done) )
            s = s_
            tot_reward += r

            if done:
                average_reward = average_reward + 1 / (episode + 1) * (
                        tot_reward - average_reward)
                if episode % 20 == 0:
                    print('Episode ', episode,' tot_reward: ', tot_reward, ' average_reward: ',average_reward)
                break
        # Agent.train_net()
        agent.learn()

main()







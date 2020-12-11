#Мультиагентное обучение с подкреплением
#Глава 3. Нейросетевое обучение
#Алгоритм VDN
#Тестирование алгоритма VDN

from smac.env import StarCraft2Env
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

np.set_printoptions(threshold=np.inf)


HIDDEN_SIZE = 66 
LSTM_HIDDEN_LAYER_SIZE = 60 

class Q_network(nn.Module):
    
    def __init__(self, batch_size, obs_size, lstm_hidden_layer_size, hidden_size, n_actions):
        super(Q_network, self).__init__()
              
        self.first_layer = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU())
        
        self.hidden_layer_size = lstm_hidden_layer_size
        
        self.hidden_cell = (torch.zeros(1, batch_size, self.hidden_layer_size).to("cpu"),
                            torch.zeros(1, batch_size, self.hidden_layer_size).to("cpu"))
        
        self.second_layer = nn.LSTM(input_size=hidden_size, hidden_size=self.hidden_layer_size, num_layers=1, batch_first=True)
        
        self.linear_advantage = nn.Sequential(nn.Linear(self.hidden_layer_size, n_actions))
        
        self.linear_value = nn.Sequential(nn.Linear(self.hidden_layer_size, 1))
   
    def forward(self, x):
        
        first_layer_out = self.first_layer(x)
        
        lstm_out, self.hidden_cell = self.second_layer(first_layer_out.view(len(first_layer_out), 1, -1), self.hidden_cell)
        
        lstm_out = F.relu(lstm_out)
        
        aofa_out = self.linear_advantage(lstm_out.view(len(first_layer_out), -1))
        
        vofs_out = self.linear_value(lstm_out.view(len(first_layer_out), -1))
               
        return vofs_out + (aofa_out - aofa_out.mean(dim=1, keepdim=True))


def select_actionFox(action_probabilities, avail_actions_ind, epsilon):
    
    for ia in action_probabilities:
            action = np.argmax(action_probabilities)
            if action in avail_actions_ind:
                
                return action
            else:
                
                action_probabilities[action] = 0
    
def main():
    
    env = StarCraft2Env(map_name="3ps1zgWallFOX", difficulty="1")
    env_info = env.get_env_info()
    obs_size =  env_info.get('obs_shape')
    
    print ("obs_size=",obs_size)
    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    
    n_episodes = 50 
    
    epsilon = 0
    
    batch_size = 1
   
    Reward_History = []
    Total_Reward = 0
    Mean_Reward_History = []
  
    total_rewards = []
    m_reward = []
     
    #создаем основную Q-сеть
    q_network0 = Q_network(batch_size, obs_size, LSTM_HIDDEN_LAYER_SIZE, HIDDEN_SIZE, n_actions)
    q_network1 = Q_network(batch_size, obs_size, LSTM_HIDDEN_LAYER_SIZE, HIDDEN_SIZE, n_actions)
    q_network2 = Q_network(batch_size, obs_size, LSTM_HIDDEN_LAYER_SIZE, HIDDEN_SIZE, n_actions)
    
    q_network0.eval()
    q_network1.eval()
    q_network2.eval()
    
    state0 = torch.load("qnet_0.dat", map_location=lambda stg, _: stg)
    q_network0.load_state_dict(state0)
    state1 = torch.load("qnet_1.dat", map_location=lambda stg, _: stg)
    q_network1.load_state_dict(state1)
    state2 = torch.load("qnet_2.dat", map_location=lambda stg, _: stg)
    q_network2.load_state_dict(state2)    
   
    print(q_network0)
    
    #Цикл по эпизодам
    ################for по эпизодам############################################
    for e in range(n_episodes):
        
        env.reset()
        
        terminated = False
        episode_reward = 0
        
        #Цикл - шаги игры внутри эпизода
        ######################цикл while#######################################
        while not terminated:
            
            actions = []
            action = 0
            
            actionsFox = np.zeros([n_agents]) 
            
            obs_agent = np.zeros([n_agents], dtype=object) 
            
            #Цикл по агентам
            for agent_id in range(n_agents):
                
                obs_agent[agent_id] = env.get_obs_agent(agent_id)
                
                obs_agentT = torch.FloatTensor([obs_agent[agent_id]])
                
                if agent_id == 0: 
                    q_network0.hidden_cell = (torch.zeros(1, batch_size, LSTM_HIDDEN_LAYER_SIZE), torch.zeros(1, batch_size, LSTM_HIDDEN_LAYER_SIZE))
                    action_probabilitiesT = q_network0(obs_agentT)
                elif agent_id == 1: 
                    q_network1.hidden_cell = (torch.zeros(1, batch_size, LSTM_HIDDEN_LAYER_SIZE), torch.zeros(1, batch_size, LSTM_HIDDEN_LAYER_SIZE))
                    action_probabilitiesT = q_network1(obs_agentT)
                elif agent_id == 2: 
                    q_network2.hidden_cell = (torch.zeros(1, batch_size, LSTM_HIDDEN_LAYER_SIZE), torch.zeros(1, batch_size, LSTM_HIDDEN_LAYER_SIZE))
                    action_probabilitiesT = q_network2(obs_agentT)
                
                action_probabilities = action_probabilitiesT.data.numpy()[0]
                
                avail_actions = env.get_avail_agent_actions(agent_id)
                
                avail_actions_ind = np.nonzero(avail_actions)[0]
               
                action = select_actionFox(action_probabilities, avail_actions_ind, epsilon)
                
                if action is None: 
                    
                    action = np.random.choice (avail_actions_ind)
                    
                actions.append(action)
                actionsFox[agent_id] = action
                ##############################################################

            reward, terminated, _ = env.step(actions)
            
            episode_reward += reward
 
        ######################цикл while#######################################
        print("Total reward in episode {} = {}".format(e, episode_reward))
        
        Reward_History.append(episode_reward)
        Total_Reward = Total_Reward + episode_reward
        Mean_Reward_History.append((Total_Reward/(e+1)))
        
        total_rewards.append(episode_reward)
        
        m_reward.append(np.mean(total_rewards))
        
        
    ################for по эпизодам############################################
    
    #Close StarCraft II
    env.close()
    
    print("MEAN reward = ", np.mean(m_reward))
    print ("Average reward = ", Total_Reward/n_episodes)
    print ("get_stats()=", env.get_stats())
    
    #Выводим на печать графики
    plt.figure(num=None, figsize=(6, 3), dpi=150, facecolor='w', edgecolor='k')
    plt.plot(Reward_History)
    plt.xlabel('Номер эпизода')
    plt.ylabel('Количество награды за эпизод')
    plt.show()
    
    plt.figure(num=None, figsize=(6, 3), dpi=150, facecolor='w', edgecolor='k')
    plt.plot(Mean_Reward_History)
    plt.xlabel('Номер эпизода')
    plt.ylabel('Средняя награда')
    plt.show()

    
if __name__ == "__main__":
    main()   
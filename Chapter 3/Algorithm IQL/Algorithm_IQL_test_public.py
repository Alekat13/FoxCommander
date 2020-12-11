#Мультиагентное обучение с подкреплением
#Глава 3. Нейросетевое обучение
#Тестирование алгоритма IQL

#Подключаем библиотеки
from smac.env import StarCraft2Env
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

#Вывод массива целиком
np.set_printoptions(threshold=np.inf)

class Q_network(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(Q_network, self).__init__()
        self.Q_network = nn.Sequential(
            nn.Linear(obs_size, 66),
            nn.ReLU(),
            nn.Linear(66, 60),
            nn.ReLU(),
            nn.Linear(60, n_actions)
        )
        self.sm_layer = nn.Softmax(dim=1)

    def forward(self, x):
        q_network_out = self.Q_network(x)
        sm_layer_out = self.sm_layer(q_network_out)
        return sm_layer_out

#Выбираем действие
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
    
    n_episodes = 10 #sudharsan - 800
    epsilon = 0
   
    Reward_History = []
    Total_Reward = 0
    Mean_Reward_History = []
    
    total_rewards = []
    m_reward = []
    
    #создаем основную Q-сеть
    q_network = Q_network(obs_size, n_actions)
    #создаем списки нейронных сетей для мультиагентного случая
    q_network_list = []
    
    for agent_id in range(n_agents):
        q_network_list.append(q_network)
        #загружаем обученные веса нейронной сети
        state = torch.load("qnet_%.0f.dat"%agent_id, map_location=lambda stg, _: stg)
        q_network_list[agent_id].load_state_dict(state)
    
    print(q_network_list)
    
    #Цикл по эпизодам
    ################for по эпизодам############################################
    for e in range(n_episodes):
        env.reset()
        terminated = False
        episode_reward = 0
        
        #Цикл - шаги игры внутри эпизода
        ######################цикл while#######################################
        while not terminated:
            state = env.get_state()
            #обнуляем промежуточные переменные
            actions = []
            action = 0
            #храним историю действий один шаг для разных агентов
            actionsFox = np.zeros([n_agents]) 
            #храним историю обзоров-состояний один шаг для разных агентов
            obs_agent = np.zeros([n_agents], dtype=object) 
            #Цикл по агентам
            for agent_id in range(n_agents):
                ##############################################################
                obs_agent[agent_id] = env.get_obs_agent(agent_id)
                obs_agentT = torch.FloatTensor([obs_agent[agent_id]])
                  
                action_probabilitiesT = q_network_list[agent_id](obs_agentT)
                action_probabilities = action_probabilitiesT.data.numpy()[0]
                
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
               
                action = select_actionFox(action_probabilities, avail_actions_ind, epsilon)
                if action is None: action = np.random.choice (avail_actions_ind)
                
                actions.append(action)
                actionsFox[agent_id] = action
                ##############################################################


            #Передаем действия агентов в среду, получаем награду и прерывание игры от среды
            reward, terminated, _ = env.step(actions)
            #суммируем награды за шаг для вычисления награды за эпизод
            episode_reward += reward
 
        ######################цикл while#######################################
        print("Total reward in episode {} = {}".format(e, episode_reward))
        
        Reward_History.append(episode_reward)
        Total_Reward = Total_Reward + episode_reward
        Mean_Reward_History.append((Total_Reward/(e+1)))
        
        total_rewards.append(episode_reward)
        m_reward.append(np.mean(total_rewards))#[-100:]))
        
    ################for по эпизодам############################################
    
    #Close StarCraft II
    env.close()
    
    print("Mean reward = ", np.mean(m_reward))
    print ("Get_stats()=", env.get_stats())
    
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
#Мультиагентное обучение с подкреплением
#Глава 3. Нейросетевое обучение
#Тестирование алгоритма MADDPG

from smac.env import StarCraft2Env
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random


#Вывод массива целиком
np.set_printoptions(threshold=np.inf)


# Сеть исполнителя
class MADDPG_Actor(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(MADDPG_Actor, self).__init__()
        
        self.MADDPG_Actor = nn.Sequential(
            nn.Linear(obs_size, 60),
            nn.ReLU(),
            nn.Linear(60, 60),
            nn.ReLU(),
            nn.Linear(60, 60),
            nn.ReLU(),
            nn.Linear(60, n_actions)
            )
        self.tanh_layer = nn.Tanh()#(dim=1)

    def forward(self, x):
        network_out = self.MADDPG_Actor(x)
        tanh_layer_out = self.tanh_layer(network_out)
        return tanh_layer_out


#выбираем действие
def select_actionFox(act_prob, avail_actions_ind, n_actions):
     #Выбираем действия в зависимости от вероятностей их выполнения
    for j in range(n_actions):
        actiontemp =  random.choices(['0','1','2','3','4','5','6'], weights=[act_prob[0],act_prob[1],act_prob[2],act_prob[3],act_prob[4],act_prob[5],act_prob[6]])
        action = int (actiontemp[0])
        if action in avail_actions_ind:
            return action
        else:
            act_prob[action] = 0

    
def sample_from_expbuf(experience_buffer, batch_size):
    #Функция random.permutation() возвращает случайную последовательность заданной длинны из его элементов 
    perm_batch = np.random.permutation(len(experience_buffer))[:batch_size]
    
    experience = np.array(experience_buffer)[perm_batch]
    
    return experience[:,0], experience[:,1], experience[:,2], experience[:,3], experience[:,4]      

def main():

    env = StarCraft2Env(map_name="3ps1zgWallFOX", difficulty="1")
 
    env_info = env.get_env_info()

    obs_size =  env_info.get('obs_shape')
    
    print ("obs_size=",obs_size)

    n_actions = env_info["n_actions"]

    n_agents = env_info["n_agents"]

    n_episodes = 50 
  
  
    #сохраняем историю для вывода графика
    Reward_History = []
    Total_Reward = 0
    Mean_Reward_History = []
    total_rewards = []
    m_reward = []
    
    #создаем основную сеть
    actor_network = MADDPG_Actor(obs_size, n_actions)
    #создаем списки нейронных сетей и буферов для мультиагентного случая
    actor_network_list = []
    
    for agent_id in range(n_agents):
        actor_network_list.append(actor_network)
        state = torch.load("actornet_%.0f.dat"%agent_id, map_location=lambda stg, _: stg)
        actor_network_list[agent_id].load_state_dict(state)
    
    print(actor_network_list)
    
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
            
            # обнуляем промежуточные переменные
            actions = []
            action = 0
            #храним историю действий один шаг для разных агентов
            actionsFox = np.zeros([n_agents]) 
            #храним историю обзоров-состояний один шаг для разных агентов
            obs_agent = np.zeros([n_agents], dtype=object) 
            
            #Цикл по агентам
            for agent_id in range(n_agents):

                obs_agent[agent_id] = env.get_obs_agent(agent_id)

                obs_agentT = torch.FloatTensor([obs_agent[agent_id]])

                action_probabilitiesT = actor_network_list[agent_id](obs_agentT)
   
                action_probabilities = action_probabilitiesT.data.numpy()[0]

                avail_actions = env.get_avail_agent_actions(agent_id)
           
                avail_actions_ind = np.nonzero(avail_actions)[0]
               
                action = select_actionFox(action_probabilities, avail_actions_ind, n_actions)

                if action is None:
                    action = np.random.choice (avail_actions_ind)
                
                #Собираем действия от разных агентов                 
                actions.append(action)
                actionsFox[agent_id] = action



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
        m_reward.append(np.mean(total_rewards))
        
    ################for по эпизодам############################################

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
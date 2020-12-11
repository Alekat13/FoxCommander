#Мультиагентное обучение с подкреплением
#Глава 3. Нейросетевое обучение
#Тестирование алгоритма CDQN

from smac.env import StarCraft2Env
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import cv2



MAPX = 32
MAPY = 32


#Вывод массива целиком
np.set_printoptions(threshold=np.inf)


class CDQN(nn.Module):
    
    def __init__(self, input_shape, n_actions):
        super(CDQN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
     
        conv_out_size = self._get_conv_out(input_shape)
         
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))    
    
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

#Выбираем действие
def select_actionFox(actionag, avail_actions_ind, epsilon):
   
    if actionag in avail_actions_ind:
        
        return actionag
    else:
        
        return np.random.choice (avail_actions_ind)
    
def sample_from_expbuf(experience_buffer, batch_size):
    #Функция random.permutation() возвращает случайную последовательность заданной длинны из его элементов. 
    perm_batch = np.random.permutation(len(experience_buffer))[:batch_size]
    
    experience = np.array(experience_buffer)[perm_batch]
    
    return experience[:,0], experience[:,1], experience[:,2], experience[:,3], experience[:,4]      

#Перекодируем выход нейронной сети 
def decode_actions(action_probabilities, n_agents, n_actions):
    actionsFox = np.zeros([n_agents])
    maxindex_out = np.argmax(action_probabilities)
    a1=0
    a2=0
    a3=0
    indexmain=0
    stop = 0
    if n_agents == 2:
        while stop == 0:
            for a1 in range(n_actions-1):
                for a2 in range(n_actions-1):
                    if maxindex_out == indexmain:
                        actionsFox[0] = a1
                        actionsFox[1] = a2
                        stop = 1
                        return actionsFox 
                    else: indexmain+=1
    elif n_agents == 3:
        while not stop:
            for a1 in range(n_actions-1):
                for a2 in range(n_actions-1):
                    for a3 in range(n_actions-1):
                        if maxindex_out == indexmain:
                            actionsFox[0] = a1
                            actionsFox[1] = a2
                            actionsFox[2] = a3
                            stop = 1
                            return actionsFox 
                        else: indexmain+=1
   

def main():
    env = StarCraft2Env(map_name="3ps1zgWallFOX", difficulty="1")
    
    env_info = env.get_env_info()
    
    obs_size =  env_info.get('obs_shape')
    
    print ("obs_size=",obs_size)
    
    n_actions = env_info["n_actions"]
    
    n_agents = env_info["n_agents"]
    
    n_episodes = 10 
    
    epsilon = 0
   
    Reward_History = []
    Total_Reward = 0
    Mean_Reward_History = []
    total_rewards = []
    m_reward = []
    
    
    #определим псевдокарту 
    map_data_NNinput = np.zeros((MAPX, MAPY, 1), np.uint8)
    print ('map_data_NNinput.shape=',map_data_NNinput.shape)
    #изменим псевдокарту для праивльной загрузки в есть
    map_data_NNinputR = map_data_NNinput.reshape(1, MAPX, MAPY)
    print ('map_data_NNinputR.shape=', map_data_NNinputR.shape)
    
    #вычислим выходной размер нейронной сети Q(s,a1,a2,a..,an)
    qofa_out = 1
    for i in range(n_agents):
        qofa_out = qofa_out*n_actions 
    
    #создаем основную Q-сеть
    q_network = CDQN(map_data_NNinputR.shape, qofa_out)
    #загружаем обученные параметры сети
    state = torch.load("qnetCNN.dat", map_location=lambda stg, _: stg)
    q_network.load_state_dict(state)
    #отключаеми обучение
    q_network.eval()
    
    print(q_network)
    
    #Цикл по эпизодам
    ################for по эпизодам############################################
    for e in range(n_episodes):
        
        env.reset()
        
        terminated = False
        episode_reward = 0
        
        #Цикл - шаги игры внутри эпизода
        ######################цикл while#######################################
        while not terminated:
            
            map_data = np.zeros((MAPY, MAPX, 3), np.uint8)
            
            #обнуляем промежуточные переменные
            actions = []
            action = 0
            #храним историю действий один шаг для разных агентов
            actionsFox = np.zeros([n_agents]) 
                      
            ###################################################################
            #соберем данные противников для псведокарты
            for e_id, e_unit in env.enemies.items():
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                #нарисуем агентов в виде окружностей на псведокарте BGR
                cv2.circle(map_data, (int(e_x), int(e_y)), 1, (255, 0, 0), -1) #(255, 0, 0) #(125, 125, 125)
            
            #соберем данные агентов для псведокарты
            for agent_id in range(n_agents):
                #соберем данные агентов для псведокарты
                #получаем характеристики юнита
                unit = env.get_unit_by_id(agent_id)
                #нарисуем агентов на псведокарте BGR
                cv2.circle(map_data, (int(unit.pos.x), int(unit.pos.y)), 1, (0, 255, 0), -1)  #(0, 255, 0) #(255, 255, 255)
            ###################################################################    
            
            #переводим изображение псевдокарты в полутоновый формат
            imggrayscale = cv2.cvtColor(map_data,cv2.COLOR_RGB2GRAY)
            map_data_NNinputR = imggrayscale.reshape(1, MAPX, MAPY)
            
            obs_agentT = torch.FloatTensor([map_data_NNinputR])
            action_probabilitiesT = q_network(obs_agentT)
            action_probabilitiesT = action_probabilitiesT.to("cpu")
            action_probabilities = action_probabilitiesT.data.numpy()[0]
            
            #Разделим общий выход Q(s,a1,a2,...,an) для каждого агента
            actionsFox = decode_actions (action_probabilities, n_agents, n_actions)
            
            #########################_Цикл по агентам для действия в игре_#####
            for agent_id in range(n_agents):
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                action = select_actionFox(actionsFox[agent_id], avail_actions_ind, epsilon)
                #проверяем на ошибку выхода сети
                if action is None:
                    action = np.random.choice (avail_actions_ind)
                #рисуем оранжевую линию на псевдокарте если агент стреляет
                if (action == 6) or (action==7) or (action==8):
                    cv2.line(map_data, (int(unit.pos.x+1), int(unit.pos.y)), (int(unit.pos.x+3), int(unit.pos.y)), (0, 130, 255), 1)
                
                #Собираем действия от разных агентов                 
                actions.append(action)
                actionsFox[agent_id] = action
            ###############_конец цикла по агентам для действия в игре_########


            #Передаем действия агентов в среду
            reward, terminated, _ = env.step(actions)
            #суммируем награды за шаг для вычисления награды за эпизод
            episode_reward += reward
            
            #нарисуем псведокарту
            ##################################################################
            #поворачиваем изображение псевдокарты
            flipped = cv2.flip(map_data, 0)
            #увеличиваем изображение псевдокарты для отображения
            resized = cv2.resize(flipped, dsize=None, fx=10, fy=10)
            cv2.imshow('PseudoMap', resized)
            cv2.waitKey(1)
            
            imggrayscale = cv2.cvtColor(map_data,cv2.COLOR_RGB2GRAY)
            map_data_NNinputR = imggrayscale.reshape(1, MAPX, MAPY)
            
        ######################цикл while#######################################
        print("Total reward in episode {} = {}".format(e, episode_reward))
        
        Reward_History.append(episode_reward)
        Total_Reward = Total_Reward + episode_reward
        Mean_Reward_History.append((Total_Reward/(e+1)))
        
        total_rewards.append(episode_reward)
        m_reward.append(np.mean(total_rewards))
                
    ################for по эпизодам############################################
    
    env.close()
    cv2.destroyAllWindows()
    
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
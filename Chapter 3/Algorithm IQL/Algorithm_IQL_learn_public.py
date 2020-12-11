#Мультиагентное обучение с подкреплением
#Глава 3. Нейросетевое обучение
#Алгоритм IQL 

#Подключаем библиотеки
from smac.env import StarCraft2Env
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.optim as optim

#Флаг вывода массива целиком
np.set_printoptions(threshold=np.inf)

#Определяем архитектуру нейронной сети
class Q_network(nn.Module):
    #На вход нейронная сеть получает состояние среды
    #На выходе нейронная сеть возвращает оценку действий в виде Q-значений
    def __init__(self, obs_size, n_actions):
        super(Q_network, self).__init__()
        self.Q_network = nn.Sequential(
            #Первый линейный слой обрабатывает входные данные состояния среды
            nn.Linear(obs_size, 66),
            nn.ReLU(),
            #Второй линейный слой обрабатывает внутренние данные 
            nn.Linear(66, 60),
            nn.ReLU(),
            #Третий линейный слой обрабатывает данные для оценки действий
            nn.Linear(60, n_actions)
        )
        #Применение к выходным данным функции Softmax
        self.sm_layer = nn.Softmax(dim=1)
    #Вначале данные x обрабатываются полносвязной сетью с функцией ReLU
    #На выходе происходит обработка функцией Softmax
    def forward(self, x):
        q_network_out = self.Q_network(x)
        sm_layer_out = self.sm_layer(q_network_out)
        #Финальный выход нейронной сети
        return sm_layer_out

#Выбираем возможное действие с максимальным Q-значением в зависимости от эпсилон
def select_actionFox(action_probabilities, avail_actions_ind, epsilon):
    p = np.random.random(1).squeeze()
    #Исследуем пространство действий
    if np.random.rand() < epsilon:
        return np.random.choice (avail_actions_ind) 
    else:
        #Находим возможное действие:
        #Проверяем есть ли действие в доступных действиях агента
        for ia in action_probabilities:
            action = np.argmax(action_probabilities)
            if action in avail_actions_ind:
                 return action
            else: 
                action_probabilities[action] = 0
                
#Создаем минивыборку определенного объема из буфера воспроизведения        
def sample_from_expbuf(experience_buffer, batch_size):
    #Функция возвращает случайную последовательность заданной длины из его элементов. 
    perm_batch = np.random.permutation(len(experience_buffer))[:batch_size]
    #Минивыборка
    experience = np.array(experience_buffer)[perm_batch]
    #Возвращаем значения минивыборки по частям
    return experience[:,0], experience[:,1], experience[:,2], experience[:,3], experience[:,4]       

#Основная функция программы
def main():
    
    #Загружаем среду Starcraft II, карту, сложность противника и расширенную награду 
    env = StarCraft2Env(map_name="3ps1zgWallFOX", reward_only_positive=False, reward_scale_rate=200, difficulty="1")
    #Получаем и выводим на печать информацию о среде
    env_info = env.get_env_info()
    print ('env_info=',env_info)
    #Получаем и выводим на печать размер наблюдений агента
    obs_size =  env_info.get('obs_shape')
    print ("obs_size=",obs_size)
    #Количество действий агента 
    n_actions = env_info["n_actions"]
    #количество дружественных агентов
    n_agents = env_info["n_agents"]
    
    #Определяем основные параметры нейросетевого обучения
    ###########################################################################   
    #Определим динамический эпсилон с затуханием
    eps_max = 1.0 #Начальное значение эпсилон
    eps_min = 0.1 #Финальное значение эпсилон
    eps_decay_steps = 15000 #Шаг затухания эпсилон
    #Основные переходы в алгоритме IQL зависят от управляющих параметров
    global_step = 0  #подсчитываем общее количество шагов в игре
    copy_steps = 100 #каждые 100 шагов синхронизируем нейронные сети 
    start_steps = 1000 #начинаем обучать через 1000 шагов
    steps_train = 4  #после начала обучения продолжаем обучть каждый 4 шаг 
    #Размер минивыборки    
    batch_size = 32     
    #Общее количество эпизодов игры
    n_episodes = 300 
    #Параметр дисконтирования
    gamma = 0.99 
    #Скорость обучения
    alpha = 0.01 
    #Объем буфера воспроизведения
    buffer_len = 10000 
    ###########################################################################   
    
    #Создаем буфер воспроизведения для каждого агента на основе обощенной очереди deque
    experience_buffer0 = deque(maxlen=buffer_len)
    experience_buffer1 = deque(maxlen=buffer_len)
    experience_buffer2 = deque(maxlen=buffer_len)
    #Pytorch определяет возможность использования графического процессора
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Определяем выходной размер нейронной сети 
    qofa_out = n_actions
    #Создаем основную нейронную сеть
    q_network = Q_network(obs_size, n_actions).to(device)
    #Создаем целевую нейронную сеть
    tgt_network = Q_network(obs_size, n_actions).to(device)
    #Создаем списки для мультиагентного случая
    q_network_list = []
    tgt_network_list = []
    optimizer_list = []
    objective_list = []
    for agent_id in range(n_agents):
        #Создаем список основных нейронных сетей для трех агентов
        q_network_list.append(q_network)
        #Создаем список целевых нейронных сетей для трех агентов
        tgt_network_list.append(tgt_network)
        #Создаем список оптимизаторов нейронных сетей для трех агентов
        optimizer_list.append(optim.Adam(params=q_network_list[agent_id].parameters(), lr=alpha))
        #Создаем список функций потерь для трех агентов
        objective_list.append(nn.MSELoss())
    #Выводим на печать списки основных нейронных сетей
    print ('q_network_list=', q_network_list)
    #Определяем вспомогательные параметры
    Loss_History = [] 
    Reward_History = []
    winrate_history = []
    total_loss = []
    m_loss = []
    
    #Основной цикл по эпизодам игры
    ################_цикл for по эпизодам_#####################################
    for e in range(n_episodes):
        #Перезагружаем среду
        env.reset()
        #Флаг окончания эпизода
        terminated = False
        #Награда за эпизод
        episode_reward = 0
        #Обновляем и выводим динамический эпсилон
        epsilon = max(eps_min, eps_max - (eps_max-eps_min) * global_step/eps_decay_steps)
        print ('epsilon=',epsilon)
     
        #Шаги игры внутри эпизода
        ######################_цикл while_#####################################
        while not terminated:
                      
            #Обнуляем промежуточные переменные
            actions = []
            action = 0
            #Храним историю действий один шаг для разных агентов
            actionsFox = np.zeros([n_agents]) 
            #Храним историю состояний среды один шаг для разных агентов
            obs_agent = np.zeros([n_agents], dtype=object) 
            obs_agent_next = np.zeros([n_agents], dtype=object)
            
            ##############_Цикл по агентам для выполнения действий в игре_#####
            for agent_id in range(n_agents):
                #Получаем состояние среды для независимого агента IQL
                obs_agent[agent_id] = env.get_obs_agent(agent_id)
                #Конвертируем данные в тензор
                obs_agentT = torch.FloatTensor([obs_agent[agent_id]]).to(device)
                #Передаем состояние среды в основную нейронную сеть 
                #и получаем Q-значения для каждого действия
                action_probabilitiesT = q_network_list[agent_id](obs_agentT)
                #Конвертируем данные в numpy
                action_probabilitiesT = action_probabilitiesT.to("cpu")
                action_probabilities = action_probabilitiesT.data.numpy()[0]
                #Находим возможные действия агента в данный момент времени 
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                #Выбираем возможное действие агента с учетом
                #максимального Q-значения и параметра эпсилон
                action = select_actionFox(action_probabilities, avail_actions_ind, epsilon)
                #Обрабатываем исключение при ошибке в возможных действиях
                if action is None: action = np.random.choice (avail_actions_ind)
                #Собираем действия от разных агентов                 
                actions.append(action)
                actionsFox[agent_id] = action
            ######_конец цикла по агентам для выполнения действий в игре_######

            #Передаем действия агентов в среду, получаем награду 
            #и прерывание игры от среды
            reward, terminated, _ = env.step(actions)
            #Суммируем награды за этот шаг для вычисления награды за эпизод
            episode_reward += reward
                        
            #####################_Цикл по агентам для обучения_################
            for agent_id in range(n_agents):
                #Получаем новое состояние среды
                obs_agent_next[agent_id] = env.get_obs_agent(agent_id)
                #Сохраняем переход в буфере воспроизведения для каждого агента 
                if agent_id == 0:
                    experience_buffer0.append([obs_agent[agent_id], actionsFox[agent_id], obs_agent_next[agent_id], reward, terminated])
                elif agent_id == 1:
                    experience_buffer1.append([obs_agent[agent_id], actionsFox[agent_id], obs_agent_next[agent_id], reward, terminated])
                elif agent_id == 2:
                    experience_buffer2.append([obs_agent[agent_id], actionsFox[agent_id], obs_agent_next[agent_id], reward, terminated])
                                
                #Если буфер воспроизведения наполнен, начинаем обучать сеть
                ########################_начало if обучения_###################
                if (global_step % steps_train == 0) and (global_step > start_steps):
                    #Получаем минивыборку из буфера воспроизведения
                    if agent_id == 0:
                        exp_obs, exp_act, exp_next_obs, exp_rew, exp_termd = sample_from_expbuf(experience_buffer0, batch_size)
                    elif agent_id == 1:
                        exp_obs, exp_act, exp_next_obs, exp_rew, exp_termd = sample_from_expbuf(experience_buffer1, batch_size)
                    elif agent_id == 2:
                        exp_obs, exp_act, exp_next_obs, exp_rew, exp_termd = sample_from_expbuf(experience_buffer2, batch_size)
                   
                    #Конвертируем данные в тензор
                    exp_obs = [x for x in exp_obs]
                    obs_agentT = torch.FloatTensor([exp_obs]).to(device)
                    
                    #Подаем минивыборку в основную нейронную сеть
                    #чтобы получить Q(s,a)
                    action_probabilitiesT = q_network_list[agent_id](obs_agentT)
                    action_probabilitiesT = action_probabilitiesT.to("cpu")
                    action_probabilities = action_probabilitiesT.data.numpy()[0]
                    
                    #Конвертируем данные в тензор
                    exp_next_obs = [x for x in exp_next_obs]
                    obs_agentT_next = torch.FloatTensor([exp_next_obs]).to(device)
                    
                    #Подаем минивыборку в основную нейронную сеть
                    #чтобы получить Q'(s',a')
                    action_probabilitiesT_next = tgt_network_list[agent_id](obs_agentT_next)
                    action_probabilitiesT_next = action_probabilitiesT_next.to("cpu")
                    action_probabilities_next = action_probabilitiesT_next.data.numpy()[0]
                    
                    #Вычисляем целевое значение y 
                    y_batch = exp_rew + gamma * np.max(action_probabilities_next, axis=-1)*(1 - exp_termd) 
                    
                    #Переформатируем y_batch размером batch_size
                    y_batch64 = np.zeros([batch_size, qofa_out])
                    for i in range (batch_size):
                        for j in range (qofa_out):
                            y_batch64[i][j] = y_batch[i]
                    #Конвертируем данные в тензор 
                    y_batchT = torch.FloatTensor([y_batch64])
                    
                    #Обнуляем градиенты
                    optimizer_list[agent_id].zero_grad()
                    
                    #Вычисляем функцию потерь
                    loss_t = objective_list[agent_id](action_probabilitiesT, y_batchT) 
                    
                    #Сохраняем данные для графиков
                    Loss_History.append(loss_t) 
                    loss_n=loss_t.data.numpy()
                    total_loss.append(loss_n)
                    m_loss.append(np.mean(total_loss[-1000:]))
                    
                    #Выполняем обратное распространение ошибки
                    loss_t.backward()
                    
                    #Выполняем оптимизацию нейронных сетей
                    optimizer_list[agent_id].step()
                
                ######################_конец if обучения_######################
                
                #Синхронизируем веса основной и целевой нейронной сети
                #каждые 100 шагов
                if (global_step+1) % copy_steps == 0 and global_step > start_steps:
                    tgt_network_list[agent_id].load_state_dict(q_network_list[agent_id].state_dict())
            
            #####################_Конец цикла по агентам для обучения_#########
            
            #Обновляем счетчик общего количества шагов
            global_step += 1
           
        ######################_конец цикла while###############################
        #Выводим счетчик шагов игры и общую награду за эпизод
        print('global_step=', global_step, "Total reward in episode {} = {}".format(e, episode_reward))
        
        #Собираем данные для графиков
        Reward_History.append(episode_reward)
        status = env.get_stats()
        winrate_history.append(status["win_rate"])
        
    ################_конец цикла по эпизодам игры_############################################
    
    #Закрываем среду StarCraft II
    env.close()
    
    #Сохраняем параметры обученных нейронных сетей
    for agent_id in range(n_agents):
        torch.save(q_network_list[agent_id].state_dict(),"qnet_%.0f.dat"%agent_id) 
        
    #Выводим на печать графики
    #Средняя награда
    plt.figure(num=None, figsize=(6, 3), dpi=150, facecolor='w', edgecolor='k')
    plt.plot(Reward_History)
    plt.xlabel('Номер эпизода')
    plt.ylabel('Количество награды за эпизод')
    plt.show()
    #Процент побед
    plt.figure(num=None, figsize=(6, 3), dpi=150, facecolor='w', edgecolor='k')
    plt.plot(winrate_history)
    plt.xlabel('Номер эпизода')
    plt.ylabel('Процент побед')
    plt.show()
    #Значения функции потерь
    plt.figure(num=None, figsize=(6, 3), dpi=150, facecolor='w', edgecolor='k')
    plt.plot(m_loss)
    plt.xlabel('Номер каждой 1000 итерации')
    plt.ylabel('Функция потерь')
    plt.show()

#Точка входа в программу  
if __name__ == "__main__":
    start_time = time.time()
    main()
    #Время обучения
    print("--- %s минут ---" % ((time.time() - start_time)/60))
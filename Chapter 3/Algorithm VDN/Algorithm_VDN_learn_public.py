#Мультиагентное обучение с подкреплением
#Глава 3. Нейросетевое обучение
#Алгоритм VDN 

#Подключаем библиотеки
from smac.env import StarCraft2Env
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#Флаг выводы массива целиком
np.set_printoptions(threshold=np.inf)

#Размер выхода первого полносвязного линейного слоя 
HIDDEN_SIZE = 66
#Размер выхода рекуррентного слоя LSTM
LSTM_HIDDEN_LAYER_SIZE = 60  

#Определяем архитектуру нейронной сети 
class VDN_network(nn.Module):
    #На вход нейронная сеть получает состояние среды
    #На выходе нейронная сеть возвращает оценку действий в виде Q-значений
    def __init__(self, batch_size, obs_size, lstm_hidden_layer_size, hidden_size, n_actions):
        super(VDN_network, self).__init__()
        #Первый линейный слой обрабатывает входные данные состояния среды        
        self.first_layer = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU())
        #Определяем выход рекуррентной сети LSTM
        self.hidden_layer_size = lstm_hidden_layer_size
        #Инициализируем внутреннее состояние LSTM
        self.hidden_cell = (torch.zeros(1, batch_size, self.hidden_layer_size).to("cuda"),
                            torch.zeros(1, batch_size, self.hidden_layer_size).to("cuda"))
        #Второй рекуррентный слой LSTM нейронной сети
        self.second_layer = nn.LSTM(input_size=hidden_size, hidden_size=self.hidden_layer_size, num_layers=1, batch_first=True)
        #Линейный слой, вычисляющий функцию преимущества
        self.linear_advantage = nn.Sequential(nn.Linear(self.hidden_layer_size, n_actions))
        #Линейный слой, вычисляющий функцию ценности состояния
        self.linear_value = nn.Sequential(nn.Linear(self.hidden_layer_size, 1))
    #Вначале данные x обрабатываются линейным, затем рекуррентным слоем
    #На выходе данные суммируются после дуэльной обработки линейными слоями
    def forward(self, x):
        #Результат после обработки данных первым слоем
        first_layer_out = self.first_layer(x)
        #Результат после обработки данных вторым слоем
        lstm_out, self.hidden_cell = self.second_layer(first_layer_out.view(len(first_layer_out), 1, -1), self.hidden_cell)
        #Применение функции ReLu
        lstm_out = F.relu(lstm_out)
        #Результат первого дуэльного слоя
        aofa_out = self.linear_advantage(lstm_out.view(len(first_layer_out), -1))
        #Результат второго дуэльного слоя
        vofs_out = self.linear_value(lstm_out.view(len(first_layer_out), -1))
        #Финальный выход нейронной сети суммирует результаты дуэльной обработки
        return vofs_out + (aofa_out - aofa_out.mean(dim=1, keepdim=True))
    
#Выбираем возможное действие с максимальным Q-значением в зависимости от 
#эпсилон
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
                
#Создаем последовательную минивыборку определенного объема из буфера воспроизведения 
def sample_sequence_from_expbuf(ind_begin, experience_buffer, big_batch_size):
    
    #Минивыборку инициализируем пустой
    experience_buffer_sequence = []
    
    #Последовательно выбираем значения (s,a,r,s') из буфера воспроизведения
    for i in range(big_batch_size):
        experience_buffer_sequence.append (experience_buffer[ind_begin])
        ind_begin+=1
      
    return experience_buffer_sequence       

#Основная функция программы
def main():
    #Загружаем среду Starcraft II, карту, сложность противника и расширенную  
    #награду 
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
    ########################################################################   
    #Определим динамический эпсилон с затуханием
    eps_max = 1.0 #Начальное значение эпсилон
    eps_min = 0.1 #Финальное значение эпсилон
    eps_decay_steps = 15000 #Шаг затухания эпсилон
    #Основные переходы в алгоритме VDN зависят от шагов игры
    global_step = 0 #подсчитываем общее количество шагов в игре
    copy_steps = 100 #каждые 100 шагов синхронизируем нейронные сети 
    start_steps = 1000 #начинаем обучать через 1000 шагов
    steps_train = 4 #после начала обучения продолжаем обучать каждый 4 шаг 
    #Размер минивыборки    
    batch_size = 1 
    #Размер минивыборки, моделируемый обучением с помощью цикла
    BIG_batch_size = 32
    #Общее количество эпизодов игры
    n_episodes = 215 
    #Параметр дисконтирования
    gamma = 0.99 
    #Скорость обучения
    alpha = 0.01 
    #Объем буфера воспроизведения
    buffer_len = 10000 
    #########################################################################  
    
    #Pytorch определяет возможность использования графического процессора
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #Определяем выходной размер нейронной сети 
    qofa_out = n_actions
    
    #Создаем буферы воспроизведения на основе deque для трех агентов
    experience_buffer0 = deque(maxlen=buffer_len)
    experience_buffer1 = deque(maxlen=buffer_len)
    experience_buffer2 = deque(maxlen=buffer_len)
    
    #Создаем основные нейронные сети для трех агентов
    q_network0 = VDN_network(batch_size, obs_size, LSTM_HIDDEN_LAYER_SIZE, HIDDEN_SIZE, n_actions).to(device)
    q_network1 = VDN_network(batch_size, obs_size, LSTM_HIDDEN_LAYER_SIZE, HIDDEN_SIZE, n_actions).to(device)
    q_network2 = VDN_network(batch_size, obs_size, LSTM_HIDDEN_LAYER_SIZE, HIDDEN_SIZE, n_actions).to(device)
    
    #Создаем целевые нейронные сети для трех агентов
    tgt_network0 = VDN_network(batch_size, obs_size, LSTM_HIDDEN_LAYER_SIZE, HIDDEN_SIZE, n_actions).to(device)
    tgt_network1 = VDN_network(batch_size, obs_size, LSTM_HIDDEN_LAYER_SIZE, HIDDEN_SIZE, n_actions).to(device)
    tgt_network2 = VDN_network(batch_size, obs_size, LSTM_HIDDEN_LAYER_SIZE, HIDDEN_SIZE, n_actions).to(device)
    
    #Выключаем обучение целевой сети  
    tgt_network0.eval()
    tgt_network1.eval()
    tgt_network2.eval()
    
    #Создаем оптимизаторы нейронных сетей для трех агентов
    optimizer0 = optim.Adam(params=q_network0.parameters(), lr=alpha)
    optimizer1 = optim.Adam(params=q_network1.parameters(), lr=alpha)
    optimizer2 = optim.Adam(params=q_network1.parameters(), lr=alpha)
    
    #Создаем функции потерь для трех агентов
    objective0= nn.MSELoss()
    objective1= nn.MSELoss()
    objective2= nn.MSELoss()
    
    #Выводим на печать архитектуру созданной нейронной сети
    print ('VDN_network=', q_network0)
    
    #Определяем вспомогательные параметры
    Loss_History = [] 
    Reward_History = []
    winrate_history = []
    total_loss = []
    m_loss = []
    
    #Основной цикл по эпизодам игры
    ################_цикл for по эпизодам_##################################
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
        ######################_цикл while_#################################
        while not terminated:
                       
            #Обнуляем промежуточные переменные
            actions = []
            action = 0
            #Храним историю действий один шаг для разных агентов
            actionsFox = np.zeros([n_agents]) 
            #Храним историю состояний среды один шаг для разных агентов
            obs_agent = np.zeros([n_agents], dtype=object) 
            obs_agent_next = np.zeros([n_agents], dtype=object)
            
            ###########_Цикл по агентам для выполнения действий в игре_#####
            for agent_id in range(n_agents):
                #Получаем состояние среды для каждого агента VDN
                obs_agent[agent_id] = env.get_obs_agent(agent_id)
                #Конвертируем данные в тензор
                obs_agentT = torch.FloatTensor([obs_agent[agent_id]]).to(device)
                  
                #Передаем состояние среды в основную нейронную сеть 
                #и получаем Q-значения для каждого действия
                if agent_id == 0: 
                    #Выключаем обучение
                    q_network0.eval()
                    #Инициализируем внутреннее состояние LSTM
                    q_network0.hidden_cell = (torch.zeros(1, batch_size, LSTM_HIDDEN_LAYER_SIZE).to(device), torch.zeros(1, batch_size, LSTM_HIDDEN_LAYER_SIZE).to(device))
                    #Получаем результат обработки данных нейронной сетью
                    action_probabilitiesT = q_network0(obs_agentT)
                elif agent_id == 1: 
                    q_network1.eval()
                    q_network1.hidden_cell = (torch.zeros(1, batch_size, LSTM_HIDDEN_LAYER_SIZE).to(device), torch.zeros(1, batch_size, LSTM_HIDDEN_LAYER_SIZE).to(device))
                    action_probabilitiesT = q_network1(obs_agentT)
                elif agent_id == 2: 
                    q_network2.eval()
                    q_network2.hidden_cell = (torch.zeros(1, batch_size, LSTM_HIDDEN_LAYER_SIZE).to(device), torch.zeros(1, batch_size, LSTM_HIDDEN_LAYER_SIZE).to(device))
                    action_probabilitiesT = q_network2(obs_agentT)   
                
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
                if action is None:
                    action = np.random.choice (avail_actions_ind)
                    
                #Собираем действия от разных агентов                 
                actions.append(action)
                actionsFox[agent_id] = action
            ####_конец цикла по агентам для выполнения действий в игре_######
     
            #Передаем действия агентов в среду, получаем награду 
            #и прерывание игры от среды
            reward, terminated, _ = env.step(actions)
            #Суммируем награды за этот шаг для вычисления награды за эпизод
            episode_reward += reward
            
            #####################_Цикл по агентам для обучения_##############
            for agent_id in range(n_agents):
                
                #Получаем новое состояние среды
                obs_agent_next[agent_id] = env.get_obs_agent(agent_id)
                
                #Сохраняем переход в буфере воспроизведения 
                if agent_id == 0:
                    experience_buffer0.append([obs_agent[agent_id], actionsFox[agent_id], obs_agent_next[agent_id], reward, terminated])
                elif agent_id == 1:
                    experience_buffer1.append([obs_agent[agent_id], actionsFox[agent_id], obs_agent_next[agent_id], reward, terminated])
                elif agent_id == 2:
                    experience_buffer2.append([obs_agent[agent_id], actionsFox[agent_id], obs_agent_next[agent_id], reward, terminated])
                        
            #Если буфер воспроизведения наполнен, начинаем обучать сеть
            ########################_начало if обучения_################
            if (global_step % steps_train == 0) and (global_step > start_steps):
                #Инициализируем переменную для хранения 
                #последовательной минивыборки из буфера воспроизведения
                experience_buffer_sequence = []
                #Находим текущий размер буфера воспроизведения
                lengthofexpbuf = len(experience_buffer0)
                #Генерим случайное число от 1 до размера буфера
                ind_begin = np.random.randint (1, (lengthofexpbuf - BIG_batch_size))
                
                for agent_id in range(n_agents):
                    if agent_id == 0:
                        #Выбираем из буфера воспроизведения 
                        #последовательную минивыборку 
                        experience_bufferTemp = sample_sequence_from_expbuf(ind_begin, experience_buffer0, BIG_batch_size)
                        experience_buffer_sequence.append(experience_bufferTemp)
                    elif agent_id == 1:
                        experience_bufferTemp = sample_sequence_from_expbuf(ind_begin, experience_buffer1, BIG_batch_size)
                        experience_buffer_sequence.append(experience_bufferTemp)
                    elif agent_id == 2:
                        experience_bufferTemp = sample_sequence_from_expbuf(ind_begin, experience_buffer2, BIG_batch_size)
                        experience_buffer_sequence.append(experience_bufferTemp)
                           
                #Моделируем минивыборку циклом
                #######_Цикл по модели минивыборки_#############################
                for batch_id in range(BIG_batch_size):
                    #Объявляем вспомогательные переменные
                    #Сумма Q-значений от трех нейронных сетей  
                    action_probabilities_next_sum = np.zeros([n_actions])
                    #Сумма выходов в которой будут отслеживаться градиенты 
                    #определенного агента чтобы выполнить обучение
                    action_probabilities_sumT0 = torch.zeros(1, n_actions)
                    action_probabilities_sumT1 = torch.zeros(1, n_actions)
                    action_probabilities_sumT2 = torch.zeros(1, n_actions)
                    #Храним общую сумму наград
                    exp_rew_sum = 0.0
                   
                    ###########_Цикл по агентам для суммирования Q-значений_###
                    for agent_id in range(n_agents):
                        #Разбираем на части последовательную минивыборку
                        exp_obs = experience_buffer_sequence[agent_id][batch_id][0]
                        exp_act = experience_buffer_sequence[agent_id][batch_id][1]
                        exp_next_obs = experience_buffer_sequence[agent_id][batch_id][2]
                        exp_rew = experience_buffer_sequence[agent_id][batch_id][3]
                        exp_termd = experience_buffer_sequence[agent_id][batch_id][4]
                                                
                        exp_rew_sum = exp_rew #exp_rew_sum + exp_rew
                       
                        #Конвертируем данные в тензор
                        exp_obs = [x for x in exp_obs]
                        obs_agentT = torch.FloatTensor([exp_obs]).to(device)
                        
                        #Подаем минивыборку в основную нейронную сеть
                        if agent_id == 0: 
                            #Включаем обучение
                            q_network0.train()
                            #Инициализируем внутреннее состояние LSTM основной сети
                            q_network0.hidden_cell = (torch.zeros(1, batch_size, LSTM_HIDDEN_LAYER_SIZE).to(device), torch.zeros(1, batch_size, LSTM_HIDDEN_LAYER_SIZE).to(device))
                            #Инициализируем внутреннее состояние LSTM целевой сети
                            tgt_network0.hidden_cell = (torch.zeros(1, batch_size, LSTM_HIDDEN_LAYER_SIZE).to(device), torch.zeros(1, batch_size, LSTM_HIDDEN_LAYER_SIZE).to(device))
                            #Подаем состояние в основную сеть чтобы получить Q(s,a)
                            action_probabilitiesT = q_network0(obs_agentT)
                        elif agent_id == 1:
                            q_network1.train()
                            q_network1.hidden_cell = (torch.zeros(1, batch_size, LSTM_HIDDEN_LAYER_SIZE).to(device), torch.zeros(1, batch_size, LSTM_HIDDEN_LAYER_SIZE).to(device))
                            tgt_network1.hidden_cell = (torch.zeros(1, batch_size, LSTM_HIDDEN_LAYER_SIZE).to(device), torch.zeros(1, batch_size, LSTM_HIDDEN_LAYER_SIZE).to(device))
                            action_probabilitiesT = q_network1(obs_agentT)
                        elif agent_id == 2:
                            q_network2.train()
                            q_network2.hidden_cell = (torch.zeros(1, batch_size, LSTM_HIDDEN_LAYER_SIZE).to(device), torch.zeros(1, batch_size, LSTM_HIDDEN_LAYER_SIZE).to(device))
                            tgt_network2.hidden_cell = (torch.zeros(1, batch_size, LSTM_HIDDEN_LAYER_SIZE).to(device), torch.zeros(1, batch_size, LSTM_HIDDEN_LAYER_SIZE).to(device))
                            action_probabilitiesT = q_network2(obs_agentT)
                        
                        #Конвертируем данные в numpy
                        action_probabilitiesT = action_probabilitiesT.to("cpu")
                        action_probabilities = action_probabilitiesT.data.numpy()[0]
                        
                        #Находим сумму Q-значений: Q=Q1+Q2+Q3
                        #Но при этом создаем три копии суммы, зависящие только 
                        #от градиентов определенного агента
                        if agent_id == 0:
                            action_probabilities_sumT0 = action_probabilities_sumT0 + action_probabilitiesT
                        elif agent_id == 1:
                            with torch.no_grad():
                                action_probabilities_sumN0 = action_probabilities_sumT0.data.numpy()[0]
                                action_probabilities_sumTnograd0 = torch.FloatTensor([action_probabilities_sumN0])
                            action_probabilities_sumT1 = action_probabilities_sumTnograd0 + action_probabilitiesT
                            
                            with torch.no_grad():
                                action_probabilities_sumN1 = action_probabilitiesT.data.numpy()[0]
                                action_probabilities_sumTnograd1 = torch.FloatTensor([action_probabilities_sumN1])
                            action_probabilities_sumT0 = action_probabilities_sumT0 + action_probabilities_sumTnograd1
                        
                        elif agent_id == 2:
                            with torch.no_grad():
                                action_probabilities_sumN0 = action_probabilities_sumT0.data.numpy()[0]
                                action_probabilities_sumTnograd0 = torch.FloatTensor([action_probabilities_sumN0])
                                
                            action_probabilities_sumT2 = action_probabilities_sumTnograd0+action_probabilitiesT
                            
                            with torch.no_grad():
                                action_probabilities_sumN2 = action_probabilitiesT.data.numpy()[0]
                                action_probabilities_sumTnograd2 = torch.FloatTensor([action_probabilities_sumN2])
                            action_probabilities_sumT0 = action_probabilities_sumT0+action_probabilities_sumTnograd2   
                            action_probabilities_sumT1 = action_probabilities_sumT1+action_probabilities_sumTnograd2 
                         
                        #Конвертируем данные в тензор
                        exp_next_obs = [x for x in exp_next_obs]
                        obs_agentT_next = torch.FloatTensor([exp_next_obs]).to(device)
                        
                        #Подаем минивыборку в целевую нейронную сеть
                        if agent_id == 0:
                            action_probabilitiesT_next = tgt_network0(obs_agentT_next)
                        elif agent_id == 1:
                            action_probabilitiesT_next = tgt_network1(obs_agentT_next)
                        elif agent_id == 2:
                            action_probabilitiesT_next = tgt_network2(obs_agentT_next)
                        
                        #Конвертируем данные в numpy    
                        action_probabilitiesT_next = action_probabilitiesT_next.to("cpu")
                        action_probabilities_next = action_probabilitiesT_next.data.numpy()[0]
                        
                        #Находим сумму Q-значений для выходов целевых нейронных сетей
                        for i in range (n_actions):
                            action_probabilities_next_sum[i] += action_probabilities_next[i]
                    ####_Конец цикла по агентам для суммирования Q-значений_###
                        
                    #Вычисляем целевое значение y 
                    y_batch = exp_rew_sum + gamma * np.max(action_probabilities_next_sum, axis=-1)*(1 - exp_termd)  
                    
                    #Переформатируем y_batch размером batch_size
                    y_batch64 = np.zeros([batch_size, qofa_out])
                    for i in range (batch_size):
                        for j in range (qofa_out):
                            y_batch64[i][j] = y_batch 
                     
                    y_batchT = torch.FloatTensor([y_batch64])
                    
                    
                    for agent_id in range(n_agents):
                        
                        if agent_id == 0:
                            #Обнуляем градиенты
                            optimizer0.zero_grad()
                            #Вычисляем функцию потерь
                            loss_t0 = objective0(action_probabilities_sumT0, y_batchT.squeeze(0))
                            #Выполняем обратное распространение ошибки
                            loss_t0.backward()
                            #Выполняем оптимизацию нейронных сетей
                            optimizer0.step()
                        elif agent_id == 1:
                            optimizer1.zero_grad()
                            loss_t1 = objective1(action_probabilities_sumT1, y_batchT.squeeze(0))
                            loss_t1.backward()
                            optimizer1.step()
                        elif agent_id == 2:
                            optimizer2.zero_grad()
                            loss_t2 = objective2(action_probabilities_sumT2, y_batchT.squeeze(0))
                            loss_t2.backward()
                            optimizer2.step()   
                                                
                        #Сохраняем данные для графиков
                        Loss_History.append(loss_t0) 
                        loss_n=loss_t0.data.numpy()
                        total_loss.append(loss_n)
                        m_loss.append(np.mean(total_loss[-10000:]))
                #######_Конец цикла по модели минивыборки_#####################
            ######################_конец if обучения_##########################
            
            #Синхронизируем веса основной и целевой нейронных сетей
            #каждые 100 шагов
            if (global_step+1) % copy_steps == 0 and global_step > start_steps:
                tgt_network0.load_state_dict(q_network0.state_dict())
                tgt_network1.load_state_dict(q_network1.state_dict())
                tgt_network2.load_state_dict(q_network2.state_dict())
            
            #####################_Конец цикла по агентам для обучения_#########
            
            #Обновляем счетчик общего количества шагов
            global_step += 1
           
        ######################_конец цикла while_##############################
        #Выводим счетчик шагов игры и общую награду за эпизод
        print('global_step=', global_step, "Total reward in episode {} = {}".format(e, episode_reward))
        
        #Собираем данные для графиков
        Reward_History.append(episode_reward)
        status = env.get_stats()
        winrate_history.append(status["win_rate"])
    
    ################_конец цикла по эпизодам игры_#############################
    
    #Закрываем среду StarCraft II
    env.close()
    
    #Сохраняем параметры обученных нейронных сетей
    torch.save(q_network0.state_dict(),"qnet_0.dat") 
    torch.save(q_network1.state_dict(),"qnet_1.dat")
    torch.save(q_network2.state_dict(),"qnet_2.dat")
    
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
    print("--- %s минут ---" % ((time.time() - start_time)/60))
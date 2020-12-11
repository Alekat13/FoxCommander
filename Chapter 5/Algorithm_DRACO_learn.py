#Мультиагентное обучение с подкреплением
#Глава 5. Роевое обучение
#Алгоритм DRACO

#Подключаем библиотеки
from smac.env import StarCraft2Env
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.optim as optim

import cv2

#Флаг вывода массива целиком
np.set_printoptions(threshold=np.inf)

#Размер карты ферромонов, совпадающий с размером игровой карты
MAPX = 32
MAPY = 32
#Объем буфера воспроизведения
BUF_LEN = 10000

#Определяем архитектуру нейронной сети
class AQ_network(nn.Module):
    #На вход нейронная сеть получает состояние среды в виде координат (x,y)
    #На выходе нейронная сеть возвращает оценку действий в виде AQ-значений
    def __init__(self, obs_size, n_actions):
        super(AQ_network, self).__init__()
        self.AQ_network = nn.Sequential(
            #Первый линейный слой обрабатывает входные данные состояния среды
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            #Второй линейный слой обрабатывает внутренние данные 
            nn.Linear(64, n_actions)           
        )
        #Применение к выходным данным функции Softmax
        self.sm_layer = nn.Softmax(dim=1)
    #Вначале данные x обрабатываются полносвязной сетью с функцией ReLU
    #На выходе происходит обработка функцией Softmax
    def forward(self, x):
        aq_network_out = self.AQ_network(x)
        sm_layer_out = self.sm_layer(aq_network_out)
        #Финальный выход нейронной сети
        return sm_layer_out

#Класс для реализации буфера воспроизведения
class Exp_Buf():
    #Буфер воспроизведения на основе обобщенной очереди 
    expbufVar = deque(maxlen=BUF_LEN)

#Вычисляем количество феромона в точке как среднее по области
def compute_pherintensity(pheromone_map, X, Y, pa):
    if (X >=1) and (Y>=1):
        pheromone_inpoint = (pheromone_map[X-pa][Y+pa] + pheromone_map[X][Y+pa] + pheromone_map[X+pa][Y+pa]+ \
                         pheromone_map[X-pa][Y]    + pheromone_map[X][Y]    + pheromone_map[X+pa][Y]+ \
                         pheromone_map[X-pa][Y-pa] + pheromone_map[X][Y-pa] + pheromone_map[X+pa][Y-pa])/9
    else: pheromone_inpoint = 0
    
    #Возвращаем значение феромона в точке
    return pheromone_inpoint

#Выбираем возможное действие с максимальным AQ-значением в зависимости от эпсилон
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
    env = StarCraft2Env(map_name="75z1сFOX", reward_only_positive=False, reward_scale_rate=200, difficulty="1")
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
    
    #Определяем основные параметры роевого обучения
    ###########################################################################   
    #Определим динамический эпсилон с затуханием
    eps_max = 1.0 #Начальное значение эпсилон
    eps_min = 0.1 #Финальное значение эпсилон
    eps_decay_steps = 20000 #Шаг затухания эпсилон
    #Основные переходы в алгоритме DRACO зависят от управляющих параметров
    global_step = 0  #подсчитываем общее количество шагов в игре
    copy_steps = 100 #каждые 100 шагов синхронизируем нейронные сети 
    start_steps = 1000 #начинаем обучать через 1000 шагов
    steps_train = 4  #после начала обучения продолжаем обучть каждый 4 шаг 
    #Размер минивыборки    
    batch_size = 32     
    #Общее количество эпизодов игры
    n_episodes = 151
    #Параметр дисконтирования
    gamma = 0.99 
    #Скорость обучения
    alpha = 0.01 
    #Коэффициент выветривания феромона
    evap = 0.98
    #Объем феромона, выделяемого каждым муравьем
    pher_volume = 0.1
    #Усиление феромона за успешное действие
    pher_intens = 50
    #Размер смежной области для вычисления феромона в точке
    pher_area = 1
    #Искусственный размер наблюдений агента в виде 4 координат
    obs_sizeXY = 4
    ###########################################################################  
  
    #Pytorch определяет возможность использования графического процессора
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Определяем выходной размер нейронной сети 
    qofa_out = n_actions
    #Создаем основную нейронную сеть
    aq_network = AQ_network(obs_sizeXY, n_actions).to(device)
    #Создаем целевую нейронную сеть
    tgt_network = AQ_network(obs_sizeXY, n_actions).to(device)
    #Создаем списки для мультиагентного случая
    aq_network_list = []
    tgt_network_list = []
    optimizer_list = []
    objective_list = []
    exp_buf_L = []
    
    for agent_id in range(n_agents):
        #Создаем список буферов
        exp_buf_L.append(Exp_Buf()) 
        #Создаем список основных нейронных сетей для каждого агента роя
        aq_network_list.append(aq_network)
        #Создаем список целевых нейронных сетей для каждого агента роя
        tgt_network_list.append(tgt_network)
        #Создаем список оптимизаторов нейронных сетей для каждого агента роя
        optimizer_list.append(optim.Adam(params=aq_network_list[agent_id].parameters(), lr=alpha))
        #Создаем список функций потерь для каждого агента роя
        objective_list.append(nn.MSELoss())
    #Выводим на печать пример одной основной нейронной сети
    print ('aq_network_list[0]=', aq_network_list[0])
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
        #Определяем карту феромонов
        pheromone_map = np.zeros([MAPX, MAPY])
        #Определим псевдокарту 32x32x1 как визуализацию феромонов 
        map_data = np.zeros((MAPY, MAPX, 3), np.uint8)
        
        #Шаги игры внутри эпизода
        ######################_цикл while_#####################################
        while not terminated:
                      
            #Обнуляем промежуточные переменные
            actions = []
            action = 0
            #Храним историю действий один шаг для разных агентов
            actionsFox = np.zeros([n_agents]) 
            #Храним историю состояний среды один шаг для разных агентов
            obs_agentXY = np.zeros([n_agents, obs_sizeXY]) 
            obs_agent_nextXY = np.zeros([n_agents, obs_sizeXY])
                        
            #Заполним карту феромонов и визуализацию феромонов
            #по правилу присутствия агента в определенной точке
            for agent_id in range(n_agents):
                #Получаем текущие характеристики юнита
                unit = env.get_unit_by_id(agent_id)
                #Заполняем карту феромонов
                pheromone_map[int(unit.pos.x)][int(unit.pos.y)] = pheromone_map[int(unit.pos.x)][int(unit.pos.y)] + pher_volume
                #Визуализируем феромоны на псведокарте BGR
                cv2.line(map_data, (int(unit.pos.x), int(unit.pos.y)), (int(unit.pos.x), int(unit.pos.y)), (0, 0, pheromone_map[int(unit.pos.x)][int(unit.pos.y)]), 1)
            
            ##############_Цикл по агентам для выполнения действий в игре_#####
            for agent_id in range(n_agents):
               
                #Получаем текущие характеристики юнита
                unit = env.get_unit_by_id(agent_id)
                obs_agentXY[agent_id][0] = unit.pos.x
                obs_agentXY[agent_id][1] = unit.pos.y
                
                #Соберем данные противников для псведокарты
                for e_id, e_unit in env.enemies.items():
                    obs_agentXY[agent_id][2] = e_unit.pos.x
                    obs_agentXY[agent_id][3] = e_unit.pos.y
                
                #Конвертируем данные в тензор
                obs_agentT = torch.FloatTensor([obs_agentXY[agent_id]]).to(device)
                #Передаем состояние среды в основную нейронную сеть 
                #и получаем AQ-значения для каждого действия
                action_probabilitiesT = aq_network_list[agent_id](obs_agentT)
                #Конвертируем данные в numpy
                action_probabilitiesT = action_probabilitiesT.to("cpu")
                action_probabilities = action_probabilitiesT.data.numpy()[0]
                #Находим возможные действия агента в данный момент времени 
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                #Выбираем возможное действие агента с учетом
                #максимального AQ-значения и параметра эпсилон
                action = select_actionFox(action_probabilities, avail_actions_ind, epsilon)
                #Обрабатываем исключение при ошибке библиотеки в возможных действиях
                if action is None: action = np.random.choice (avail_actions_ind)
                
                #Создаем дополнительный феромон за полезное действие в игре
                if (action == 6):
                    #Получаем текущие характеристики юнита
                    unit = env.get_unit_by_id(agent_id)
                    #Заполняем карту феромонов
                    pheromone_map[int(unit.pos.x)][int(unit.pos.y)] = pheromone_map[int(unit.pos.x)][int(unit.pos.y)] + pher_intens + pher_volume
                    
                #Собираем действия от разных агентов                 
                actions.append(action)
                actionsFox[agent_id] = action
            ######_конец цикла по агентам для выполнения действий в игре_######

            #Передаем действия агентов в среду, получаем награду 
            #и прерывание игры от среды
            reward, terminated, _ = env.step(actions)
            #Суммируем награды за этот шаг для вычисления награды за эпизод
            episode_reward += reward
            
            #Поворачиваем изображение псевдокарты
            flipped = cv2.flip(map_data, 0)
            #Увеличиваем изображение псевдокарты для отображения
            resized = cv2.resize(flipped, dsize=None, fx=10, fy=10)
            #Выводим на экран псведокарту
            cv2.imshow('Pheromone map', resized)
            cv2.waitKey(1)
                      
            #####################_Цикл по агентам для обучения_################
            for agent_id in range(n_agents):
                #Получаем характеристики переместившегося юнита
                unit = env.get_unit_by_id(agent_id)
                obs_agent_nextXY[agent_id][0] = unit.pos.x
                obs_agent_nextXY[agent_id][1] = unit.pos.y
                
                #Соберем данные противников для псведокарты
                for e_id, e_unit in env.enemies.items():
                    obs_agent_nextXY[agent_id][2] = e_unit.pos.x
                    obs_agent_nextXY[agent_id][3] = e_unit.pos.y
                
                #Вычисляем количество феромона в точке
                pheromone_inpoint = compute_pherintensity (pheromone_map, int(unit.pos.x), int(unit.pos.y), pher_area)
                #Создаем подкрепляющий сигнал на основе феромона и награды
                pher_reinf = reward + pheromone_inpoint
                #Сохраняем переход в буфере воспроизведения для каждого агента 
                exp_buf_L[agent_id].expbufVar.append([obs_agentXY[agent_id], actionsFox[agent_id], obs_agent_nextXY[agent_id], pher_reinf, terminated])
               
                #Если буфер воспроизведения наполнен, начинаем обучать сеть
                ########################_начало if обучения_###################
                if (global_step % steps_train == 0) and (global_step > start_steps):
                    #Получаем минивыборку из буфера воспроизведения
                    exp_obs, exp_act, exp_next_obs, exp_pher_reinf, exp_termd = sample_from_expbuf(exp_buf_L[agent_id].expbufVar, batch_size)
                    
                    #Конвертируем данные в тензор
                    exp_obs = [x for x in exp_obs]
                    obs_agentT = torch.FloatTensor([exp_obs]).to(device)
                    
                    #Подаем минивыборку в основную нейронную сеть
                    #чтобы получить AQ(s,a)
                    action_probabilitiesT = aq_network_list[agent_id](obs_agentT)
                    action_probabilitiesT = action_probabilitiesT.to("cpu")
                    action_probabilities = action_probabilitiesT.data.numpy()[0]
                    
                    #Конвертируем данные в тензор
                    exp_next_obs = [x for x in exp_next_obs]
                    obs_agentT_next = torch.FloatTensor([exp_next_obs]).to(device)
                    
                    #Подаем минивыборку в основную нейронную сеть
                    #чтобы получить AQ'(s',a')
                    action_probabilitiesT_next = tgt_network_list[agent_id](obs_agentT_next)
                    action_probabilitiesT_next = action_probabilitiesT_next.to("cpu")
                    action_probabilities_next = action_probabilitiesT_next.data.numpy()[0]
                    
                    #Вычисляем целевое значение y 
                    y_batch = exp_pher_reinf + gamma * np.max(action_probabilities_next, axis=-1)*(1 - exp_termd) 
                    
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
                    tgt_network_list[agent_id].load_state_dict(aq_network_list[agent_id].state_dict())
            
            #####################_Конец цикла по агентам для обучения_#########
                  
            #Обновляем счетчик общего количества шагов
            global_step += 1
            
            #Применяем коэффициент испарения феромона
            for i in range(MAPX):
                for j in range(MAPY):
                    pheromone_map[i][j] = evap*pheromone_map[i][j] 
           
        ######################_конец цикла while_##############################
        #Выводим счетчик шагов игры и общую награду за эпизод
        print('global_step=', global_step, "Total reward in episode {} = {}".format(e, episode_reward))
        
        #Собираем данные для графиков
        Reward_History.append(episode_reward)
        status = env.get_stats()
        winrate_history.append(status["win_rate"])
        
    ################_конец цикла по эпизодам игры_############################################
    
    #Закрываем среду StarCraft II
    env.close()
    #Убираем с экрана псевдокарту
    cv2.destroyAllWindows()
    
    #Сохраняем параметры обученных нейронных сетей
    for agent_id in range(n_agents):
        torch.save(aq_network_list[agent_id].state_dict(),"aqnet_%.0f.dat"%agent_id) 
        
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
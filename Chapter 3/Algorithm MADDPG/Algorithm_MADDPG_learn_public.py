#Мультиагентное обучение с подкреплением
#Глава 3. Нейросетевое обучение
#Алгоритм MADDPG

#Подключаем библиотеки
from smac.env import StarCraft2Env
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim

#Флаг вывода массива целиком
np.set_printoptions(threshold=np.inf)

#Определяем архитектуру нейронной сети исполнителя
class MADDPG_Actor(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(MADDPG_Actor, self).__init__()
        #На вход нейронная сеть получает состояние среды для отдельного агента 
        #На выходе нейронная сеть возвращает стратегию действий
        self.MADDPG_Actor = nn.Sequential(
            #Первый линейный слой обрабатывает входные данные состояния среды
            nn.Linear(obs_size, 60),
            nn.ReLU(),
            #Второй линейный слой обрабатывает внутренние данные 
            nn.Linear(60, 60),
            nn.ReLU(),
            #Третий линейный слой обрабатывает внутренние данные 
            nn.Linear(60, 60),
            nn.ReLU(),
            #Четвертый линейный слой обрабатывает данные для стратегии действий
            nn.Linear(60, n_actions)
            )
        #Финальный выход нерйонной сети обрабатывается функцией Tanh()
        self.tanh_layer = nn.Tanh()
    #Вначале данные x обрабатываются полносвязной сетью с функцией ReLU
    #На выходе происходит обработка функцией Tanh()
    def forward(self, x):
        #Обработка полносвязными линейными слоями
        network_out = self.MADDPG_Actor(x)
        #Обработка функцией Tanh()
        tanh_layer_out = self.tanh_layer(network_out)
        #Выход нейронной сети
        return tanh_layer_out

#Определяем архитектуру нейронной сети критика
class MADDPG_Critic(nn.Module):
    def __init__(self, full_obs_size, n_actions_agents):
        super(MADDPG_Critic, self).__init__()
        #На вход нейронная сеть получает состояние среды,
        #включающее все локальные состояния среды от отдельных агентов
        #и все выполненные действия отдельных агентов
        #На выходе нейронная сеть возвращает корректирующее значение
        self.network = nn.Sequential(
            #Первый линейный слой обрабатывает входные данные    
            nn.Linear(full_obs_size+n_actions_agents, 202),
            nn.ReLU(),
            #Второй линейный слой обрабатывает внутренние данные
            nn.Linear(202, 60),
            nn.ReLU(),
            #Третий линейный слой обрабатывает внутренние данные
            nn.Linear(60, 30),
            nn.ReLU(),
            #Четвертый линейный слой обрабатывает выходные данные
            nn.Linear(30, 1)
            )
    #Данные x последовательно обрабатываются полносвязной сетью с функцией ReLU
    def forward(self, state, action):
        #Объединяем данные состояний и действий для передачи в сеть
        x = torch.cat([state, action], dim=2)
        #Результаты обработки 
        Q_value = self.network(x)
        #Финальный выход нейронной сети
        return Q_value
        
#Выбираем возможное действие с максимальным из стратегии действий
#с учетом дополнительного случайного шума
def select_actionFox(act_prob, avail_actions_ind, n_actions, noise_rate):
    p = np.random.random(1).squeeze()
    #Добавляем случайный шум к действиям для исследования
    #разных вариантов действий
    for i in range(n_actions):
        #Создаем шум заданного уровня
        noise = noise_rate*(np.random.rand())
        #Добавляем значение шума к значению вероятности выполнения действия
        act_prob [i] =  act_prob [i] + noise
    
    #Выбираем действия в зависимости от вероятностей их выполнения
    for j in range(n_actions):
        #Выбираем случайный элемент из списка 
        actiontemp =  random.choices(['0','1','2','3','4','5','6'], weights=[act_prob[0],act_prob[1],act_prob[2],act_prob[3],act_prob[4],act_prob[5],act_prob[6]])
        #Преобразуем тип данных
        action = int (actiontemp[0])
        #Проверяем наличие выбранного действия в списке действий
        if action in avail_actions_ind:
            return action
        else:
            act_prob[action] = 0
          
#Создаем минивыборку определенного объема из буфера воспроизведения        
def sample_from_expbuf(experience_buffer, batch_size):
    #Функция возвращает случайную последовательность заданной длины
    perm_batch = np.random.permutation(len(experience_buffer))[:batch_size]
    #Минивыборка
    experience = np.array(experience_buffer)[perm_batch]
    #Возвращаем значения минивыборки по частям
    return experience[:,0], experience[:,1], experience[:,2], experience[:,3], experience[:,4], experience[:,5]        

#Основная функция программы
def main():
    #Загружаем среду Starcraft II, карту, сложность противника и расширенную  
    #награду 
    env = StarCraft2Env(map_name="3ps1zgWallFOX", reward_only_positive=False, reward_scale_rate=200, difficulty="1")
    #Получаем и выводим на печать информацию о среде
    env_info = env.get_env_info()
    print ('env_info=',env_info)
    #Получаем и выводим на печать размер локальных состояний среды для агента
    obs_size =  env_info.get('obs_shape')
    print ("obs_size=",obs_size)
    #Количество действий агента 
    n_actions = env_info["n_actions"]
    #Количество дружественных агентов
    n_agents = env_info["n_agents"]
    
    #Определяем основные параметры нейросетевого обучения    
    ##########################################################################
    #Некоторые переходы в алгоритме MADDPG зависят от шагов игры
    global_step = 0 #подсчитываем общее количество шагов в игре
    start_steps = 1000 #начинаем обучать через 1000 шагов
    steps_train = 4 #после начала обучения продолжаем обучать каждый 4 шаг 
    #Размер минивыборки 
    batch_size = 32 
    #Общее количество эпизодов игры
    n_episodes = 510 
    #Параметр дисконтирования.
    gamma = 0.99 
    #Скорость обучения исполнителя
    alpha_actor = 0.01
    #Скорость обучения критика
    alpha_critic = 0.01 
    #Уровень случайного шума
    noise_rate = 0.01 
    #Начальное значение случайного шума
    noise_rate_max = 0.9
    #Финальное значение случайного шума
    noise_rate_min = 0.01 
    #Шаг затухания уровня случайного шума
    noise_decay_steps = 15000
    #Параметр мягкой замены
    tau = 0.01 
    #Объем буфера воспроизведения
    buffer_len = 10000 
    ###########################################################################   
        
    #Создаем буфер воспроизведения на основе deque
    experience_buffer = deque(maxlen=buffer_len)
        
    #Pytorch определяет возможность использования графического процессора
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #Реализуем модифицированный алгоритм MADDPG 
    #с одной нейронной сетью критика и тремя нейронными сетями исполнителей  
    #Создаем основную нейронную сеть исполнителя
    actor_network = MADDPG_Actor(obs_size, n_actions).to(device)
    #Создаем целевую нейронную сеть исполнителя
    tgtActor_network = MADDPG_Actor(obs_size, n_actions).to(device)
    #Синхронизуем веса нейронных сетей исполнителей
    tgtActor_network.load_state_dict(actor_network.state_dict())
    
    #Создаем основную нейронную сеть критика
    critic_network = MADDPG_Critic(obs_size*n_agents, n_agents).to(device)
    #Создаем целевую нейронную сеть критика
    tgtCritic_network = MADDPG_Critic(obs_size*n_agents, n_agents).to(device)
    #Синхронизуем веса нейронных сетей критиков
    tgtCritic_network.load_state_dict(critic_network.state_dict())
    
    #Создаем списки для мультиагентного случая
    actor_network_list = []
    tgtActor_network_list = []
    optimizerActor_list = []
    objectiveActor_list = []
        
    for agent_id in range(n_agents):
        #Создаем список основных нейронных сетей исполнителей для трех агентов
        actor_network_list.append(actor_network)
        #Создаем список целевых нейронных сетей исполнителей
        tgtActor_network_list.append(tgtActor_network)
        #Создаем список оптимизаторов нейронных сетей исполнителей
        optimizerActor_list.append(optim.Adam(params=actor_network_list[agent_id].parameters(), lr=alpha_actor))
        #Создаем список функций потерь исполнителей
        objectiveActor_list.append(nn.MSELoss())
        
    #Создаем оптимизатор нейронной сети критика
    optimizerCritic = optim.Adam(params=critic_network.parameters(), lr=alpha_critic)
    #Создаем функцию потерь критика
    objectiveCritic = nn.MSELoss()
    
    #Выводим на печать архитектуру нейронных сетей
    print ('Actor_network_list=', actor_network_list)
    print ('Critic_network_list=', critic_network)
            
    #Определяем вспомогательные параметры
    Loss_History = [] 
    Loss_History_actor = []
    Reward_History = []
    winrate_history = []
    total_loss = []
    total_loss_actor = []
    m_loss = []
    m_loss_actor = []
    

    #Основной цикл по эпизодам игры
    ################_цикл for по эпизодам_#####################################
    for e in range(n_episodes):
       
        #Перезагружаем среду
        env.reset()
        #Флаг окончания эпизода
        terminated = False
        #Награда за эпизод
        episode_reward = 0
        #Обновляем и выводим динамический уровень случайного шума
        noise_rate = max(noise_rate_min, noise_rate_max - (noise_rate_max-noise_rate_min) * global_step/noise_decay_steps)
        print ('noise_rate=', noise_rate)
                
        #Шаги игры внутри эпизода
        ######################_цикл while_#####################################
        while not terminated:
            #Обнуляем промежуточные переменные
            actions = []
            observations = []
            action = 0
            #Храним историю действий один шаг для разных агентов
            actionsFox = np.zeros([n_agents]) 
            #Храним историю состояний среды один шаг для разных агентов
            obs_agent = np.zeros([n_agents], dtype=object) 
            obs_agent_next = np.zeros([n_agents], dtype=object)
                        
            ###########_Цикл по агентам для выполнения действий в игре_########
            for agent_id in range(n_agents):
                #Получаем состояние среды для независимого агента 
                obs_agent[agent_id] = env.get_obs_agent(agent_id)
                #Конвертируем данные в тензор
                obs_agentT = torch.FloatTensor([obs_agent[agent_id]]).to(device)
                #Передаем состояние среды в основную нейронную сеть 
                #и получаем стратегию действий
                action_probabilitiesT = actor_network_list[agent_id](obs_agentT)
                #Конвертируем данные в numpy
                action_probabilitiesT = action_probabilitiesT.to("cpu")
                action_probabilities = action_probabilitiesT.data.numpy()[0]
                
                #Находим возможные действия агента в данный момент времени 
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                #Выбираем возможное действие агента с учетом
                #стратегии действий и уровня случайного шума
                action = select_actionFox(action_probabilities, avail_actions_ind, n_actions, noise_rate)
                #Обрабатываем исключение при ошибке в возможных действиях
                if action is None:
                    action = np.random.choice (avail_actions_ind)
                    
                #Собираем действия от разных агентов               
                actions.append(action)
                actionsFox[agent_id] = action
                #Собираем локальные состояния среды от разных агентов
                for i in range(obs_size):
                    observations.append(obs_agent[agent_id][i])
            ######_конец цикла по агентам для выполнения действий в игре_######

            #Передаем действия агентов в среду, получаем награду
            #и прерывание игры от среды
            reward, terminated, _ = env.step(actions)
            #Суммируем награды за этот шаг для вычисления награды за эпизод
            episode_reward += reward
            
            #Подготовляем данные для сохранения в буфере воспроизведения
            actions_next = []
            observations_next = []
            #Если эпизод не завершился, то можно найти новые действия и состояния
            if terminated == False:
                for agent_id in range(n_agents):
                    #Получаем новое состояние среды для независимого агента 
                    obs_agent_next[agent_id] = env.get_obs_agent(agent_id)
                    #Собираем от разных агентов новые состояния
                    for i in range(obs_size):
                        observations_next.append(obs_agent_next[agent_id][i])
                    #Конвертируем данные в тензор
                    obs_agent_nextT = torch.FloatTensor([obs_agent_next[agent_id]]).to(device)
                    #Получаем новые действия агентов для новых состояний
                    #из целевой сети исполнителя
                    action_probabilitiesT = tgtActor_network_list[agent_id](obs_agent_nextT)
                    #Конвертируем данные в numpy
                    action_probabilitiesT = action_probabilitiesT.to("cpu")
                    action_probabilities = action_probabilitiesT.data.numpy()[0]
                    #Находим новые возможные действия агента
                    avail_actions = env.get_avail_agent_actions(agent_id)
                    avail_actions_ind = np.nonzero(avail_actions)[0]
                    #Выбираем новые возможные действия
                    action = select_actionFox(action_probabilities, avail_actions_ind, n_actions, noise_rate)
                    if action is None:
                        action = np.random.choice (avail_actions_ind)
                    #Собираем новые действия от разных агентов
                    actions_next.append(action)
            elif terminated == True:
                #если эпизод на этом шаге завершился, то новых действий не будет
                actions_next = actions
                observations_next = observations
                                
            #Сохраняем переход в буфере воспроизведения 
            experience_buffer.append([observations, actions, observations_next, actions_next, reward, terminated])
                        
            #Если буфер воспроизведения наполнен, начинаем обучать сеть
            ########################_начало if обучения_#######################
            if (global_step % steps_train == 0) and (global_step > start_steps):
                #Получаем минивыборку из буфера воспроизведения
                exp_obs, exp_acts, exp_next_obs, exp_next_acts, exp_rew, exp_termd = sample_from_expbuf(experience_buffer, batch_size)
                    
                #Конвертируем данные в тензор
                exp_obs = [x for x in exp_obs]
                obs_agentsT = torch.FloatTensor([exp_obs]).to(device)
                exp_acts = [x for x in exp_acts]
                act_agentsT = torch.FloatTensor([exp_acts]).to(device)
                                    
                ###############_Обучаем нейронную сеть критика_################
                
                #Получаем значения из основной сети критика
                action_probabilitieQT = critic_network(obs_agentsT, act_agentsT)
                action_probabilitieQT = action_probabilitieQT.to("cpu")
                               
                #Конвертируем данные в тензор
                exp_next_obs = [x for x in exp_next_obs]
                obs_agents_nextT = torch.FloatTensor([exp_next_obs]).to(device)
                exp_next_acts = [x for x in exp_next_acts]
                act_agents_nextT = torch.FloatTensor([exp_next_acts]).to(device)
                                        
                #Получаем значения из целевой сети критика
                action_probabilitieQ_nextT = tgtCritic_network(obs_agents_nextT, act_agents_nextT)
                action_probabilitieQ_nextT = action_probabilitieQ_nextT.to("cpu")
                action_probabilitieQ_next = action_probabilitieQ_nextT.data.numpy()[0]
                    
                #Переформатируем y_batch размером batch_size
                y_batch = np.zeros([batch_size])
                action_probabilitieQBT = torch.empty(1, batch_size, dtype=torch.float)
                
                for i in range (batch_size):
                    #Вычисляем целевое значение y 
                    y_batch[i] = exp_rew[i] + (gamma*action_probabilitieQ_next[i])*(1 - exp_termd[i])
                    action_probabilitieQBT[0][i] = action_probabilitieQT[0][i]
                
                y_batchT = torch.FloatTensor([y_batch])
                
                #Обнуляем градиенты
                optimizerCritic.zero_grad()
                 
                #Вычисляем функцию потерь критика
                loss_t_critic = objectiveCritic(action_probabilitieQBT, y_batchT) 
                    
                #Сохраняем данные для графиков
                Loss_History.append(loss_t_critic) 
                loss_n_critic = loss_t_critic.data.numpy()
                total_loss.append(loss_n_critic)
                m_loss.append(np.mean(total_loss[-1000:]))
                    
                #Выполняем обратное распространение ошибки для критика
                loss_t_critic.backward()
                
                #Выполняем оптимизацию нейронной сети критика
                optimizerCritic.step()
                ###################_Закончили обучать критика_#################
                
                ##############_Обучаем нейронные сети исполнителей_############
                #Разбираем совместное состояние на локальные состояния
                obs_local1 = np.zeros([batch_size, obs_size])
                obs_local2 = np.zeros([batch_size, obs_size])
                obs_local3 = np.zeros([batch_size, obs_size])
                for i in range (batch_size):
                    for j in range (obs_size):
                         obs_local1[i][j] = exp_obs[i][j]
                for i in range (batch_size):
                    k=0
                    for j in range (obs_size, obs_size*2):
                         obs_local2[i][k] = exp_obs[i][j]
                         k = k + 1
                for i in range (batch_size):
                    k=0
                    for j in range (obs_size*2, obs_size*3):
                         obs_local3[i][k] = exp_obs[i][j]
                         k = k + 1
                #Конвертируем данные в тензор                
                obs_agentT1 = torch.FloatTensor([obs_local1]).to(device)
                obs_agentT2 = torch.FloatTensor([obs_local2]).to(device)
                obs_agentT3 = torch.FloatTensor([obs_local3]).to(device)
                
                #Обнуляем градиенты 
                optimizerActor_list[0].zero_grad()
                optimizerActor_list[1].zero_grad()
                optimizerActor_list[2].zero_grad()
                
                #Подаем в нейронные сети исполнителей локальные состояния
                action_probabilitiesT1 = actor_network_list[0](obs_agentT1)
                action_probabilitiesT2 = actor_network_list[1](obs_agentT2)
                action_probabilitiesT3 = actor_network_list[2](obs_agentT3)
                                
                #Конвертируем данные в numpy
                action_probabilitiesT1 = action_probabilitiesT1.to("cpu")
                action_probabilitiesT2 = action_probabilitiesT2.to("cpu")
                action_probabilitiesT3 = action_probabilitiesT3.to("cpu")
                action_probabilities1 = action_probabilitiesT1.data.numpy()[0]
                action_probabilities2 = action_probabilitiesT2.data.numpy()[0]
                action_probabilities3 = action_probabilitiesT3.data.numpy()[0]
                
                #Вычисляем максимальные значения с учетом объема минивыборки
                act_full = np.zeros([batch_size, n_agents])
                for i in range (batch_size):
                    act_full[i][0] = np.argmax(action_probabilities1[i])
                    act_full[i][1] = np.argmax(action_probabilities2[i])
                    act_full[i][2] = np.argmax(action_probabilities3[i])
                act_fullT = torch.FloatTensor([act_full]).to(device)
                
                #Конвертируем данные в тензор
                exp_obs = [x for x in exp_obs]
                obs_agentsT = torch.FloatTensor([exp_obs]).to(device)
                                
                #Задаем значение функции потерь для нерйонных сетей исполнителей
                #как отрицательный выход критика
                actor_lossT = -critic_network(obs_agentsT, act_fullT)
                
                #Усредняем значение по количеству элементов минивыборки
                actor_lossT = actor_lossT.mean()    
                
                #Выполняем обратное распространение ошибки
                actor_lossT.backward()
                
                #Выполняем оптимизацию нейронных сетей исполнителей
                optimizerActor_list[0].step()
                optimizerActor_list[1].step()
                optimizerActor_list[2].step()
                
                #Собираем данные для графиков
                actor_lossT = actor_lossT.to("cpu")
                Loss_History_actor.append(actor_lossT) 
                actor_lossN = actor_lossT.data.numpy()
                total_loss_actor.append(actor_lossN)
                m_loss_actor.append(np.mean(total_loss_actor[-1000:]))
                ##############_Закончили обучать исполнителей_#################
                
                #Рализуем механизм мягкой замены
                #Обновляем целевую сеть критика
                for target_param, param in zip(tgtCritic_network.parameters(), critic_network.parameters()):
                    target_param.data.copy_((1 - tau) * param.data + tau * target_param.data)
                #Обновляем целевые сети акторов
                for agent_id in range(n_agents):
                    for target_param, param in zip(tgtActor_network_list[agent_id].parameters(), actor_network_list[agent_id].parameters()):
                        target_param.data.copy_((1 - tau) * param.data + tau * target_param.data)
 
                ######################_конец if обучения_######################
                
            #Обновляем счетчик общего количества шагов
            global_step += 1
        
        ######################_конец цикла while_##############################
        
        #Выводим на печать счетчик шагов игры и общую награду за эпизод
        print('global_step=', global_step, "Total reward in episode {} = {}".format(e, episode_reward))
        #Собираем данные для графиков
        Reward_History.append(episode_reward)
        status = env.get_stats()
        winrate_history.append(status["win_rate"])
        
    ################_конец цикла по эпизодам игры_#############################
    
    #Закрываем среду StarCraft II
    env.close()
    
    #Сохраняем параметры обученных нейронных сетей
    for agent_id in range(n_agents):
        torch.save(actor_network_list[agent_id].state_dict(),"actornet_%.0f.dat"%agent_id) 
    
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
    #Значения функции потерь исполнителя
    plt.figure(num=None, figsize=(6, 3), dpi=150, facecolor='w', edgecolor='k')
    plt.plot(m_loss_actor)
    plt.xlabel('Номер каждой 1000 итерации')
    plt.ylabel('Функция потерь исполнителя')
    plt.show()
    #Значения функции потерь критика
    plt.figure(num=None, figsize=(6, 3), dpi=150, facecolor='w', edgecolor='k')
    plt.plot(m_loss)
    plt.xlabel('Номер каждой 1000 итерации')
    plt.ylabel('Функция потерь критика')
    plt.show()
   
#Точка входа в программу  
if __name__ == "__main__":
    start_time = time.time()
    main() 
    print("--- %s минут ---" % ((time.time() - start_time)/60))

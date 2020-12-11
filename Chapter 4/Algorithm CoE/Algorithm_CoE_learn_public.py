#Мультиагентное обучение с подкреплением
#Глава 4. Эволюционное обучение
#Алгоритм CoE

#Подключаем библиотеки
from smac.env import StarCraft2Env
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import OrderedDict

import torch
import torch.nn as nn


#Флаг вывода массива целиком
np.set_printoptions(threshold=np.inf)

#Определяем архитектуру нейронной сети
class Q_network(nn.Module):
    #На вход нейронная сеть получает состояние среды
    #На выходе нейронная сеть возвращает оценку действий в виде Q-значений
    def __init__(self, obs_size, n_actions):
        super(Q_network, self).__init__()
        self.Qlayers = nn.Sequential(OrderedDict([
            ('fl1', nn.Linear(obs_size, 64)),
            ('relu1', nn.ReLU()),
            ('fl2',nn.Linear(64, n_actions))
        ]))
        #Применение к выходным данным функции Softmax
        self.sm_layer = nn.Softmax(dim=1)
    #Вначале данные x обрабатываются полносвязной сетью с функцией ReLU
    #На выходе происходит обработка функцией Softmax
    def forward(self, x):
        q_network_out = self.Qlayers(x)
        sm_layer_out = self.sm_layer(q_network_out)
        #Финальный выход нейронной сети
        return sm_layer_out
    
#Выбираем возможное действие с максимальным Q-значением 
def select_actionFox(action_probabilities, avail_actions_ind):
    p = np.random.random(1).squeeze()
    #Находим возможное действие:
    #Проверяем есть ли действие в доступных действиях агента
    for ia in action_probabilities:
        action = np.argmax(action_probabilities)
        if action in avail_actions_ind:
            return action
        else:
            action_probabilities[action] = 0


#Основная функция программы
def main():
    
    #Загружаем среду Starcraft II, карту, сложность противника и расширенную награду 
    env = StarCraft2Env(map_name="2p4zFOX", reward_only_positive=False, reward_scale_rate=200, difficulty="1")
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
    
    #Определяем основные параметры эволюционного обучения
    ###########################################################################   
    population_amount = 50 #Количество агентов в популяции
    noise_level = 0.01 #Уровень шума для мутации
    mutant_amount = 20 #Количество агентов, которые будут мутированы
    elite_amount = 10 #Количество элитных особей для следующего поколения
    crossover_amount = 10 #Количество агентов для оператора скрещивания
    global_step = 0  #Подсчитываем общее количество шагов в игре
    n_episodes = 50 #Количество эпизодов игры или поколений эволюции
    ###########################################################################   
    
    #Используем для обучения только центральный процессор
    device = "cpu" 
      
    #Создаем списки популяций для двух агентов
    q_network_pop0 = []
    q_network_pop1 = []
    q_network_pop_next0 = []
    q_network_pop_next1 = []

    for agent_id in range(n_agents):
        for pai in range(population_amount):
            #Создаем основную нейронную сеть
            #Заново для каждого агента популяции, чтобы иметь разные веса
            q_network = Q_network(obs_size, n_actions).to(device)
            if agent_id == 0:
                #Создаем список нейронных сетей популяции
                q_network_pop_next0.append(q_network)
            elif agent_id == 1:
                q_network_pop_next1.append(q_network)
  
    #Выводим на печать архитектуру базовой нейронной сети популяции 
    print ('q_network_pop_next0 =', q_network_pop_next0[0])
    
    #Определяем вспомогательные параметры
    reward_history = []
    winrate_history = []
    bestindex_historyS = []
    bestindex_historyA = np.zeros([population_amount])
    #Параметр выбора уровня нейронной сети для скрещивания
    layer_choice = [0, 1]
       
    #Основной цикл по эпизодам игры или поколениям эволюции
    ################_цикл for по эпизодам_#####################################
    for e in range(n_episodes):
        print ("-----------------------Episode--------------------", e)
        
        global_reward = 0
        
        #Задаем нулевые значения фитнесс функции вначале эпизода
        fitness_pai = np.zeros([population_amount])
                
        #Обновляем поколение
        q_network_pop0 = q_network_pop_next0 
        q_network_pop1 = q_network_pop_next1 
        
        ################_цикл for pai по популяции агента в эпизоде_###########
        for pai in range(population_amount):
            print ("Population №", pai)
            #Перезагружаем среду
            env.reset()
            #Флаг окончания эпизода
            terminated = False
            #Награда за эпизод
            episode_reward = 0
            
            #Шаги игры внутри эпизода
            ######################_цикл while_#################################
            while not terminated:
                                
                #Обнуляем промежуточные переменные
                actions = []
                action = 0
                
                #Храним историю состояний среды один шаг для разных агентов
                obs_agent = np.zeros([n_agents], dtype=object) 
                                
                #########_Цикл по агентам для выполнения действий в игре_######
                for agent_id in range(n_agents):
                    
                    #Получаем состояние среды для независимого агента 
                    obs_agent[agent_id] = env.get_obs_agent(agent_id)
                    
                    #Конвертируем данные в тензор
                    obs_agentT = torch.FloatTensor([obs_agent[agent_id]]).to(device)
                    
                    #Передаем состояние среды в основную нейронную сеть 
                    #и получаем Q-значения для каждого действия
                    if agent_id == 0:
                        action_probabilitiesT = q_network_pop0[pai](obs_agentT)
                        
                    elif agent_id == 1:
                        action_probabilitiesT = q_network_pop1[pai](obs_agentT)
                    
                    #Конвертируем данные в numpy
                    action_probabilitiesT = action_probabilitiesT.to("cpu")
                    action_probabilities = action_probabilitiesT.data.numpy()[0]
                    #Находим возможные действия агента в данный момент времени 
                    avail_actions = env.get_avail_agent_actions(agent_id)
                    avail_actions_ind = np.nonzero(avail_actions)[0]
                                   
                    #Выбираем возможное действие агента с учетом
                    #максимального Q-значения 
                    action = select_actionFox(action_probabilities, avail_actions_ind)
                    
                    #Обрабатываем исключение при ошибке в возможных действиях
                    if action is None:
                        action = np.random.choice (avail_actions_ind)
                        
                    #Собираем действия от разных агентов                 
                    actions.append(action)
                   
                ####_конец цикла по агентам для выполнения действий в игре_####
                
                #Передаем действия агентов в среду, получаем награду 
                #и прерывание игры от среды
                reward, terminated, info = env.step(actions)
                #Суммируем награды за этот шаг для вычисления награды за популяцию
                episode_reward += reward
                #Обновляем счетчик общего количества шагов
                global_step += 1
                
            ######################_конец цикла while###########################
            
            #Вычисляем значение фитнесс функции для нейронной сети популяции
            #Обрабатываем исключение KeyError
            if info.get('battle_won') is None:
                #В качестве фитнесса берем награду за эпизод
                fitness_pai[pai] = episode_reward
            else:
                if (info['battle_won'] == True):
                    #Если была победа, то присваиваем максимальный фитнесс
                    fitness_pai[pai] = 300 + episode_reward
                else:
                    #В качесттве фитнесса берем награду за эпизод
                    fitness_pai[pai] = episode_reward
            
            #Выводим счетчик шагов игры и общую награду за эпизод
            print('Среднее количество шагов=', (global_step/population_amount)/(e+1), "Награда для НС {} = {}".format(pai, episode_reward))
            print('Фитнесс = ', fitness_pai[pai])
            global_reward = global_reward + episode_reward
        
        ############_конец цикла for pai по популяции агента в эпизоде_########
        
        #Собираем данные для графиков
        reward_history.append(global_reward/population_amount)
        status = env.get_stats()
        winrate_history.append(status["win_rate"])
        
        ################_Начинаем обучать_#####################################
                     
        #Новое поколение создается на основе предыдущего поколения
        q_network_pop_next0 = q_network_pop0  
        q_network_pop_next1 = q_network_pop1 
        
        #Но для эволюции некоторые особи нового поколения будут изменены
        #Выбираем нейронную сеть с лучшим фитнесом
        #Вначале находим индекс с максимальным значением фитнесса
        fitness_sort = np.sort(fitness_pai)[::-1] 
        bestindex_mas = np.zeros([elite_amount])
        for eai in range(elite_amount):
            for i in range(population_amount):
                if (fitness_sort[eai] == fitness_pai[i]):
                    bestindex_mas [eai] = i
        
        #Элитное множество особей нового поколения оставялем без изменений
        best_index = 0
        for eai in range(elite_amount):
            best_index = int (bestindex_mas[eai])
            bestindex_historyA[best_index] = bestindex_historyA[best_index] + 1
            q_network_pop_next0[eai] = q_network_pop0[best_index]
            q_network_pop_next1[eai] = q_network_pop1[best_index]
            
        #Скрещиваем некоторое количество нейронных сетей
        #Вначале выбираем кандидатов для срещивания и генерим случайные индексы
        index_rand0 = np.zeros([crossover_amount])
        index_rand1 = np.zeros([crossover_amount])
        for i in range(crossover_amount):
            index_rand0[i] = np.random.randint(elite_amount, population_amount)
            index_rand1[i] = np.random.randint(elite_amount, population_amount)
            
        #Индекс для особей после elite_amount и до количества мутантов
        for i in range(crossover_amount):
            
            ind0 = int (index_rand0[i])
            ind1 = int (index_rand1[i])
            
            #Заполняем веса 0 уровня первой сети
            rand_1_0 = np.random.choice (layer_choice) 
            if (rand_1_0==0):
                q_network_pop_next0[ind0].Qlayers.fl1.weight = q_network_pop0[ind0].Qlayers.fl1.weight
            elif (rand_1_0==1):
                q_network_pop_next0[ind0].Qlayers.fl1.weight = q_network_pop1[ind1].Qlayers.fl1.weight
                
            #Заполняем веса 2 уровня первой сети
            rand_1_2 = np.random.choice (layer_choice)
            if (rand_1_2==0):
                q_network_pop_next0[ind0].Qlayers.fl2.weight = q_network_pop0[ind0].Qlayers.fl2.weight
            elif (rand_1_2==1):
                q_network_pop_next0[ind0].Qlayers.fl2.weight = q_network_pop1[ind1].Qlayers.fl2.weight
                
            #Заполняем веса 0 уровня второй сети
            rand_2_0 = np.random.choice (layer_choice)
            if (rand_2_0==0):
                q_network_pop_next1[ind1].Qlayers.fl1.weight = q_network_pop0[ind0].Qlayers.fl1.weight
            elif (rand_2_0==1):
                q_network_pop_next1[ind1].Qlayers.fl1.weight = q_network_pop1[ind1].Qlayers.fl1.weight
            
            #Заполняем веса 2 уровня второй сети
            rand_2_2 = np.random.choice (layer_choice)
            if (rand_2_2==0):
                q_network_pop_next1[ind1].Qlayers.fl2.weight = q_network_pop0[ind0].Qlayers.fl2.weight
            elif (rand_2_2==1):
                q_network_pop_next1[ind1].Qlayers.fl2.weight = q_network_pop1[ind1].Qlayers.fl2.weight
           
        #Мутируем некоторое количество нейронных сетей
        #Вначале выбираем кандидатов для мутирования и генерим случайные индексы
        index_rand0 = np.zeros([mutant_amount])
        index_rand1 = np.zeros([mutant_amount])
        for i in range(mutant_amount):
            index_rand0[i] = np.random.randint(elite_amount, population_amount)
            index_rand1[i] = np.random.randint(elite_amount, population_amount)
        
        #Индекс для особей после elite_amount и до количества мутантов
        k=elite_amount
        for i in range(mutant_amount):
            #Добавляем шум в веса нейронных сетей первого агента
            ind0 = int (index_rand0[i])
            for par in q_network_pop0[ind0].parameters():
                noise = np.random.normal(size=par.data.size())
                noiseT = torch.FloatTensor(noise)
                par.data += noise_level * noiseT
            #Передаем мутированных особей новому поколению первого агента
            q_network_pop_next0[k] = q_network_pop0[ind0]
            
            #Добавляем шум в веса нейронных сетей второго агента
            ind1 = int (index_rand1[i])
            for par in q_network_pop1[ind1].parameters():
                noise = np.random.normal(size=par.data.size())
                noiseT = torch.FloatTensor(noise)
                par.data += noise_level * noiseT  
            #Передаем мутированных особей новому поколению второго агента
            q_network_pop_next1[k] = q_network_pop1[ind1]
            #Увеличивыаем счетчик
            k += 1
            
        ################_конец обучения_#######################################
       
    ################_конец цикла по эпизодам игры_#############################
    
    #Закрываем среду StarCraft II
    env.close()
    
    #Сохраняем параметры лучшей нейронной сети из популяции
    torch.save(q_network_pop_next0[0].state_dict(),"qnet_0.dat")
    torch.save(q_network_pop_next1[0].state_dict(),"qnet_1.dat")
        
    #Выводим на печать графики
    #Средняя награда
    plt.figure(num=None, figsize=(6, 3), dpi=150, facecolor='w', edgecolor='k')
    plt.plot(reward_history)
    plt.xlabel('Номер эпизода')
    plt.ylabel('Количество награды за эпизод')
    plt.show()
    #Процент побед
    plt.figure(num=None, figsize=(6, 3), dpi=150, facecolor='w', edgecolor='k')
    plt.plot(winrate_history)
    plt.xlabel('Номер эпизода')
    plt.ylabel('Процент побед')
    plt.show()

    #Частота использования определенной нейронной сети
    for pai in range(population_amount):
        bestindex_historyS.append(bestindex_historyA[pai])

    nnpops = [f"{i}" for i in range(population_amount)]
    plt.figure(num=None, figsize=(15, 9), dpi=150, facecolor='w', edgecolor='k')
    plt.bar(nnpops, bestindex_historyS)
    plt.xlabel('Популяция нейронных сетей')
    plt.ylabel('Частота попадания в элиту')
    plt.show()

#Точка входа в программу  
if __name__ == "__main__":
    start_time = time.time()
    main()
    #Время обучения
    print("--- %s минут ---" % ((time.time() - start_time)/60))
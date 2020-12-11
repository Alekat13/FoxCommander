

# Мультиагентное обучение с подкреплением
# Глава 2. Обучение в матричных и стохастических играх
# Реализация WoLF-PHC для матричной игры SMAC


from smac.env import StarCraft2Env
import numpy as np
import random 
import matplotlib.pyplot as plt

# Функция выбора действия в зависимости от стратегии и парамтера исследования 
def select_actionMatrix(pi_agent, epsilon):
    if random.uniform(0, 1) < (1 - epsilon):
        # Исследуем пространство действий
        action = np.random.choice([0,1])  
    else:
        #в зависимости от стратегии выбираем действия
        if pi_agent [0] > pi_agent [1]: 
            action = 0 
        elif pi_agent [0] < pi_agent [1]:
            action = 1 
        #если вероятность действий 0.5, то выбираем случайное действие
        elif pi_agent [0] == pi_agent [1]:
            action = np.random.choice([0,1])
    #возвращаем выбранное действие   
    return action      

# Функция задает матрицу наград игры SMAC         
def rewardMatrix (action_1, action_2):
    reward_1 = 0
    reward_2 = 0
    if (action_1, action_2) == (0, 0):
        reward_1 = 10
        reward_2 = 10
    elif (action_1, action_2) == (1, 0):
        reward_1 = -10
        reward_2 = -10
    elif (action_1, action_2) == (0, 1):
        reward_1 = 0
        reward_2 = 0
    elif (action_1, action_2) == (1, 1):
        reward_1 = 1
        reward_2 = 1
    return (reward_1, reward_2)    

#Главная функция программы, содержащая реализацию алгоритма WoLF-PHC и визуализацию в SMAC
def main():
    # определяем количество агентов-игроков
    n_agents = 2 
    
    n_actions_agent1 = 2 # количество действий первого агента
    n_actions_agent2 = 2 # количество действий второго агента
    
    # определяем количество эпизодов игры   
    n_episodes = 1000 
    # задаем параметр для модификации обучающих параметров
    timeStep = 0 
    #инициализируем параметры управляющие обучением
    alpha =  1 / (10 + 0.00001 * timeStep)    
    gamma = 0.9   
    epsilon = 0.5 / (1 + 0.0001 * timeStep) 
    delta = 0 
    delta_win = 0 
    delta_lose = 0 
       
    deltaAction = 0 # параметр для вычисления стратегии Pi первого агента
    deltaAction1 = 0 # параметр действия 1 для вычисления deltaAction
    deltaAction2 = 0 # параметр действия 2 для вычисления deltaAction
    deltaAction3 = 0 # параметр для вычисления стратегии Pi второго агента
    deltaAction4 = 0 # параметр действия 1 для вычисления deltaAction
    deltaAction5 = 0 # параметр действия 2 для вычисления deltaAction
    
    Q_table_agent1 = np.zeros([n_actions_agent1]) #задаем пустую q таблицу для агента 1
    Q_table_agent2 = np.zeros([n_actions_agent2])  #задаем пустую q таблицу для агента 2
    
    Pi_agent1 = np.zeros([n_actions_agent1]) #задаем пустую стратегию для агента 1
    Pi_agent2 = np.zeros([n_actions_agent2]) #задаем пустую стратегию для агента 2
    
    Pi_agent1_average = np.zeros([n_actions_agent1]) #задаем пустую среднюю стратегию для агента 1 WoLF-PHC
    Pi_agent2_average = np.zeros([n_actions_agent2]) #задаем пустую среднюю стратегию для агента 2 WoLF-PHC
    CountAction__agent1 = np.zeros([n_actions_agent1]) # подсчитываем количество выполнения действия, параметр для WoLF-PHC
    CountAction__agent2 = np.zeros([n_actions_agent2]) # подсчитываем количество выполнения действия, параметр для WoLF-PHC
    
    #сохраняем историю действий для вывода графика
    Pi_agent1_History = [] 
    Pi_agent2_History = []
    
    Pi_agent1 [0] = 0.5 # инициализируем стратегию агента 1 
    Pi_agent1 [1] = 0.5 # инициализируем стратегию агента 1 
    
    Pi_agent2 [0] = 0.5 # инициализируем стратегию агента 2 
    Pi_agent2 [1] = 0.5 # инициализируем стратегию агента 2 
    
    #цикл по эпизодам игры 
    for e in range(n_episodes):
       
        #если эпизод закончен то параметр = True
        terminated = False
        #инициализируем награду за эпизод
        episode_reward = 0
            
        #цикл внутри эпизода игры 
        while not terminated:
                     
            #обнуляем вспомогательные параметры 
            actions = []
            action = 0
            reward_agent1=0
            reward_agent2=0
            actionsFox = np.zeros([n_agents])
            #вычисляем обучающие параметры алгоритма 
            timeStep+=1
            alpha = 1 / (10 + 0.00001 * timeStep)
            epsilon = 0.5 / (1 + 0.0001 * timeStep)
            delta_win = 1.0 / (20000 + timeStep)
            delta_lose = 2.0 * delta_win
            
            #цикл по множеству агентов участвующих в игре 
            for agent_id in range(n_agents):
                
                #Выбираем действие агента в соответствии со стратегией 
                if agent_id == 0:
                    action = select_actionMatrix(Pi_agent1, epsilon)
                elif agent_id == 1:
                    action = select_actionMatrix(Pi_agent2, epsilon)
                
                #Собираем действия разных агентов для передачи в среду
                actions.append(action)
                actionsFox[agent_id] = action
            
            
            #Завершаем эпизод игры и получаем награду от среды - матрицы наград
            terminated = True
            reward_agent1,reward_agent2 = rewardMatrix (int(actionsFox[0]), int(actionsFox[1]))
            episode_reward += reward_agent1
            
            # Обновляем значения в Q-таблице
            Q_table_agent1 [int(actionsFox[0])] = (1-alpha)*Q_table_agent1[int(actionsFox[0])] + alpha*\
                                                  (reward_agent1 + gamma*np.max(Q_table_agent1[:]))
            
            Q_table_agent2 [int(actionsFox[1])] = (1-alpha)*Q_table_agent2[int(actionsFox[1])] + alpha*\
                                                  (reward_agent2 + gamma*np.max(Q_table_agent2[:])) 
             
            # Вычисляем среднюю стратегию 
            CountAction__agent1[int(actionsFox[0])]+=1
            CountAction__agent2[int(actionsFox[1])]+=1
            
            Pi_agent1_average[int(actionsFox[0])]= Pi_agent1_average [int(actionsFox[0])]+(1/CountAction__agent1[int(actionsFox[0])])*\
                                (Pi_agent1 [int(actionsFox[0])]- Pi_agent1_average[int(actionsFox[0])])
            
            Pi_agent2_average[int(actionsFox[1])]= Pi_agent2_average [int(actionsFox[1])]+(1/CountAction__agent2[int(actionsFox[1])])*\
                                (Pi_agent2 [int(actionsFox[1])]- Pi_agent2_average[int(actionsFox[1])])
            
            # Вычисляем быстрое или медленное delta для первого агента
            expected_value = 0
            expected_value_average = 0
            
            for aidx, _ in enumerate(Pi_agent1):
                expected_value += Pi_agent1[aidx]*Q_table_agent1[aidx]
                expected_value_average += Pi_agent1_average[aidx]*Q_table_agent1[aidx]
            
            if expected_value > expected_value_average:
                delta = delta_win 
            else:
                delta = delta_lose #
              
            # Обновляем стратгию Pi первого агента
                    
            deltaAction1 = np.min([Pi_agent1 [0], delta / (n_actions_agent1 - 1)])
            deltaAction2 = np.min([Pi_agent1 [1], delta / (n_actions_agent1 - 1)])
                   
             
            if  int(actionsFox[0]) != np.argmax (Q_table_agent1[:]):
                deltaAction = (-1)*deltaAction1 
            else:
                deltaAction = deltaAction2
                
            Pi_agent1 [0] = Pi_agent1 [0] + deltaAction
            #Проверка на то, чтобы значение вероятности оставалось в границах [0;1]
            if Pi_agent1 [0]>1: Pi_agent1 [0]= 1
            if Pi_agent1 [0]<0: Pi_agent1 [0]= 0
            #сумма вероятностей выбора действий всегда должна быть равна 1
            Pi_agent1 [1] = 1 - Pi_agent1 [0] 
            Pi_agent1_History.append(Pi_agent1 [0])
            
            # Вычисляем быстрое или медленное delta для второго агента 
            expected_value = 0
            expected_value_average = 0
            for aidx, _ in enumerate(Pi_agent2):
                expected_value += Pi_agent2[aidx]*Q_table_agent2[aidx]
                expected_value_average += Pi_agent2_average[aidx]*Q_table_agent2[aidx]
            if expected_value > expected_value_average:
                delta = delta_win 
            else:
                delta = delta_lose
            
            #Обновляем стратгию Pi второго агента
            deltaAction4 = np.min([Pi_agent2 [0], delta / (n_actions_agent2 - 1)])
            deltaAction5 = np.min([Pi_agent2 [1], delta / (n_actions_agent2 - 1)])
            if  int(actionsFox[1]) != np.argmax (Q_table_agent2[:]):
                deltaAction3 = (-1)*deltaAction4 
            else:
                deltaAction3 = deltaAction5
            Pi_agent2 [0] = Pi_agent2 [0] + deltaAction3
            if Pi_agent2 [0]>1: Pi_agent2 [0]= 1
            if Pi_agent2 [0]<0: Pi_agent2 [0]= 0
            Pi_agent2 [1] = 1 - Pi_agent2 [0] 
            Pi_agent2_History.append(Pi_agent2 [0])
            
        
        #Выводим на печать выученную стратегию действий в эпизоде
        print ("---------------------------------------------------------")
        print("Strategy Pi_agent1 in episode {} = {}".format(e, Pi_agent1))
        print("Strategy Pi_agent2 in episode {} = {}".format(e, Pi_agent2))
        
    #Выводим на печать выученные Q-таблицы
    print("Q_table_agent1=", Q_table_agent1)
    print("Q_table_agent2=", Q_table_agent2)
    
    #Выводим на печать графики
    plt.figure(num=None, figsize=(6, 3), dpi=100, facecolor='w', edgecolor='k')
    plt.plot(Pi_agent1_History)
    plt.xlabel('Номер итерации')
    plt.ylabel('Вероятность выполнить действие 1 агентом 2')
    plt.show()
    
    plt.figure(num=None, figsize=(6, 3), dpi=100, facecolor='w', edgecolor='k')
    plt.plot(Pi_agent2_History)
    plt.xlabel('Номер итерации')
    plt.ylabel('Вероятность выполнить действие 1 агентом 2')
    plt.show()
    
    #Визуализируем выученные стратегии в Starcraft II с помощью SMAC
    env = StarCraft2Env(map_name="2m2zFOXmatrix")
    env_info = env.get_env_info()
    n_agents = env_info["n_agents"]
    n_episodes = 3 # количество повторов
    #Визуализация стратегии Вперед, Вперед
    forward_both = np.array([[3, 3, 1, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1,
                          4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1,
                          4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1]])
    #Визуализация стратегии Прятаться, Прятаться  
    hide_both = np.array([[5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    #Визуализация стратегии Вперед, Прятаться  
    forward_hide = np.array([[3, 3, 1, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1,
                          4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    #Визуализация стратегии Прятаться, Вперед  
    hide_forward = np.array([[5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1,
                          4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1]])


    for e in range(n_episodes):
        
        env.reset()
        terminated = False
        episode_reward = 0
        n_steps = 1
        
        if (Pi_agent1[0]>Pi_agent1[1]) and (Pi_agent2[0]>Pi_agent2[1]):
            strategy = forward_both
        elif (Pi_agent1[1]>Pi_agent1[0]) and (Pi_agent2[1]>Pi_agent2[0]):
            strategy = hide_both
        elif (Pi_agent1[0]>Pi_agent1[1]) and (Pi_agent2[0]<Pi_agent2[1]):
            strategy = forward_hide
        elif (Pi_agent1[0]<Pi_agent1[1]) and (Pi_agent2[0]>Pi_agent2[1]):
            strategy = hide_forward
        
        while not terminated:
            actions = []
            
            for agent_id in range(n_agents):
                if n_steps < len(strategy[0]):
                    action = strategy[agent_id][n_steps - 1]
                else:
                    avail_actions = env.get_avail_agent_actions(agent_id)
                    if avail_actions[6] == 1:
                        action = 6
                    elif avail_actions[7] == 1:
                        action = 7
                    else:
                        avail_actions_ind = np.nonzero(avail_actions)[0]
                        action = np.random.choice(avail_actions_ind)
                actions.append(action)
            
            reward, terminated, _ = env.step(actions)
            reward = reward / n_steps
            n_steps += 1
            episode_reward += reward
            
    env.close() 
        
    
#Точка входа в программу    
if __name__ == "__main__":
    main()
 
    
    
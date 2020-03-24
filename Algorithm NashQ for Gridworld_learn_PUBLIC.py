# Alekat13 / FoxCommander 
# Глава 2. Обучение в матричных и стохастических играх
# Реализация алгоритма Q-обучения Нэша для стохастической игры в мире-сетки 3x3


import numpy as np
import random 
import matplotlib.pyplot as plt
import nashpy as nash
import pickle

#Вывод массива целиком
np.set_printoptions(threshold=np.inf)

#создаем глобальную переменную для подсчета совместных действий
def func5():
    if not hasattr(func5, '_state'):  # инициализация значения
        func5._state = 0
    print(func5._state)
    func5._state = func5._state + 1

#выбираем действия в зависимости от параметра эпсилон и значения Q-таблицы
def select_actionFox(agent_id, state, avail_actions_ind1, avail_actions_ind2, n_actionsFox, epsilon, Q_table):
    
    action = 0
    
    # исследуем пространство действий
    if random.uniform(0, 1) < (1 - epsilon): 
        if agent_id == 0:
            action = np.random.choice(avail_actions_ind1)  
        elif agent_id == 1:
            action = np.random.choice(avail_actions_ind2)  
    
    else:
        maxelement = -1000
        maxaction1 = 0
        maxaction2 = 0
        
        avail_actions_ind1int = 0
        avail_actions_ind2int = 0
        stateFoxint = int(state)
        
        #на основе действий выбираем значения из Q-таблицы
        if agent_id == 0:
                         
            for i in range(len(avail_actions_ind1)):
                for j in range(len(avail_actions_ind2)):
                    avail_actions_ind1int = int (avail_actions_ind1[i])
                    avail_actions_ind2int = int (avail_actions_ind2[j])
                    if maxelement < Q_table[stateFoxint, avail_actions_ind1int, avail_actions_ind2int]:
                        maxelement = Q_table[stateFoxint, avail_actions_ind1int, avail_actions_ind2int]
                        maxaction1 = avail_actions_ind1int
            
            action = maxaction1   
            
        elif agent_id == 1:
            
            for i in range(len(avail_actions_ind2)):
                for j in range(len(avail_actions_ind1)):
                    avail_actions_ind2int = int (avail_actions_ind2[i])
                    avail_actions_ind1int = int (avail_actions_ind1[j])
                    if maxelement < Q_table[stateFoxint, avail_actions_ind2int, avail_actions_ind1int]:
                        maxelement = Q_table[stateFoxint, avail_actions_ind2int, avail_actions_ind1int]
                        maxaction2 = avail_actions_ind2int
            
            action = maxaction2   
            
    return action         

#получаем состояние как позицию агента в клетке мира-сетки
def get_stateFox(agent_id, gridWorld, gridIndexes):
    
    state = 0
    
    if agent_id == 0:
        for i in range(len(gridWorld)):
            for j in range(len(gridWorld[i])):
                if gridWorld[i][j] == 1:
                    state = gridIndexes[i][j]
    
    if agent_id == 1:
        for i in range(len(gridWorld)):
            for j in range(len(gridWorld[i])):
                if gridWorld[i][j] == 2:
                    state = gridIndexes[i][j]
    
    return state

#получаем возможные действия агента в зависимости от состояния
def get_avail_agent_actionsFox(stateFox):
    
    agent_actions = []
    if stateFox == 0:
        agent_actions = [1, 2]
    elif stateFox == 1:
        agent_actions = [1, 2, 3]
    elif stateFox == 2:
        agent_actions = [2, 3]
    elif stateFox == 3:
        agent_actions = [0, 1, 2]
    elif stateFox == 4:
        agent_actions = [0, 1, 2, 3]
    elif stateFox == 5:
        agent_actions = [0, 2, 3]
    elif stateFox == 6:
        agent_actions = [0, 1]
    elif stateFox == 7:
        agent_actions = [0, 1, 3]
    elif stateFox == 8:
        agent_actions = [0, 3]
    
    return agent_actions

#выполняем действие агента в мире-сетке
def stepFox(actionsFox, stateFox, gridIndexes, gridWorld, Joint_Goalstate):
    
    terminated = False
    reward = np.zeros([2]) # инициализируем награду отдельную для каждого агента
    stateFoxOld = np.zeros([2])
    stateFoxOld[0] = stateFox[0]
    stateFoxOld[1] = stateFox[1]
    
    stateFoxNew = np.zeros([2])
    
    gridWorldOld = np.zeros([3, 3])
    
    for i in range(len(gridWorld)):
            for j in range(len(gridWorld[i])):
                gridWorldOld[i][j] = gridWorld[i][j]
    
    # перемещаем первого агента
    for i in range(len(gridIndexes)):
        for j in range(len(gridIndexes[i])):
            if gridIndexes[i][j] == stateFox[0]:
                if actionsFox[0] == 0:
                    gridWorld[i-1][j]=1
                    gridWorld[i][j]=0
                elif actionsFox[0] == 1:
                    gridWorld[i][j+1]=1
                    gridWorld[i][j]=0
                elif actionsFox[0] == 2:
                    gridWorld[i+1][j]=1
                    gridWorld[i][j]=0
                elif actionsFox[0] == 3:
                    gridWorld[i][j-1]=1
                    gridWorld[i][j]=0
    
    #находим новое состояние первого агента
    stateFoxNew[0] = get_stateFox(0, gridWorld, gridIndexes)
    
    # перемещаем второго агента
    for i in range(len(gridIndexes)):
        for j in range(len(gridIndexes[i])):
            if gridIndexes[i][j] == stateFox[1]:
                if actionsFox[1] == 0:
                    gridWorld[i-1][j]=2
                    gridWorld[i][j]=0
                elif actionsFox[1] == 1:
                    gridWorld[i][j+1]=2
                    gridWorld[i][j]=0
                elif actionsFox[1] == 2:
                    gridWorld[i+1][j]=2
                    gridWorld[i][j]=0
                elif actionsFox[1] == 3:
                    gridWorld[i][j-1]=2
                    gridWorld[i][j]=0 
    
    #находим новое состояние второго агента
    stateFoxNew[1] = get_stateFox(1, gridWorld, gridIndexes)
    
    if (stateFoxNew[0] == 7) and (stateFoxNew[1] == 7):
        Joint_Goalstate += 1
        print ('Количество совместных попаданий в целевое состояние:')
        func5() 
        
    # проверяем если агенты в целевом состоянии, то заканчиваем эпизод с наградой
    if (stateFoxNew[0] == 7):
        terminated = True
        reward[0] = 100
    elif (stateFoxNew[1] == 7):
        terminated = True
        reward[1] = 100
    # проверяем нет ли свопадений в одной клетке, то откат
    elif (stateFoxNew[0] == stateFoxNew[1]) and (stateFoxNew[0] != 7) and (stateFoxNew[1] != 7):
        reward[0] = -1
        reward[1] = -1
        stateFoxNew = stateFoxOld
        gridWorld = gridWorldOld
    
    return gridWorld, stateFoxNew, reward, terminated, Joint_Goalstate

   
#Главная функция программы, содержащая реализацию алгоритма Q-обучения Нэша
def main():
    
    # задаем размер мира-сетки
    world_height = 3
    world_width = 3
    
    # определяем индексы клеток мира-сетки
    gridIndexes = np.array([[0, 1, 2], 
                            [3, 4, 5],
                            [6, 7, 8]])
    
    #инициализируем мир-сетку нулями
    gridWorld = np.zeros([world_height, world_width])
    
    #задаем начальную позицию агента 1
    gridWorld [0][0] = 1
    
    #задаем начальную позицию агента 2
    gridWorld [0][2] = 2
    
    #задаем обозначение цели в состоянии 7 
    gridWorld [2][1] = 3
     
    # определяем количество агентов-игроков
    n_agents = 2 
    
    n_actions_agent1 = 4 # количество действий первого агента
    n_actions_agent2 = 4 # количество действий второго агента
    
    # определяем количество эпизодов игры   
    n_episodes = 5000  
    
    # количество состояний для одного агента в мире-сетке
    statesNumber = world_height*world_width
    
    # инициализируем параметры управляющие обучением
    Alpha1 = np.zeros([statesNumber, n_actions_agent1, n_actions_agent2])
    Alpha2 = np.zeros([statesNumber, n_actions_agent2, n_actions_agent1])
    gamma = 0.99  
    epsilon = 0.1  
     
    #задаем свою пустую Q-таблицу для агента 1
    Q_table_agent1_own = np.zeros([statesNumber, n_actions_agent1, n_actions_agent2])
        
    #задаем свою пустую Q-таблицу для агента 2
    Q_table_agent2_own = np.zeros([statesNumber, n_actions_agent2, n_actions_agent1])
        
    #задаем пустую стратегию для агента 1 (свою и оппонента)
    Pi_agent1 = np.zeros([n_actions_agent1*n_actions_agent2, n_actions_agent1]) 
    Pi_agent1_op = np.zeros([n_actions_agent1*n_actions_agent2, n_actions_agent1])
    
    #задаем пустую стратегию для агента 2 (свою и оппонента)
    Pi_agent2 = np.zeros([n_actions_agent1*n_actions_agent2, n_actions_agent2]) 
    Pi_agent2_op = np.zeros([n_actions_agent1*n_actions_agent2, n_actions_agent2]) 
    
    #задаем пустые значения NashQ для агента 1
    NashQ1 = np.zeros([n_actions_agent1, n_actions_agent2]) 
    #задаем пустые значения NashQ для агента 2
    NashQ2 = np.zeros([n_actions_agent2, n_actions_agent1])
    
    # подсчитываем количество попадания в состояние и выполнение совместного действия
    CountStateJAc__agent1 = np.ones([statesNumber, n_actions_agent1, n_actions_agent2])
    # подсчитываем количество попадания в состояние и выполнение совместного действия
    CountStateJAc__agent2 = np.ones([statesNumber, n_actions_agent2, n_actions_agent1]) 
    
    #сохраняем историю наград для вывода графика
    Reward_history_agent1 = [] 
    Reward_history_agent2 = []
    
    #сохраняем количество совместных попаданий в целевое состояние
    Joint_Goalstate = 0 
    Joint_Goalstate_history = [] 
 
    #цикл по эпизодам игры 
    for e in range(n_episodes):
        
        #инициализируем мир-сетку нулями
        gridWorld = np.zeros([world_height, world_width])
        
        #задаем начальную позицию агента 1
        gridWorld [0][0] = 1
        
        #задаем начальную позицию агента 2
        gridWorld [0][2] = 2
        
        #задаем обозначение цели в состоянии 7
        gridWorld [2][1] = 3
        
        print ("---------------Эпизод---------------")
        print ("-----------------", e,"-----------------")
        print ("Начальный мир-сетка")
        print (gridWorld)
      
        #если эпизод закончен то параметр = True
        terminated = False
        #инициализируем награду за эпизод
        episode_reward = 0
        #подсчитываем количество шагов в эпизоде
        stepnumber = 0
            
        #цикл внутри эпизода игры 
        while not terminated:
                     
            #обнуляем вспомогательные параметры 
            reward = np.zeros([n_agents]) 
            avail_actions_ind1 = []
            avail_actions_ind2 = []
            # храним историю действий один шаг для разных агентов
            actionsFox = np.zeros([n_agents])
            # храним историю состояний один шаг для разных агентов
            stateFox= np.zeros([n_agents])
                     
            #получаем состояния агентов как позицию в клетке мира-сетки
            stateFox[0] = get_stateFox(0, gridWorld, gridIndexes)
            stateFox[1] = get_stateFox(1, gridWorld, gridIndexes)
             
            #получаем возможные действия агентов
            avail_actions_ind1 = get_avail_agent_actionsFox(stateFox[0])
            avail_actions_ind2 = get_avail_agent_actionsFox(stateFox[1])
              
            #выбираем действия агентов в зависимости от параметра эпсилон и Q-таблицы
            actionsFox[0] = select_actionFox(0, stateFox[0], avail_actions_ind1, avail_actions_ind2, n_actions_agent1, epsilon, Q_table_agent1_own)
            actionsFox[1] = select_actionFox(1, stateFox[1], avail_actions_ind1, avail_actions_ind2, n_actions_agent2, epsilon, Q_table_agent2_own)
               
            # подсчитываем количество совместных действий в определенном состоянии
            CountStateJAc__agent1[int (stateFox[0]), int (actionsFox[0]), int (actionsFox[1])] += 1
            CountStateJAc__agent2[int (stateFox[1]), int (actionsFox[1]), int (actionsFox[0])] += 1
            
            #Завершаем эпизод игры и получаем награду от среды 
            gridWorld, stateFoxNew, reward, terminated, Joint_Goalstate = stepFox(actionsFox, stateFox, gridIndexes, gridWorld, Joint_Goalstate)
            stepnumber += 1
            
            if terminated == True:
                Reward_history_agent1.append(reward [0])
                Reward_history_agent2.append(reward [1])
                Joint_Goalstate_history.append(Joint_Goalstate)
            
            episode_reward += reward[0]+reward[1]
            
            print ("Шаг: ", stepnumber)
            print (gridWorld) 
            print ('Награда за эпизод: ', episode_reward)
            
            ###################_Обучаем_##############################################
    
            #получаем возможные действия агентов в новом состоянии
            avail_actions_ind1 = []
            avail_actions_ind2 = []
            avail_actions_ind1 = get_avail_agent_actionsFox(stateFoxNew[0])
            avail_actions_ind2 = get_avail_agent_actionsFox(stateFoxNew[1])
           
            #инициализируем матрицу наград
            rewardMatrix1 = np.zeros([n_actions_agent1, n_actions_agent2]) 
            rewardMatrix2 = np.zeros([n_actions_agent2, n_actions_agent1]) 
            
            #выбираем данные из первой Q-таблицы для первого агента
            avail_actions_ind1index = 0
            avail_actions_ind2index = 0
            stateFoxNewint = int (stateFoxNew[0])
            for i in range(len(avail_actions_ind1)):
                for j in range(len(avail_actions_ind2)):
                    avail_actions_ind1index = int (avail_actions_ind1[i])
                    avail_actions_ind2index = int (avail_actions_ind2[j])
                    rewardMatrix1[avail_actions_ind1index][avail_actions_ind2index] = Q_table_agent1_own[stateFoxNewint, avail_actions_ind1index, avail_actions_ind2index]
                    
            #выбираем данные из второй Q-таблицы для второго агента
            avail_actions_ind2index = 0
            avail_actions_ind1index = 0
            stateFoxNewint = int (stateFoxNew[1])
            for i in range(len(avail_actions_ind2)):
                for j in range(len(avail_actions_ind1)):
                    avail_actions_ind2index = int (avail_actions_ind2[i])
                    avail_actions_ind1index = int (avail_actions_ind1[j])
                    rewardMatrix2[avail_actions_ind2index][avail_actions_ind1index] = Q_table_agent2_own[stateFoxNewint, avail_actions_ind2index, avail_actions_ind1index]
           
            ##################################################################
            # вычисляем стратегии pi как равновесия Нэша матричной игры с помощью библиотеки nashpy
            # равновесия Нэша для первого агента
            rps = nash.Game(rewardMatrix1, rewardMatrix2)  
            
            equilibria = rps.support_enumeration()
            
            #передаем данные из генератора в массив
            tempi = 0
            tempj = 0
            for eq in equilibria:   
                for tempj in range(4):
                    Pi_agent1[tempi][tempj] = eq[0][tempj]
                    Pi_agent1_op[tempi][tempj] = eq[1][tempj]
                    tempj += 1
                tempi += 1
                tempj = 0
            
            # равновесия Нэша для второго агента
            rps2 = nash.Game(rewardMatrix2, rewardMatrix1)  
            
            equilibria2 = rps2.support_enumeration()
            
            tempi = 0
            tempj = 0
            for eq in equilibria2:   
                for tempj in range(4):
                    Pi_agent2[tempi][tempj] = eq[0][tempj]
                    Pi_agent2_op[tempi][tempj] = eq[1][tempj]
                    tempj += 1
                tempi += 1
                tempj = 0
        
            ###############################################################
            #извлекаем индексы ненулевых действий из равновесных стратегий
            indexAgent1Action1 = []
            indexAgent1Action2 = []
            
            for i in range(len(Pi_agent1)):
                for j in range(len(Pi_agent1[i])):
                    if Pi_agent1[i][j] != 0:
                        indexAgent1Action1.append(j)
                        
            for k in range(len(Pi_agent1_op)):
                for l in range(len(Pi_agent1_op[k])):
                    if Pi_agent1_op[k][l] != 0:
                        indexAgent1Action2.append(l)
            
            ################################################################
            # вычисляем NashQ и обновляем Q-таблицу для первого агента
            stateFoxNewint = int (stateFoxNew[0])
            NashQ1 = np.zeros([n_actions_agent1, n_actions_agent2]) 
            for i in range(len(indexAgent1Action1)):
                for j in range(len(indexAgent1Action2)):
                    NashQ1[indexAgent1Action1[i], indexAgent1Action2[j]] = Q_table_agent1_own [stateFoxNewint, indexAgent1Action1[i], indexAgent1Action2[j]]
                    Alpha1[stateFoxNewint, indexAgent1Action1[i], indexAgent1Action2[j]] = 1 / CountStateJAc__agent1[stateFoxNewint, indexAgent1Action1[i], indexAgent1Action2[j]]
                    Q_table_agent1_own [stateFoxNewint, indexAgent1Action1[i], indexAgent1Action2[j]] = (1-Alpha1[stateFoxNewint, indexAgent1Action1[i], indexAgent1Action2[j]])*Q_table_agent1_own [stateFoxNewint,indexAgent1Action1[i], indexAgent1Action2[j]] + \
                                                                Alpha1[stateFoxNewint, indexAgent1Action1[i], indexAgent1Action2[j]]*(reward[0] + gamma*NashQ1[indexAgent1Action1[i], indexAgent1Action2[j]])
                   
            indexAgent2Action1 = []
            indexAgent2Action2 = []
            
            for i in range(len(Pi_agent2)):
                for j in range(len(Pi_agent2[i])):
                    if Pi_agent2[i][j] != 0:
                        indexAgent2Action1.append(j)
                        
            for k in range(len(Pi_agent2_op)):
                for l in range(len(Pi_agent2_op[k])):
                    if Pi_agent2_op[k][l] != 0:
                        indexAgent2Action2.append(l)
            
            ################################################################
            # вычисляем NashQ и обновляем Q-таблицу для второго агента
            stateFoxNewint = int (stateFoxNew[1])
            NashQ2 = np.zeros([n_actions_agent2, n_actions_agent1]) 
            for i in range(len(indexAgent2Action1)):
                for j in range(len(indexAgent2Action2)):
                    NashQ2[indexAgent2Action1[i], indexAgent2Action2[j]] = Q_table_agent2_own [stateFoxNewint, indexAgent2Action1[i], indexAgent2Action2[j]]
                    Alpha2[stateFoxNewint, indexAgent2Action1[i], indexAgent2Action2[j]] = 1 / CountStateJAc__agent2[stateFoxNewint, indexAgent2Action1[i], indexAgent2Action2[j]]
                    Q_table_agent2_own [stateFoxNewint, indexAgent2Action1[i], indexAgent2Action2[j]] = (1-Alpha2[stateFoxNewint, indexAgent2Action1[i], indexAgent2Action2[j]])*Q_table_agent2_own [stateFoxNewint,indexAgent2Action1[i], indexAgent2Action2[j]] + \
                                                                Alpha2[stateFoxNewint, indexAgent2Action1[i], indexAgent2Action2[j]]*(reward[0] + gamma*NashQ2[indexAgent2Action1[i], indexAgent2Action2[j]])
                            
   
    #Выводим на печать выученные Q-таблицы
    print("Q_table_agent1_own=", Q_table_agent1_own)
    print("Q_table_agent2_own=", Q_table_agent2_own)
  
    with open("se31_1.pkl", 'wb') as f:
        pickle.dump(Q_table_agent1_own, f)
       
    with open("se31_2.pkl", 'wb') as f:
        pickle.dump(Q_table_agent2_own, f)
        
    #Выводим на печать график
    plt.figure(num=None, figsize=(6, 3), dpi=150, facecolor='w', edgecolor='k')
    plt.plot(Joint_Goalstate_history)
    plt.xlabel('Номер итерации')
    plt.ylabel('Количество равновесий Нэша')
    plt.show()
   
#Точка входа в программу    
if __name__ == "__main__":
    main()



 
    
    

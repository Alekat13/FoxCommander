# Alekat13 / FoxCommander 
# Глава 2. Обучение в матричных и стохастических играх
# Реализация алгоритма Q-обучения Нэша для стохастической игры в мире-сетки 3x3


import numpy as np
import pickle

#Вывод массива целиком
np.set_printoptions(threshold=np.inf)


def select_actionFox(agent_id, state, avail_actions_ind1, avail_actions_ind2, n_actionsFox, epsilon, Q_table):
    
    action = 0
    
    # исследуем пространство действий
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

#получаем состояние агента как позицию в клетке мира-сетки
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
def stepFox(actionsFox, stateFox, gridIndexes, gridWorld):
    
    terminated = False
    # инициализируем награду отдельную для каждого агента
    reward = np.zeros([2]) 
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
          
    
    return gridWorld, stateFoxNew, reward, terminated

    
   
#Главная функция программы, содержащая реализацию мира-сетки и действий на оснвое обученных Q-таблиц
def main():
    # задаем параметры мира-сетки
    world_height = 3
    world_width = 3
    
    # индекс клеток мира-сетки
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
    n_episodes = 1
    
    # инициализируем параметры управляющие обучением
    epsilon = 0.7
   
    total_reward = 0
    
    #загружаем обученные Q-таблицы
    with open("se31_1.pkl", 'rb') as f:
        Q_table_agent1_own = pickle.load(f)
        print ("Q_table_agent1_own:")
        print (Q_table_agent1_own)
    
    with open("se31_2.pkl", 'rb') as f:
        Q_table_agent2_own = pickle.load(f)
        print ("Q_table_agent2_own:")
        print (Q_table_agent2_own)
        
 
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
        
        print ("---------------Episode---------------")
        print ("-----------------", e,"-----------------")
        print ("Initial gridWorld")
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
            actionsFox = np.zeros([n_agents]) # храним историю действий один шаг для разных агентов
            stateFox= np.zeros([n_agents]) # храним историю состояний один шаг для разных агентов
            
            #----цикл по множеству агентов участвующих в игре 
                            
            #получаем состояния агентов как позицию в клетке мира-сетки
            stateFox[0] = get_stateFox(0, gridWorld, gridIndexes)
            stateFox[1] = get_stateFox(1, gridWorld, gridIndexes)
            
            #получаем возможные дейтвия агентов
            avail_actions_ind1 = get_avail_agent_actionsFox(stateFox[0])
            avail_actions_ind2 = get_avail_agent_actionsFox(stateFox[1])
             
            #выбираем действия агентов в зависимости от параметра эпсилон и Q-таблицы
            actionsFox[0] = select_actionFox(0, stateFox[0], avail_actions_ind1, avail_actions_ind2, n_actions_agent1, epsilon, Q_table_agent1_own)
            actionsFox[1] = select_actionFox(1, stateFox[1], avail_actions_ind1, avail_actions_ind2, n_actions_agent2, epsilon, Q_table_agent2_own)
           
            #Завершаем эпизод игры и получаем награду от среды 
        
            gridWorld, stateFoxNew, reward, terminated = stepFox(actionsFox, stateFox, gridIndexes, gridWorld)
            stepnumber += 1
            episode_reward += reward[0]+reward[1]
            
            print ("Step ", stepnumber)
            print (gridWorld) 
            print ('episode_reward=', episode_reward)
            
        total_reward +=episode_reward 
       
    
    #Выводим на печать среднюю награду
    print ("Average reward = ", total_reward/n_episodes)
   
#Точка входа в программу    
if __name__ == "__main__":
    main()

 
    
    

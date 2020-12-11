# -*- coding: utf-8 -*-
# Мультиагентное обучение с подкреплением
# Глава 1. Независимое табличное Q-обучение
# Реализация алгоритма Q-обучения для карты SMAC

from smac.env import StarCraft2Env
import numpy as np
import random 
import pickle

#Вывод массива целиком
np.set_printoptions(threshold=np.inf)

#получаем состояние агента как позицию на карте
def get_stateFox(agent_id, agent_posX, agent_posY):
    
    if agent_id == 0:
        state = 3
        
        if 6 < agent_posX < 7 and 15 < agent_posY < 16.5 :
            state = 0
        elif 7 < agent_posX < 8 and 15 < agent_posY < 16.5  :
            state = 1
        elif 8 < agent_posX < 8.9 and 15 < agent_posY < 16.5  :
            state = 2
        elif 8.9 < agent_posX < 9.1 and 15 < agent_posY < 16.5  :
            state = 3
        elif 9.1 < agent_posX < 10 and 15 < agent_posY < 16.5  :
            state = 4
        elif 10 < agent_posX < 11 and 15 < agent_posY < 16.5  :
            state = 5
        elif 11 < agent_posX < 12 and 15 < agent_posY < 16.5  :
            state = 6
        elif 12 < agent_posX < 13.1 and 15 < agent_posY < 16.5  :
            state = 7
        elif 6 < agent_posX < 7 and 14 < agent_posY < 15 :
            state = 8
        elif 7 < agent_posX < 8 and 14 < agent_posY < 15 :
            state = 9
        elif 8 < agent_posX < 8.9 and 14 < agent_posY < 15 :
            state = 10
        elif 8.9 < agent_posX < 9.1 and 14 < agent_posY < 15 :
            state = 11
        elif 9.1 < agent_posX < 10 and 14 < agent_posY < 15 :
            state = 12
        elif 10 < agent_posX < 11 and 14 < agent_posY < 15 :
            state = 13
        elif 11 < agent_posX < 12 and 14 < agent_posY < 15 :
            state = 14
        elif 12 < agent_posX < 13.1 and 14 < agent_posY < 15 :
            state = 15
        
    if agent_id == 1:
        state = 11
        
        if 6 < agent_posX < 7 and 16.2 < agent_posY < 17 :
            state = 0
        elif 7 < agent_posX < 8 and 16.2 < agent_posY < 17 :
            state = 1
        elif 8 < agent_posX < 8.9 and 16.2 < agent_posY < 17 :
            state = 2
        elif 8.9 < agent_posX < 9.1 and 16.2 < agent_posY < 17 :
            state = 3
        elif 9.1 < agent_posX < 10 and 16.2 < agent_posY < 17 :
            state = 4
        elif 10 < agent_posX < 11 and 16.2 < agent_posY < 17 :
            state = 5
        elif 11 < agent_posX < 12 and 16.2 < agent_posY < 17 :
            state = 6
        elif 12 < agent_posX < 13.1 and 16.2 < agent_posY < 17 :
            state = 7
        elif 6 < agent_posX < 7 and 15.5 < agent_posY < 16.2 :
            state = 8
        elif 7 < agent_posX < 8 and 15.5 < agent_posY < 16.2 :
            state = 9
        elif 8 < agent_posX < 8.9 and 15.5 < agent_posY < 16.2 :
            state = 10
        elif 8.9 < agent_posX < 9.1 and 15.5 < agent_posY < 16.2 :
            state = 11
        elif 9.1 < agent_posX < 10 and 15.5 < agent_posY < 16.2 :
            state = 12
        elif 10 < agent_posX < 11 and 15.5 < agent_posY < 16.2 :
            state = 13
        elif 11 < agent_posX < 12 and 15.5 < agent_posY < 16.2 :
            state = 14
        elif 12 < agent_posX < 13.1 and 15.5 < agent_posY < 16.2 :
            state = 15
   
    return state

#выбираем действие
def select_actionFox(agent_id, state, avail_actions_ind, n_actionsFox, epsilon, Q_table):
    
    if random.uniform(0, 1) < (1 - epsilon):
        # Исследуем пространство действий
        action = np.random.choice(avail_actions_ind)  
    else:
        # Выбираем действие с исопльзованием Q-таблицы
        qt_arr = np.zeros(len(avail_actions_ind))
        keys = np.arange(len(avail_actions_ind))
        act_ind_decode = dict(zip(keys, avail_actions_ind))
        stateFoxint = int(state)
        for act_ind in range(len(avail_actions_ind)):
            qt_arr[act_ind] = Q_table[agent_id, stateFoxint, act_ind_decode[act_ind]]
           
        action = act_ind_decode[np.argmax(qt_arr)]  
    
    return action      


#MAIN
def main():
    #Загружаем среду StarCraft II, карту и минимальную сложность противника 
    env = StarCraft2Env(map_name="2m2zFOX", difficulty="1")
    #Получаем информацию о среде
    env_info = env.get_env_info()

    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
   
    n_episodes = 120 # количество эпизодов 
    alpha = 0.1    #скорость обучения
    gamma = 0.9   #скидочный фактор
    epsilon = 0.7 #исследование пространства действий
    
    n_statesFox = 16 # количество состояний мира-сетки
    n_actionsFox = 7 # вводим свое количество действий, которые понадобятся
    
    Q_table = np.zeros([n_agents, n_statesFox, n_actions]) #задаем пустую q таблицу

    for e in range(n_episodes):
        # Обнуляем среду
        env.reset()
        # Параметр равен True, когда битва заканчивается
        terminated = False
        episode_reward = 0

        #используем динамический параметр эпсилон
        if e % 15 == 0:
            epsilon += (1 - epsilon) * 10 / n_episodes
            print("epsilon = ", epsilon)
      
        while not terminated:
            # обнуляем промежуточные переменные
            actions = []
            action = 0
            # храним историю состояний один шаг для разных агентов
            stateFox= np.zeros([n_agents]) 
            # храним историю действий один шаг для разных агентов
            actionsFox = np.zeros([n_agents]) 

            # agent_id= 0, agent_id= 1
            for agent_id in range(n_agents):
                
                #получаем характеристики юнита
                unit = env.get_unit_by_id(agent_id)
                #получаем состояние по координатам юнита
                stateFox[agent_id] = get_stateFox(agent_id, unit.pos.x, unit.pos.y)
    
                #получаем возможные действия агента
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                # выбираем действие
                action = select_actionFox(agent_id, stateFox[agent_id], avail_actions_ind, n_actionsFox, epsilon, Q_table)
                #собираем действия от разных агентов
                actions.append(action)
                actionsFox[agent_id] = action
  
            #получаем награду и прерывание игры от среды
            reward, terminated, _ = env.step(actions)
         
            episode_reward += reward
            
            ###################_Обучаем_##############################################
            
            for agent_id in range(n_agents):
                #получаем характеристики юнита
                unit = env.get_unit_by_id(agent_id)
                #получаем состояние по координатам юнита
                stateFox_next = get_stateFox(agent_id, unit.pos.x, unit.pos.y)
                stateFoxint= int(stateFox[agent_id])
                actionsFoxint = int(actionsFox[agent_id])
                #Вычиялем значения для Q-таблицы
                Q_table[agent_id, stateFoxint, actionsFoxint] = Q_table[agent_id, stateFoxint, actionsFoxint] + alpha * \
                             (reward + gamma * np.max(Q_table[agent_id, stateFox_next, :]) - Q_table[agent_id, stateFoxint, actionsFoxint])
            
            ##########################################################################            
       
        #Выводим общую награду за эпизод
        print("Total reward in episode {} = {}".format(e, episode_reward))
        #Выводим результаты игр
        print ("get_stats()=", env.get_stats())
    
    
    #Закрываем среду StarCraft II
    env.close()
    #Выводис и сохраняем выученную Q-таблицу
    print(Q_table)
    with open("se25.pkl", 'wb') as f:
        pickle.dump(Q_table, f)
    
#Точка входа в программу    
if __name__ == "__main__":
    main()
 
    
    

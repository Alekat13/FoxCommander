# -*- coding: utf-8 -*-

# А.Н. Алфимцев Мультиагентное обучение
# Глава 1. Независимое табличное Q-обучение
# Тестирование алгоритма Q-обучения для карты SMAC

from smac.env import StarCraft2Env
import numpy as np
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


def select_actionFox(agent_id, state, avail_actions_ind, n_actionsFox, epsilon, Q_table):
    
    qt_arr = np.zeros(len(avail_actions_ind))
    #Функция arange() возвращает одномерный массив с равномерно разнесенными значениями внутри заданного интервала. 
    keys = np.arange(len(avail_actions_ind))
    #Функция zip объединяет в кортежи элементы из последовательностей переданных в качестве аргументов.
    act_ind_decode = dict(zip(keys, avail_actions_ind))
    stateFoxint = int(state)
    
    for act_ind in range(len(avail_actions_ind)):
        qt_arr[act_ind] = Q_table[agent_id, stateFoxint, act_ind_decode[act_ind]]

    #Returns the indices of the maximum values along an axis.
    # Exploit learned values
    action = act_ind_decode[np.argmax(qt_arr)]  
    
    return action
    

#MAIN
def main():
    
    env = StarCraft2Env(map_name="2m2zFOX", difficulty="1")
   
    env_info = env.get_env_info()
    
    n_agents = env_info["n_agents"]
   
    n_episodes = 10 # количество эпизодов

    epsilon = 0.7 #e-greedy sayon - 0.3 больш - 0.7 lapan = = 1.0 (100% random actions)
 
    n_actionsFox = 7 # вводим свое количество действий, которые понадобятся

    total_reward = 0
    #Загружаем обученную Q-таблицу
    with open("se25.pkl", 'rb') as f:
        Q_table = pickle.load(f)
        print (Q_table)

    for e in range(n_episodes):
        env.reset()
        terminated = False
        episode_reward = 0
        actions_history = []
        
        while not terminated:
            actions = []
            action = 0
            stateFox= np.zeros([n_agents]) # храним историю состояний один шаг для разных агентов
            actionsFox = np.zeros([n_agents]) # храним историю действий один шаг для разных агентов

            '''agent_id= 0, agent_id= 1'''
            for agent_id in range(n_agents):
                  
                #получаем характеристики юнита
                unit = env.get_unit_by_id(agent_id)
                #получаем состояние по координатам юнита
                stateFox[agent_id] = get_stateFox(agent_id, unit.pos.x, unit.pos.y)
                
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                # выбираем действие
                action = select_actionFox(agent_id, stateFox[agent_id], avail_actions_ind, n_actionsFox, epsilon, Q_table)
                #собираем действия от разных агентов
                actions.append(action)
                actions_history.append(action)
                
                actionsFox[agent_id] = action
                
            reward, terminated, _ = env.step(actions)
            episode_reward += reward
 
        total_reward +=episode_reward 
        print("Total reward in episode {} = {}".format(e, episode_reward))
        print ("get_stats()=", env.get_stats())
        print("actions_history=", actions_history)

    print ("Average reward = ", total_reward/n_episodes)
    env.close()
    
    
    
if __name__ == "__main__":
    main()
 
    
    
#Мультиагентное обучение с подкреплением
#Глава 3. Нейросетевое обучение
#Алгоритм CDQN 

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

#Определяем размер псведокарты
MAPX = 32
MAPY = 32

#Флаг вывода массива целиком
np.set_printoptions(threshold=np.inf)

#Определяем архитектуру нейронной сети
#Реализация нейронной сети основана на работе
#M. Lapan Deep Reinforcement Learning Hands On 
class CDQN(nn.Module):
    #На вход нейронная сеть получает наблюдение из среды в виде изображения 
    #На выходе нейронная сеть возвращает вероятности действий
    def __init__(self, input_shape, n_actions):
        super(CDQN, self).__init__()        
        self.conv = nn.Sequential(
            #Первый сверточный слой на вход получает данные в формате 1x32x32
            #на выходе возвращает данные в формате 1x32x32
            #Размер сверточного фильтра равен 8, шаг фильтра равен 3
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=3),
            nn.ReLU(),
            #Второй сверточный слой на вход получает данные в формате 1x32x32
            #на выходе возвращает данные в формате 1x64x64
            #Размер сверточного фильтра равен 4, шаг фильтра равен 2
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            #Третий сверточный слой на вход получает данные в формате 1x64x64
            #на выходе возвращает данные в формате 1x64x64 
            #Размер сверточного фильтра равен 3, шаг фильтра равен 1
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
   
        #Преобразуем 3D выход сверточного слоя в 2D для передачи данных линейным слоям 
        conv_out_size = self._get_conv_out(input_shape)
        
        #Линейные слои обрабатывают данные после сверточных слоев
        self.fc = nn.Sequential(
            #Первый линейный слой c входным размером данных 64 и выходным 64    
            nn.Linear(conv_out_size, 64),
            nn.ReLU(),
            #Второй линейный слой c входным размером данных 64 и выходным 64 
            nn.Linear(64, 64),
            nn.ReLU(),
            #Третий линейный слой c входным размером данных 64 и выходным 128
            nn.Linear(64, 128),
            nn.ReLU(),
            #Четвертый линейный слой c входным размером данных 128 и выходным 343
            nn.Linear(128, n_actions)
        )
    
    #Функция преобразования сверточных данных для передачи линейным слоям
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))    
    #Определяем последовательность обработки данных
    def forward(self, x):
        #В начале данные подаются в сверточные слои
        conv_out = self.conv(x).view(x.size()[0], -1)
        #На выход передаются данные, обработанные линейными слоями
        return self.fc(conv_out)


           
#Создаем минивыборку определенного объема из буфера воспроизведения       
def sample_from_expbuf(experience_buffer, batch_size):
    #Функция возвращает случайную последовательность заданной длины
    perm_batch = np.random.permutation(len(experience_buffer))[:batch_size]
    #Минивыборка
    experience = np.array(experience_buffer)[perm_batch]
    #Возвращаем значения минивыборки по частям
    return experience[:,0], experience[:,1], experience[:,2], experience[:,3], experience[:,4]       

#Выбираем возможное действие с максимальным Q-значением в зависимости от 
#эпсилон
def select_actionFox(actionag, avail_actions_ind, epsilon):
    p = np.random.random(1).squeeze()
    #Исследуем пространство действий
    if np.random.rand() < epsilon:
        return np.random.choice (avail_actions_ind) 
    else:
        #Находим возможное действие:
        #Проверяем есть ли действие в доступных действиях агента
        if actionag in avail_actions_ind:
            return actionag
        else:
            return np.random.choice (avail_actions_ind)
         


#Перекодируем выход нейронной сети Q(s,a1,a2,..,an) в действия
#для 2 и 3 агентов
def decode_actions(action_probabilities, n_agents, n_actions):
    actionsFox = np.zeros([n_agents])
    maxindex_out = np.argmax(action_probabilities)
    a1=0
    a2=0
    a3=0
    indexmain=0
    stop = 0
    #для двух агентов
    if n_agents == 2:
        while stop == 0:
            for a1 in range(n_actions-1):
                for a2 in range(n_actions-1):
                    if maxindex_out == indexmain:
                        actionsFox[0] = a1
                        actionsFox[1] = a2
                        stop = 1
                        return actionsFox 
                    else: indexmain+=1
    #для трех агентов
    elif n_agents == 3:
        while not stop:
            for a1 in range(n_actions-1):
                for a2 in range(n_actions-1):
                    for a3 in range(n_actions-1):
                        if maxindex_out == indexmain:
                            actionsFox[0] = a1
                            actionsFox[1] = a2
                            actionsFox[2] = a3
                            stop = 1
                            return actionsFox 
                        else: indexmain+=1
   

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
    #Количество дружественных агентов
    n_agents = env_info["n_agents"]
    #Количество отдельных действий каждого агента
    n_actions = env_info["n_actions"]
    
    #Определяем основные параметры нейросетевого обучения
    ###########################################################################
    #Определим динамический эпсилон с затуханием
    eps_max = 1.0 #Начальное значение эпсилон
    eps_min = 0.1 #Финальное значение эпсилон
    eps_decay_steps = 15000 #Шаг затухания эпсилон
    #Основные переходы в алгоритме CDQN зависят от шагов игры
    global_step = 0 #подсчитываем общее количество шагов в игре
    copy_steps = 100 #каждые 100 шагов синхронизируем нейронные сети 
    start_steps = 2000 #начинаем обучать через 2000 шагов
    steps_train = 4 #после начала обучения продолжаем обучать каждый 4 шаг 
    #Размер минивыборки 
    batch_size = 32 
    #Общее количество эпизодов игры
    n_episodes = 210 
    #Параметр дисконтирования
    gamma = 0.99 
    #Скорость обучения
    alpha = 0.01 
    #Объем буфера воспроизведения
    buffer_len = 10000 
    ###########################################################################   
    
    #############################_Создаем нейронные сети_######################
    #Создаем буфер воспроизведения единый для всех агентов 
    experience_buffer = deque(maxlen=buffer_len)
    #Pytorch определяет возможность использования графического процессора
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    #Определим псевдокарту как общее наблюдение среды для обучения нейронной сети
    map_data_NNinput = np.zeros((MAPX, MAPY, 1), np.uint8)
    print ('map_data_NNinput.shape=',map_data_NNinput.shape)
    #Изменим псевдокарту для правильной загрузки в нейронную сеть
    map_data_NNinputR = map_data_NNinput.reshape(1, MAPX, MAPY)
    print ('map_data_NNinputR.shape=', map_data_NNinputR.shape)
    #Вычислим выходной размер нейронной сети Q(s,a1,a2,a..,an)
    qofa_out = 1
    for i in range(n_agents):
        qofa_out = qofa_out*n_actions 
    print ('Выходной размер нейронной сети Q(s,a1,a2,a..,an)', qofa_out)
    #Создаем основную нейронную сеть
    q_network = CDQN(map_data_NNinputR.shape, qofa_out).to(device)
    #Создаем целевую нейронную сеть
    tgt_network = CDQN(map_data_NNinputR.shape, qofa_out).to(device)
    print ('CDQN=', q_network)
    #Создаем оптимизатор нейронной сети
    optimizer = optim.Adam(q_network.parameters(), lr=alpha)
    #Создаем функцию потерь
    objective = nn.MSELoss()
    ##########################################################################
        
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
        ######################_цикл while_#################################
        while not terminated:
            #Определим псевдокарту 32x32x3 как общее наблюдение среды 
            map_data = np.zeros((MAPY, MAPX, 3), np.uint8)
            #Обнуляем промежуточные переменные
            actions = []
            action = 0
            #Храним историю действий один шаг для разных агентов
            actionsFox = np.zeros([n_agents]) 
            
            #Соберем данные противников для псведокарты
            for e_id, e_unit in env.enemies.items():
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                #Нарисуем агентов на псведокарте BGR
                cv2.circle(map_data, (int(e_x), int(e_y)), 1, (255, 0, 0), -1) 
            
            #Соберем данные агентов для псведокарты
            for agent_id in range(n_agents):
                #Получаем характеристики юнита
                unit = env.get_unit_by_id(agent_id)
                #Нарисуем агентов на псведокарте BGR
                cv2.circle(map_data, (int(unit.pos.x), int(unit.pos.y)), 1, (0, 255, 0), -1) 
            
            #Переводим изображение псевдокарты в полутоновый формат
            imggrayscale = cv2.cvtColor(map_data,cv2.COLOR_RGB2GRAY)
            #Изменим псевдокарту для правильной загрузки в нейронную сеть
            map_data_NNinputR = imggrayscale.reshape(1, MAPX, MAPY)
            
            #Конвертируем данные в тензор
            obs_agentT = torch.FloatTensor([map_data_NNinputR]).to(device)
            #Передаем состояние среды в основную нейронную сеть 
            #и получаем Q-значения для каждого действия
            action_probabilitiesT = q_network(obs_agentT)
            #Конвертируем данные в numpy
            action_probabilitiesT = action_probabilitiesT.to("cpu")
            action_probabilities = action_probabilitiesT.data.numpy()[0]
            #Разделим общий выход Q(s,a1,a2,...,an) для каждого агента
            actionsFox = decode_actions (action_probabilities, n_agents, n_actions)
            
            ###########_Цикл по агентам для выполнения действий в игре_########
            for agent_id in range(n_agents):
                #Находим возможные действия агента в данный момент времени 
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                #Выбираем возможное действие агента с учетом
                #максимального Q-значения и параметра эпсилон
                action = select_actionFox(actionsFox[agent_id], avail_actions_ind, epsilon)
                #Обрабатываем исключение при ошибке в возможных действиях
                if action is None: 
                    action = np.random.choice (avail_actions_ind)
                #Рисуем оранжевую линию на псевдокарте, если агент стреляет
                if (action == 6) or (action==7) or (action==8):
                    cv2.line(map_data, (int(unit.pos.x+1), int(unit.pos.y)), (int(unit.pos.x+3), int(unit.pos.y)), (0, 130, 255), 1)
                #Собираем действия от разных агентов                 
                actions.append(action)
                actionsFox[agent_id] = action
            ########_конец цикла по агентам для выполнения действий в игре_####

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
            cv2.imshow('PseudoMap', resized)
            cv2.waitKey(1)
            
            #Переводим изображение псевдокарты в полутоновый формат
            #с добавленным выстрелом для сохранения в буфере воспроизведения
            imggrayscale = cv2.cvtColor(map_data,cv2.COLOR_RGB2GRAY)
            map_data_NNinputR = imggrayscale.reshape(1, MAPX, MAPY)
            #Получаем новое наблюдение из среды
            map_data_next = np.zeros((MAPY, MAPX, 3), np.uint8)
            #Соберем данные противников для псведокарты
            for e_id, e_unit in env.enemies.items():
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                cv2.circle(map_data_next, (int(e_x), int(e_y)), 1, (255, 0, 0), -1)
            #Соберем данные агентов для псведокарты
            for agent_id in range(n_agents):
                unit = env.get_unit_by_id(agent_id)
                cv2.circle(map_data_next, (int(unit.pos.x), int(unit.pos.y)), 1, (0, 255, 0), -1)
            #Переводим изображение в полутоновый формат
            imggrayscale_next = cv2.cvtColor(map_data_next,cv2.COLOR_RGB2GRAY)
            #Изменим псевдокарту для правильной загрузки в нейронную сеть
            map_data_NNinputR_next = imggrayscale_next.reshape(1, MAPX, MAPY)
            
            #Сохраняем переход в буфере воспроизведения
            experience_buffer.append([map_data_NNinputR, actionsFox, map_data_NNinputR_next, reward, terminated])
            
            #Если буфер воспроизведения наполнен, начинаем обучать сеть
            ###################################################################
            if (global_step % steps_train == 0) and (global_step > start_steps):
                #Получаем минивыборку из буфера воспроизведения
                exp_obs, exp_act, exp_obs_next, exp_rew, exp_termd = sample_from_expbuf(experience_buffer, batch_size)
                #Конвертируем данные в тензор
                obs_agentT = torch.FloatTensor([exp_obs]).to(device)
                    
                #Подаем минивыборку в основную нейронную сеть
                action_probabilitiesT = q_network(obs_agentT.squeeze(0))
                action_probabilitiesT = action_probabilitiesT.to("cpu")
                action_probabilities = action_probabilitiesT.data.numpy()[0]
                
                #Конвертируем данные в тензор
                obs_agentT_next = torch.FloatTensor([exp_obs_next]).to(device)
                #Подаем минивыборку в целевую нейронную сеть
                action_probabilitiesT_next = tgt_network(obs_agentT_next.squeeze(0))
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
                optimizer.zero_grad()
                
                #Вычисляем функцию потерь
                loss_t = objective(action_probabilitiesT, y_batchT.squeeze(0))
                
                #Сохраняем данные для графиков
                Loss_History.append(loss_t) 
                loss_n=loss_t.data.numpy()
                total_loss.append(loss_n)
                m_loss.append(np.mean(total_loss[-10000:]))
                                 
                #Выполняем обратное распространение ошибки
                loss_t.backward()
                
                #Выполняем оптимизацию нейронной сети
                optimizer.step()
                
            ######################_конец if буфера воспроизведения_############
                
            #Синхронизируем веса основной и целевой нейронной сети
            #каждые 100 шагов
            if (global_step+1) % copy_steps == 0 and global_step > start_steps:
                tgt_network.load_state_dict(q_network.state_dict())
            
            #Обновляем счетчик общего количества шагов
            global_step += 1
           
        #####################_конец цикла while_###############################
        
        #Выводим счетчик шагов игры и общую награду за эпизод
        print('global_step =', global_step, "Total reward in episode {} = {}".format(e, episode_reward))
        
        #Собираем данные для графиков
        Reward_History.append(episode_reward)
        status = env.get_stats()
        winrate_history.append(status["win_rate"])
        
    ################_конец цикла for по эпизодам_##############################
    
    #Закрываем среду StarCraft II
    env.close()
    #Убираем с экрана псевдокарту
    cv2.destroyAllWindows()
    
    #Сохраняем параметры обученной нейронной сети
    torch.save(q_network.state_dict(),"qnetCNN.dat") 
    
    #Выводим среднее количество шагов за эпизод
    print('Среднее количество шагов =', global_step/(e+1))
    
    #Выводим на печать графики
    plt.figure(num=None, figsize=(6, 3), dpi=150, facecolor='w', edgecolor='k')
    plt.plot(Reward_History)
    plt.xlabel('Номер эпизода')
    plt.ylabel('Количество награды за эпизод')
    plt.show()
    
    plt.figure(num=None, figsize=(6, 3), dpi=150, facecolor='w', edgecolor='k')
    plt.plot(winrate_history)
    plt.xlabel('Номер эпизода')
    plt.ylabel('Процент побед')
    plt.show()
    
    plt.figure(num=None, figsize=(6, 3), dpi=150, facecolor='w', edgecolor='k')
    plt.plot(m_loss)
    plt.xlabel('Номер каждой 10000 итерации')
    plt.ylabel('Функция потерь')
    plt.show()
    
if __name__ == "__main__":
    start_time = time.time()
    main() 
    print("--- %s минут ---" % ((time.time() - start_time)/60))
from DDPG_network import Agent
from Robotic_env import RobotEnv
import random
import numpy as np

if __name__ == '__main__':
    # 设定学习率
    LR_A = 0.0001  # learning rate for actor
    LR_C = 0.001  # learning rate for critic
    # 设定机器人和AP个数
    numberOfRobots = 5
    numberOfAPs = 4
    # 初始化环境
    env = RobotEnv(numOfRobo=numberOfRobots, numOfAPs=numberOfAPs)  # number of robots and AP

    # 初始化机器人AC网络
    Robot0 = Agent('actor0', 'critic0', 'actor0_target', 'critic0_target', Lr_A=LR_A, Lr_C=LR_C, input_dims=[192],
                   tau=0.001, n_actions=8)
    Robot1 = Agent('actor1', 'critic1', 'actor1_target', 'critic1_target', Lr_A=LR_A, Lr_C=LR_C, input_dims=[192],
                   tau=0.001, n_actions=8)
    Robot2 = Agent('actor2', 'critic2', 'actor2_target', 'critic2_target', Lr_A=LR_A, Lr_C=LR_C, input_dims=[192],
                   tau=0.001, n_actions=8)
    Robot3 = Agent('actor3', 'critic3', 'actor3_target', 'critic3_target', Lr_A=LR_A, Lr_C=LR_C, input_dims=[192],
                   tau=0.001, n_actions=8)
    Robot4 = Agent('actor4', 'critic4', 'actor4_target', 'critic4_target', Lr_A=LR_A, Lr_C=LR_C, input_dims=[192],
                   tau=0.001, n_actions=8)

    # 初始化机器人路径规划
    Robot0_mover = Agent('actor0_mover', 'critic0_mover', 'actor0_mover_target', 'critic0_mover_target', Lr_A=LR_A,
                         Lr_C=LR_C, input_dims=[192], tau=0.001, n_actions=2)
    Robot1_mover = Agent('actor1_mover', 'critic1_mover', 'actor1_mover_target', 'critic1_mover_target', Lr_A=LR_A,
                         Lr_C=LR_C, input_dims=[192], tau=0.001, n_actions=2)
    Robot2_mover = Agent('actor2_mover', 'critic2_mover', 'actor2_mover_target', 'critic2_mover_target', Lr_A=LR_A,
                         Lr_C=LR_C, input_dims=[192], tau=0.001, n_actions=2)
    Robot3_mover = Agent('actor3_mover', 'critic3_mover', 'actor3_mover_target', 'critic3_mover_target', Lr_A=LR_A,
                         Lr_C=LR_C, input_dims=[192], tau=0.001, n_actions=2)
    Robot4_mover = Agent('actor4_mover', 'critic4_mover', 'actor4_mover_target', 'critic4_mover_target', Lr_A=LR_A,
                         Lr_C=LR_C, input_dims=[192], tau=0.001, n_actions=2)

    # 初始化AP的AC网络
    AP1 = Agent('actor1_AP', 'critic1_AP', 'actor1_AP_target', 'critic1_AP_target', Lr_A=LR_A, Lr_C=LR_C,
                input_dims=[192], tau=0.001, n_actions=8)
    AP2 = Agent('actor2_AP', 'critic2_AP', 'actor2_AP_target', 'critic2_AP_target', Lr_A=LR_A, Lr_C=LR_C,
                input_dims=[192], tau=0.001, n_actions=8)
    AP3 = Agent('actor3_AP', 'critic3_AP', 'actor3_AP_target', 'critic3_AP_target', Lr_A=LR_A, Lr_C=LR_C,
                input_dims=[192], tau=0.001, n_actions=8)
    AP4 = Agent('actor4_AP', 'critic4_AP', 'actor4_AP_target', 'critic4_AP_target', Lr_A=LR_A, Lr_C=LR_C,
                input_dims=[192], tau=0.001, n_actions=8)

    # 初始化奖励、时延、能量、拒绝
    # 步数100，探索100000
    score_Robot = 0
    score_AP = 0
    score1_history = []
    score0_history = []
    delay_history = []
    energy_history = []
    reject_history = []
    steps = 100
    Explore = 100000.
    epsilon = 1
    epsilon_move = 1
    i = 0
    warmed = 0
    new_state = 0

    # 累计循环1000回合
    for i in range(5):
        obs = env.reset()
        obs_move = obs
        a = 0
        test = 0  # this is for debugging, and when it is '1', we can choose action to debug
        print("new turn ready!!!!!!!!!!!!")
        # 每次回合运动1000时间步
        for move in range(1000):
            epsilon_move -= 1 / Explore
            if np.random.random() < epsilon_move:
                act0_robot_move = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                act1_robot_move = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                act2_robot_move = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                act3_robot_move = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                act4_robot_move = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)

            else:
                act0_robot_move = Robot0_mover.choose_action(obs_move)
                act1_robot_move = Robot1_mover.choose_action(obs_move)
                act2_robot_move = Robot2_mover.choose_action(obs_move)
                act3_robot_move = Robot3_mover.choose_action(obs_move)
                act4_robot_move = Robot4_mover.choose_action(obs_move)

            # 每次运动训练中进行100次任务卸载和机器人运动
            for j in range(100):
                # while not done:
                epsilon -= 1 / Explore
                a += 1
                if np.random.random() <= epsilon:
                    act0_robot = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act1_robot = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act2_robot = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act3_robot = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act4_robot = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
                    act0_AP = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0,1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0,1.0), np.random.uniform(-1.0, 1.0)
                    act1_AP = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0,1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0,1.0), np.random.uniform(-1.0, 1.0)
                    act2_AP = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0,1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0,1.0), np.random.uniform(-1.0, 1.0)
                    act3_AP = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0,1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0,1.0), np.random.uniform(-1.0, 1.0)

                else:
                    act0_robot = Robot0.choose_action(obs)
                    act1_robot = Robot1.choose_action(obs)
                    act2_robot = Robot2.choose_action(obs)
                    act3_robot = Robot3.choose_action(obs)
                    act4_robot = Robot4.choose_action(obs)
                    act0_AP = AP1.choose_action(obs)
                    act1_AP = AP2.choose_action(obs)
                    act2_AP = AP3.choose_action(obs)
                    act3_AP = AP4.choose_action(obs)

                # 整合所有智能体到一维数组中
                act_robots = np.concatenate([act0_robot, act1_robot, act2_robot, act3_robot, act4_robot])
                act_robo_move = np.concatenate([act0_robot_move, act1_robot_move, act2_robot_move, act3_robot_move, act4_robot_move])
                act_APs = np.concatenate([act0_AP, act1_AP, act2_AP, act3_AP])
                # 进行任务卸载
                new_state, reward_Robot, reward_AP, done, info, accept, AoI, energy, posX0, posY0, posX1, posY1 = env.step_task_offloading(act_APs, act_robots, act_robo_move)
                # 任务卸载回忆
                Robot0.remember(obs, act0_robot, reward_Robot[0], new_state, done)
                Robot1.remember(obs, act1_robot, reward_Robot[1], new_state, done)
                Robot2.remember(obs, act2_robot, reward_Robot[2], new_state, done)
                Robot3.remember(obs, act3_robot, reward_Robot[3], new_state, done)
                Robot4.remember(obs, act4_robot, reward_Robot[4], new_state, done)
                AP1.remember(obs, act0_AP, reward_AP[0], new_state, done)
                AP2.remember(obs, act1_AP, reward_AP[1], new_state, done)
                AP3.remember(obs, act2_AP, reward_AP[2], new_state, done)
                AP4.remember(obs, act3_AP, reward_AP[3], new_state, done)

                # 任务卸载学习
                Robot0.learn()
                Robot1.learn()
                Robot2.learn()
                Robot3.learn()
                Robot4.learn()
                AP1.learn()
                AP2.learn()
                AP3.learn()
                AP4.learn()

                # 取平均值
                score_Robot += np.average(reward_Robot)
                score_AP += np.average(reward_AP)
                if j%10 == 0:  # 每10次任务卸载记录一次各项奖励
                    with open("00-accept_LR_high_quantize_Energy_agressive_VoI_Reject.txt", 'a') as reward_APs:
                        reward_APs.write(str(accept) + '\n')
                    with open("00-AoI_LR_high_quantize_Energy_agressive_VoI_Reject.txt", 'a') as AoI_file:
                        AoI_file.write(str(AoI) + '\n')
                    with open("00-energy_LR_high_quantize_Energy_agressive_VoI_Reject.txt", 'a') as energy_file:
                        energy_file.write(str(energy) + '\n')
                obs = new_state
                obs_move = new_state

            # 每100次任务卸载记录一次运动位置、得分
            print("move is ", move)
            print("Robots reward = ", score_Robot)
            print("APs reward = ", score_AP)
            with open("00-reward_robot_LR_high_quantize_Energy_agressive_VoI_Reject.txt", 'a') as reward_robots:
                reward_robots.write(str(score_Robot) + '\n')
            with open("00-reward_AP_LR_high_quantize_Energy_agressive_VoI_Reject.txt", 'a') as reward_APs:
                reward_APs.write(str(score_AP) + '\n')
            with open("00-pos_0_high_Energy_quantize_agressive_VoI_Reject.txt", 'a') as pos0:   # 第1个机器人的坐标
                pos0.write(str(posX0) + ', ' + str(posY0) + '\n')
            with open("00-pos_1_high_Energy_quantize_agressive_VoI_Reject.txt", 'a') as pos1:   # 第2个机器人的坐标
                pos1.write(str(posX1) + ', ' + str(posY1) + '\n')

            # 路径规划回忆
            Robot0_mover.remember(obs, act0_robot_move, score_Robot, new_state, False)
            Robot1_mover.remember(obs, act1_robot_move, score_Robot, new_state, False)
            Robot2_mover.remember(obs, act2_robot_move, score_Robot, new_state, False)
            Robot3_mover.remember(obs, act3_robot_move, score_Robot, new_state, False)
            Robot4_mover.remember(obs, act4_robot_move, score_Robot, new_state, False)

            # 路径规划学习
            Robot0_mover.learn()
            Robot1_mover.learn()
            Robot2_mover.learn()
            Robot3_mover.learn()
            Robot4_mover.learn()

            # 清零得分
            score_Robot = 0
            score_AP = 0
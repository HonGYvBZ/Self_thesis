from time import sleep

from MADRL import MADDPG
import matplotlib.pyplot as plt
from mec_env1 import MecEnv1
import sys

MAX_EPISODES = 500
EPISODES_BEFORE_TRAIN = 1
NUM_OF_EVAL_EPISODES = 50

DONE_PENALTY = None

ENV_SEED = 37   # system seed
NUMofTDS = 20   # terminal device
NUMofESS = 1    # edge server

AREA_X = 1000
AREA_Y = 1000

def plot_all_segments(env):
    """绘制所有设备的线段"""
    plt.figure(figsize=(10, 8))

    # 设置坐标轴范围
    plt.xlim(0, AREA_X)
    plt.ylim(0, AREA_Y)

    # 绘制所有线段
    for i in range(env.n_tds):
        # 绘制线段（蓝色实线）
        plt.plot([env.S_start_X[i], env.S_end_X[i]],
                 [env.S_start_Y[i], env.S_end_Y[i]],
                 'b-', alpha=0.6, label='Segment' if i == 0 else "")

        # 标记起点（红色圆圈）和终点（绿色叉号）
        plt.scatter(env.S_start_X[i], env.S_start_Y[i], c='red', marker='o',
                    label='Start' if i == 0 else "")
        plt.scatter(env.S_end_X[i], env.S_end_Y[i], c='green', marker='x',
                    s=100, label='End' if i == 0 else "")

    # 添加图例和标签
    plt.title(f"Movement Segments of {env.n_tds} Devices")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()

    # 显示图形
    plt.grid(True)
    plt.show()

def create_ddpg(Index_of_Result, env, env_eval, EPISODES_BEFORE_TRAIN, MAX_EPISODES):
    maddpg = MADDPG(Index_of_Result=Index_of_Result, env=env, env_eval=env_eval, n_agents=env.n_agents,
                          state_dim=env.state_size, action_dim=env.action_size,
                          action_lower_bound=env.action_lower_bound, action_higher_bound=env.action_higher_bound,
                          episodes_before_train=EPISODES_BEFORE_TRAIN, epsilon_decay=MAX_EPISODES)

    maddpg.interact(MAX_EPISODES, EPISODES_BEFORE_TRAIN, NUM_OF_EVAL_EPISODES)
    return maddpg


def plot_ddpg(ddpg, parameter, variable="reward"):
    plt.figure()
    if (variable == "reward"):
        for i in range(len(ddpg)):
            plt.plot(ddpg[i].episodes, ddpg[i].mean_rewards)
            plt.xlabel("Episode")
            plt.ylabel("Reward")

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(["MADDPG"])
    plt.savefig("./figure/MaDDPG_run%s.png" % parameter)


def run(Index_of_Result):
    env = MecEnv1(n_tds=NUMofTDS, n_ess=1, area_x=AREA_X, area_y=AREA_Y, env_seed=ENV_SEED)
    plot_all_segments(env)
    while(True):
        print("please stop")
        sleep(2)
    eval_env = MecEnv1(n_tds=NUMofTDS, n_ess=1, area_x=AREA_X, area_y=AREA_Y, env_seed=ENV_SEED)
    ddpg = [create_ddpg(Index_of_Result, env, eval_env, EPISODES_BEFORE_TRAIN, MAX_EPISODES)]
    plot_ddpg(ddpg, "_%s" % Index_of_Result)


if __name__ == "__main__":
    Index_of_Result = sys.argv[1]  # set run run_number for indexing results,
    run(Index_of_Result)
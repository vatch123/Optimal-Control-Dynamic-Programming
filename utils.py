import os
import numpy as np
import gymnasium as gym
import pickle
import matplotlib.pyplot as plt
import imageio
import random
from minigrid.core.world_object import Goal, Key, Door



def step(env, action):
    """
    Take Action
    ----------------------------------
    actions:
        0 # Move forward (MF)
        1 # Turn left (TL)
        2 # Turn right (TR)
        3 # Pickup the key (PK)
        4 # Unlock the door (UD)
    """
    actions = {
        0: env.actions.forward,
        1: env.actions.left,
        2: env.actions.right,
        3: env.actions.pickup,
        4: env.actions.toggle,
    }

    (obs, reward, terminated, truncated, info) = env.step(actions[action])
    return terminated


def generate_random_env(seed, task):
    """
    Generate a random environment for testing
    -----------------------------------------
    seed:
        A Positive Integer,
        the same seed always produces the same environment
    task:
        'MiniGrid-DoorKey-5x5-v0'
        'MiniGrid-DoorKey-6x6-v0'
        'MiniGrid-DoorKey-8x8-v0'
    """
    if seed < 0:
        seed = np.random.randint(50)
    env = gym.make(task, render_mode="rgb_array")
    env.reset(seed=seed)
    return env


def load_env(path):
    """
    Load Environments
    ---------------------------------------------
    Returns:
        gym-environment, info
    """
    with open(path, "rb") as f:
        env = pickle.load(f)

    info = {"height": env.height, "width": env.width, "init_agent_pos": env.agent_pos, "init_agent_dir": env.dir_vec}

    for i in range(env.height):
        for j in range(env.width):
            if isinstance(env.grid.get(j, i), Key):
                info["key_pos"] = np.array([j, i])
            elif isinstance(env.grid.get(j, i), Door):
                info["door_pos"] = np.array([j, i])
            elif isinstance(env.grid.get(j, i), Goal):
                info["goal_pos"] = np.array([j, i])

    return env, info


def load_random_env(num):
    """
    Load a random DoorKey environment
    ---------------------------------------------
    Returns:
        gym-environment, info
    """
    # env_list = [os.path.join(env_folder, env_file) for env_file in os.listdir(env_folder)]
    # env_path = random.choice(env_list)
    env_path = f'envs/random_envs/DoorKey-8x8-{num}.env'
    with open(env_path, "rb") as f:
        env = pickle.load(f)

    info = {
        "height": env.height,
        "width": env.width,
        "init_agent_pos": env.agent_pos,
        "init_agent_dir": env.dir_vec,
        "door_pos": [],
        "door_open": [],
    }

    for i in range(env.height):
        for j in range(env.width):
            if isinstance(env.grid.get(j, i), Key):
                info["key_pos"] = np.array([j, i])
            elif isinstance(env.grid.get(j, i), Door):
                info["door_pos"].append(np.array([j, i]))
                if env.grid.get(j, i).is_open:
                    info["door_open"].append(True)
                else:
                    info["door_open"].append(False)
            elif isinstance(env.grid.get(j, i), Goal):
                info["goal_pos"] = np.array([j, i])

    return env, info, env_path


def save_env(env, path):
    with open(path, "wb") as f:
        pickle.dump(env, f)


def plot_env(env):
    """
    Plot current environment
    ----------------------------------
    """
    img = env.render()
    plt.figure()
    plt.imshow(img)
    plt.show()


def draw_gif_from_seq(seq, env, pathFolder="./gif/doorkey.gif"):
    """
    Save gif with a given action sequence
    ----------------------------------------
    seq:
        Action sequence, e.g [0,0,0,0] or [MF, MF, MF, MF]

    env:
        The doorkey environment
    """
    gif_path = pathFolder + "trajectory.gif"
    frame = 1
    with imageio.get_writer(gif_path, mode="I", duration=0.8) as writer:
        img = env.render()
        # plt.imsave(pathFolder+f"{frame}.png", img, dpi=200)
        writer.append_data(img)
        for act in seq:
            img = env.render()
            plt.imsave(pathFolder+f"{frame}.png", img, dpi=200)
            step(env, act)
            writer.append_data(img)
            frame += 1
    print(f"GIF is written to {gif_path}")
    return

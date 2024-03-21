from config import *
from utils import *
import numpy as np
from minigrid.core.world_object import Goal, Key, Door, Wall


def stage_cost(env, state, act):
    """
    Defines stage cost for all state-action pairs for the known environments
    """
    j,i,key,dir,door1 = state
    nj = j + inv_dir_map[dir][0]
    ni = i + inv_dir_map[dir][1]
    if act == MF:
        if isinstance(env.grid.get(nj, ni), Wall):
            return np.inf
        elif isinstance(env.grid.get(nj, ni), Key):
            return np.inf if key==0 else 1
        elif isinstance(env.grid.get(nj, ni), Door):
            if door1==0:
                return np.inf
        return 1
    elif act==TR or act==TL:
        return 1
    elif act==PK:
        if isinstance(env.grid.get(nj, ni), Key):
            return np.inf if key==1 else 1
        else:
            return np.inf
    elif act==UD:
        if isinstance(env.grid.get(nj, ni), Door):
            if door1==1:
                return np.inf
            elif door1==0 and key==1:
                return 1
            else:
                return np.inf
        else:
            return np.inf

def value_iteration(env):
    """
    Runs value iteration on a given known environment
    """
    # Create the value function with the terminal cost
    key_options = 2
    direction_options = 4
    first_door_options = 2
    value_fn = np.zeros((env.height, env.width, key_options, direction_options, first_door_options))
    for i in range(env.height):
        for j in range(env.width):
            if isinstance(env.grid.get(j,i), Goal):
                value_fn[j,i,:,:,:] = -100
            else:
                value_fn[j,i,:,:,:] = np.inf

    Q = np.stack([value_fn]*5, axis=-1)
    T = env.width * env.height
    P = []
    V = [value_fn]
    Q_func = [Q]
    for t in range(T-1,0,-1):
        value_fn_t = np.copy(V[-1])
        p_t = np.zeros_like(value_fn_t)
        for key in range(1,-1,-1):
            for door1 in range(1,-1,-1):
                for i in range(env.height):
                    for j in range(env.width):
                        if isinstance(env.grid.get(j,i), (Wall, Goal)):
                            continue
                        if isinstance(env.grid.get(j,i), Key) and key==0:
                            continue
                        for dir in range(4):
                            state = (j,i,key,dir,door1)
                            for act in [MF,TL,TR,PK,UD]:
                                if act == MF:
                                    nj = j + inv_dir_map[dir][0]
                                    ni = i + inv_dir_map[dir][1]
                                    Q[j,i,key,dir,door1,act] = stage_cost(env, state, act) + value_fn[nj,ni,key,dir,door1]
                                if act==TR or act==TL:
                                    nd = dir_map[rot_map[act][inv_dir_map[dir]]]
                                    Q[j,i,key,dir,door1,act] = stage_cost(env, state, act) + value_fn[j,i,key,nd,door1]
                                if act==PK:
                                    Q[j,i,key,dir,door1,act] = stage_cost(env, state, act) + value_fn[j,i,1,dir,door1]
                                if act==UD:
                                    Q[j,i,key,dir,door1,act] = stage_cost(env, state, act) + value_fn[j,i,key,dir,1]
                            value_fn_t[j,i,key,dir,door1] = min(Q[j,i,key,dir,door1])
                            p_t[j,i,key,dir,door1] = np.argmin(Q[j,i,key,dir,door1])
                value_fn = value_fn_t
        if len(V) >2 and np.all(V[-1]==V[-2]):
            break
        V.append(value_fn)
        P.append(p_t)
        Q_func.append(Q)

    P = P[::-1]
    V = V[::-1]
    Q_func = Q_func[::-1]

    return P, V, Q_func


def doorkey_problemA(env, info):
    """
    You are required to find the optimal path in
        doorkey-5x5-normal.env
        doorkey-6x6-normal.env
        doorkey-8x8-normal.env

        doorkey-6x6-direct.env
        doorkey-8x8-direct.env

        doorkey-6x6-shortcut.env
        doorkey-8x8-shortcut.env

    Feel Free to modify this fuction
    """

    P, V, Q_func = value_iteration(env)

    # Generate optimal sequence
    seq = []
    for t in range(len(P)):
        cur_pos = env.agent_pos
        cur_dir = tuple(env.dir_vec)
        key = 1 if env.carrying is not None else 0
        door1 = 1 if env.grid.get(info["door_pos"][0],info["door_pos"][1]).is_open else 0
        print(cur_pos, cur_dir, key, door1)
        print(V[t][:,:,key,dir_map[cur_dir], door1].T)
        print(P[t][:,:,key,dir_map[cur_dir],door1].T)
        act = P[t][cur_pos[0], cur_pos[1], key, dir_map[cur_dir], door1]
        seq.append(act)
        done = step(env, act)
        if done:
            break

    seq.append(MF)
    return seq

def save_gif_and_frames(env_path, seq):
    env, info = load_env(env_path)
    env_name = env_path.rsplit('/')[-1].split('.')[0]
    env.grid.get(info["door_pos"][0],info["door_pos"][1]).is_locked = True
    pathFolder=f"gif/{env_name}/"
    os.makedirs(pathFolder,exist_ok=True)
    draw_gif_from_seq(seq, env, pathFolder=pathFolder)
    seq_str = ', '.join([inv_act_map[x] for x in seq[:-1]])

    with open(pathFolder + "sequence.txt", "w") as text_file:
        text_file.write(seq_str)
    


def partA(env_name):
    env_folder = "envs/known_envs"
    env_path = os.path.join(os.getcwd(), env_folder, f"{env_name}.env")
    env, info = load_env(env_path)
    env.grid.get(info["door_pos"][0],info["door_pos"][1]).is_locked = True
    seq = doorkey_problemA(env, info)  # find the optimal action sequence
    save_gif_and_frames(env_path, seq)


if __name__ == "__main__":
    env_name = "doorkey-6x6-shortcut"
    partA(env_name)

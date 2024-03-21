from config import *
from utils import *
import numpy as np
from minigrid.core.world_object import Goal, Key, Door, Wall

WALL = 0
FREE = 1
KEY = 2
DOOR = 3
GOAL = 4

def grid_definition():
    grid = np.ones((8,8))

    grid[:,4] = WALL

    grid[2,4] = DOOR
    grid[5,4] = DOOR

    grid[1,1] = KEY
    grid[3,2] = KEY
    grid[6,1] = KEY

    grid[1,5] = GOAL
    grid[3,6] = GOAL
    grid[6,5] = GOAL

    return grid


def stage_cost(grid, state, act):
    try:    
        j,i,key,dir,door1,door2,key_pos,goal_pos = state
        nj = j + inv_dir_map[dir][1]
        ni = i + inv_dir_map[dir][0]
        if nj<0 or ni<0:
            raise IndexError
        if act == MF:
            if grid[nj, ni] == WALL:
                return np.inf
            elif grid[nj, ni] == KEY and key_pos == key_pos_map[(nj,ni)]:
                return np.inf if key==0 else 1
            elif grid[nj, ni] == DOOR:
                door = door1 if door_pos_map[(nj,ni)] == 0 else door2
                if door==0:
                    return np.inf
            return 1
        elif act==TL or act==TR:
            return 1
        elif act==PK:
            if grid[nj, ni] == KEY and key_pos == key_pos_map[(nj,ni)]:
                if key == 1:
                    return np.inf
                else:
                    return 1
            else:
                return np.inf
        elif act==UD:
            if grid[nj, ni] == DOOR:
                door = door1 if door_pos_map[(nj,ni)] == 0 else door2
                if door==1:
                    return np.inf
                elif door==0 and key==1:
                        return 1
                else:
                    return np.inf
            else:
                return np.inf
    except IndexError:
        return np.inf


def value_iteration(grid):

    key_options = 2
    direction_options = 4
    first_door_options = 2
    second_door_options = 2
    key_positions = 3
    goal_positions = 3

    value_fn = np.ones(
        (
            8,
            8,
            key_options,
            direction_options,
            first_door_options,
            second_door_options,
            key_positions,
            goal_positions
        )
    ) * np.inf

    for i in range(8):
        for j in range(8):
            if grid[j,i] == GOAL:
                value_fn[j,i,:,:,:,:,:,goal_pos_map[(j,i)]] = -100
            else:
                value_fn[j,i,:,:,:,:,:,:] = np.inf
    

    Q = np.stack([value_fn]*5, axis=-1)
    T = 64
    P = []
    V = [value_fn]
    Q_func = [Q]
    for t in range(T-1,0,-1):
        value_fn_t = np.copy(V[-1])
        p_t = np.ones_like(value_fn_t) * np.inf
        for goal_pos in range(goal_positions):
            for key_pos in range(key_positions):
                for key in range(1,-1,-1):
                    for door1 in range(1,-1,-1):
                        for door2 in range(1,-1,-1):
                            for i in range(8):
                                for j in range(8):
                                    if grid[j,i] == WALL or grid[j,i] == GOAL and goal_pos == goal_pos_map[(j,i)]:
                                        continue
                                    # if grid[j,i] == KEY and key_pos == key_pos_map[(j,i)] and key==0:
                                    #     continue
                                    for dir in range(4):
                                        state = (j,i,key,dir,door1,door2,key_pos,goal_pos)
                                        for act in [MF,TL,TR,PK,UD]:
                                            try:
                                                nj = j + inv_dir_map[dir][1]
                                                ni = i + inv_dir_map[dir][0]
                                                if nj<0 or ni<0:
                                                    raise IndexError
                                                if act == MF:
                                                    v_next = value_fn[nj,ni,key,dir,door1,door2,key_pos,goal_pos]
                                                if act==TR or act==TL:
                                                    nd = dir_map[rot_map[act][inv_dir_map[dir]]]
                                                    v_next = value_fn[j,i,key,nd,door1,door2,key_pos,goal_pos]
                                                if act==PK:
                                                    v_next = value_fn[j,i,1,dir,door1,door2,key_pos,goal_pos]
                                                if act==UD:
                                                    if grid[nj, ni] == DOOR and door_pos_map[(nj,ni)] == 0:
                                                        v_next = value_fn[j,i,key,dir,1,door2,key_pos,goal_pos]
                                                    elif grid[nj, ni] == DOOR and door_pos_map[(nj,ni)] == 1:
                                                        v_next = value_fn[j,i,key,dir,door1,1,key_pos,goal_pos]
                                                    else:
                                                        v_next = value_fn[j,i,key,dir,door1,door2,key_pos,goal_pos]
                                            except IndexError:
                                                v_next = np.inf
                                            Q[j,i,key,dir,door1,door2,key_pos,goal_pos,act] = stage_cost(grid, state, act) + v_next

                                        value_fn_t[j,i,key,dir,door1,door2,key_pos,goal_pos] = min(Q[j,i,key,dir,door1,door2,key_pos,goal_pos])
                                        p_t[j,i,key,dir,door1,door2,key_pos,goal_pos] = np.argmin(Q[j,i,key,dir,door1,door2,key_pos,goal_pos])
        value_fn = value_fn_t
        if len(V)>2 and np.all(V[-1]==V[-2]):
            break
        V.append(value_fn)
        P.append(p_t)
        Q_func.append(Q)

    P = P[::-1]
    V = V[::-1]
    Q_func = Q_func[::-1]

    P = np.stack(P)
    V = np.stack(V)
    Q_func = np.stack(Q_func)

    return P,V,Q

def save_value_function_and_policy(P,V,Q_func):
    np.save("policy.npy", P)
    np.save("value_function.npy", V)
    np.save("Q-function.npy", Q_func)


def partB_value_iteration():
    grid = grid_definition()
    P,V,Q_func = value_iteration(grid)
    save_value_function_and_policy(P,V,Q_func)
    return P,V,Q_func


def doorkey_problemB(env, info, P):

    seq = []
    for t in range(P.shape[0]):
        cur_pos = env.agent_pos
        cur_dir = tuple(env.dir_vec)
        key = 1 if env.carrying is not None else 0

        if door_pos_map[tuple(info["door_pos"][0])[::-1]] == 0:
            door1 = 1 if env.grid.get(info["door_pos"][0][0],info["door_pos"][0][1]).is_open else 0
        else:
            door2 = 1 if env.grid.get(info["door_pos"][0][0],info["door_pos"][0][1]).is_open else 0
        if door_pos_map[tuple(info["door_pos"][1])[::-1]] == 0:
            door1 = 1 if env.grid.get(info["door_pos"][1][0],info["door_pos"][1][1]).is_open else 0
        else:
            door2 = 1 if env.grid.get(info["door_pos"][1][0],info["door_pos"][1][1]).is_open else 0

        key_pos = key_pos_map[tuple(info["key_pos"])[::-1]]
        goal_pos = goal_pos_map[tuple(info["goal_pos"])[::-1]]
        act = P[t,cur_pos[1], cur_pos[0], key, dir_map[cur_dir], door1, door2, key_pos, goal_pos]
        seq.append(act)
        done = step(env, act)
        if done:
            break

    seq.append(MF)
    return seq

def save_gif_and_frames(env_path, seq):
    env, info = load_env(env_path)
    env_name = env_path.rsplit('/')[-1].split('.')[0]
    pathFolder=f"gif/{env_name}/"
    os.makedirs(pathFolder,exist_ok=True)
    draw_gif_from_seq(seq, env, pathFolder=pathFolder)
    seq_str = ', '.join([inv_act_map[x] for x in seq[:-1]])

    with open(pathFolder + "sequence.txt", "w") as text_file:
        text_file.write(seq_str)
    

def partB_evaluate():

    try:
        P = np.load("policy.npy")
        V = np.load("value_function.npy")
        Q_func = np.load("Q-function.npy")
    except FileNotFoundError:
        P,V,Q_func = partB_value_iteration()

    # Run for all 36 random environments
    for num in range(1,37):
        env, info, env_path = load_random_env(num)
        seq = doorkey_problemB(env, info, P)
        save_gif_and_frames(env_path, seq)

if __name__ == "__main__":
    # partB_value_iteration()
    partB_evaluate()

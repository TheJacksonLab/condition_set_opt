import numpy as np

def convert_to_onehot(shape: tuple, point:list)-> list:
    '''
    @params
    point: in order of condition, reactant
    '''
    onehot = np.zeros(np.sum(shape))
    for i, p in enumerate(point):
        onehot[int(p + np.sum(shape[:i]))] = 1
    return onehot

def convert_from_onehot(shape: tuple, onehot: int)-> list:
    '''
    returns
    point: in order of condition, reactant
    '''
    point = []
    num = 0
    shape_counter = 0
    for i in range(len(onehot)):
        if onehot[i] == 1:
            point.append(num)
        num += 1
        if num >= shape[shape_counter]:
            num = 0
            shape_counter += 1
    return point

def convert_point_to_idx(shape:tuple, point:list)-> int:
    '''
    @params
    point: in order of condition, reactant
    '''
    idx = 0
    for i, n in enumerate(point):
        idx += n
        if i < len(shape) - 1:
            idx *= shape[i+1]
    return idx

def convert_idx_to_point(shape:tuple, idx:int)-> list:
    '''
    returns:
    point: in order of condition, reactant
    '''
    point = [0]*len(shape)
    for i in range(len(shape)-1, -1, -1):
        point[i] = (idx % shape[i])
        idx = idx // shape[i]
    return point
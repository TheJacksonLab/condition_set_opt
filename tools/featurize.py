import numpy as np

def convert_to_onehot(shape: tuple, point:list)-> list:
    '''
    @params
    shape: tuple of the shape of the condition and reactant space
    point: in order of condition, reactant
    @returns
    onehot: onehot encoding of the point
    '''
    onehot = np.zeros(np.sum(shape))
    for i, p in enumerate(point):
        onehot[int(p + np.sum(shape[:i]))] = 1
    return onehot

def convert_from_onehot(shape: tuple, onehot: int)-> list:
    '''
    @params
    shape: tuple of the shape of the condition and reactant space
    onehot: onehot encoding of the point in the same order as the shape
    @returns
    point: in the same order as shape
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
    shape: tuple of the shape of the condition and reactant space
    point: in the same order as shape
    @returns
    idx: index of the point in the flattened space
    '''
    idx = 0
    for i, n in enumerate(point):
        idx += n
        if i < len(shape) - 1:
            idx *= shape[i+1]
    return idx

def convert_idx_to_point(shape:tuple, idx:int)-> list:
    '''
    @params:
    shape: tuple of the shape of the condition and reactant space
    idx: index of the point in the flattened space
    @returns:
    point: in the same order as shape
    '''
    point = [0]*len(shape)
    for i in range(len(shape)-1, -1, -1):
        point[i] = (idx % shape[i])
        idx = idx // shape[i]
    return point
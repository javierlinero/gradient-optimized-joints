import numpy as np
import random

def rand_params(vals):
    randomized = []
    for value in vals:
        dist = np.random.normal(loc=value, scale=0.50, size=1000)
        dist = dist.tolist()
        samples = random.sample(dist, 30)
        avg = np.average(samples)
        randomized.append(avg)
    return randomized


def generate_params(params):
    for key in params:
        print(f"Key: {key}")
        first_params = rand_params(params[key])
        second_params = rand_params(params[key])        
        print(first_params)
        print(second_params)

if __name__ == '__main__':
    init_param = {
        'Gooseneck': [6.25, 5., 6.5, 3.5, 16, 7.5],
        'Scarf': [1.5, 5., 14.5, 14.5, 13.5, 10.5],
        'Lap': [15., 5., 5., 10., 20., 10.],
        'Dovetail_Scarf': [2.5, 7.5, 5., 15., 14.5, 11.5, 12.5, 14.5],
        'Rabbet': [20., 10., 10., 10., 15., 10., 25., 20.],
    }

    generate_params(init_param)
    print(init_param)
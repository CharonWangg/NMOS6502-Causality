import numpy as np
from tqdm import tqdm


def make_effect_label(root_dir):
    orig = np.load(root_dir + "original_3510_512.npy", mmap_mode='r')
    high = np.load(root_dir + "high_3510_512.npy", mmap_mode='r')
    low = np.load(root_dir + "low_3510_512.npy", mmap_mode='r')

    # calculate the effect of causal effect when turn high and turn low
    causal_effect = []
    for i in tqdm(range(orig.shape[0])):
        if (orig[i] == 1).all():
            # all 1, so turn low
            causal_effect.append(low[i] - orig)
        elif (orig[i] == 0).all():
            # all 0, so turn high
            causal_effect.append(high[i] - orig)
        else:
            causal_effect.append(high[i] - orig)
    causal_effect = np.stack(causal_effect)
    causal_effect = np.where(causal_effect == 255, -1, causal_effect)

    # convert causal effect to causality label based on only half clock difference from divergence point
    causal_1c_label = []
    for i in tqdm(range(causal_effect.shape[0])):
        temp = []
        if (orig[i] == 1).all():
            # become low at start
            t_div = 0
        elif (orig[i] == 0).all():
            # become high at start
            t_div = 0
        else:
            t_div = np.where(orig[i] != 1)[0][0]
        for j in range(causal_effect.shape[1]):
            if i == j:
                temp.append(0)
            else:
                if causal_effect[i, j, t_div] != 0:
                    temp.append(1)
                else:
                    temp.append(0)
        causal_1c_label.append(temp)
    causal_1c_label = np.stack(causal_1c_label)

    return causal_effect, causal_1c_label


if __name__ == "__main__":
    # original data
    game = "SpaceInvaders"
    root_dir = "./{}/".format(game)
    causal_effect, causal_1c_label = make_effect_label(root_dir)
    np.save(".../nmos_inference/envs/{}/causal_effect_512.npy".format(game), causal_effect)
    np.save(".../nmos_inference/envs/{}/causal_1c_label.npy".format(game), causal_1c_label)
    print("Done!")

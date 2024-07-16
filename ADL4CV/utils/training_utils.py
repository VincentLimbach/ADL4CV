import random
import torch


def generate_pairs(num_images, val_size, number_of_occurances, seed=None):
    left_images = [i for i in range(num_images) for _ in range(number_of_occurances)]
    right_images = [i for i in range(num_images) for _ in range(number_of_occurances)]

    if seed is not None:
        random.seed(seed)

    random.shuffle(left_images)
    random.shuffle(right_images)

    all_pairs = []

    for left, right in zip(left_images, right_images):
        all_pairs.append((left, right))

    random.shuffle(all_pairs)

    val_pairs = all_pairs[:val_size]
    train_pairs = all_pairs[val_size:]

    return train_pairs, val_pairs


def dict2cuda(a_dict, device):
    for _, value in a_dict.items():
        if isinstance(value, torch.Tensor):
            tmp = value.to(device)
        elif isinstance(value, dict):
            tmp = dict2cuda(value, device)
        elif isinstance(value, list) or isinstance(value, tuple):
            if isinstance(value[0], torch.Tensor):
                tmp = [v.to(device) for v in value]
        else:
            tmp = value
    return tmp

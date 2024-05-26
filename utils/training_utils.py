import random

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

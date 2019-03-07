import numpy as np
from string import ascii_uppercase
from time import asctime
from PIL import Image
from perceptron_network import Perceptron


def collect_dataset(start, end):
    print(asctime(), ' - Collecting dataset...')
    dataset = {}
    for let in ascii_uppercase:
        pix = []
        for i in range(start, end):
            pic = Image.open(f'selection/{let}-{i}')
            w, h = pic.size
            load = pic.load()
            for x in range(w):
                for y in range(h):
                    pix.append(load[(x, y)])


if __name__ == '__main__':
    learn_dataset = collect_dataset(0, 1000)
    test_dataset = collect_dataset(950, 1050)

    neural_network = Perceptron(layers=[121, 250, 100, 26], speed=0.3)




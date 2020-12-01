#!/usr/bin/env python

import click

from functools import reduce
from itertools import combinations

class Expense(object):

    def __init__(self, filename: str):
        with open(filename, 'r') as fh:
            self.input = fh.readlines()

    def multiply_summands(self, target: int, num: int)->int:
        combos = combinations(self.input, num)
        for combo in combos:
            summands = [int(combo[i]) for i in range(len(combo))]
            if sum(summands) == target:
                return reduce(lambda a,b: a*b, summands, 1)


def day_1(fname: str):
    ex = Expense(fname)
    print('Part 1: {}'.format(ex.multiply_summands(2020, 2)))
    print('Part 2: {}'.format(ex.multiply_summands(2020, 3)))


@click.command()
@click.argument('day')
@click.argument('fname', type=click.Path())
def main(day, fname):

    commands = {
        '1': lambda x: day_1(fname)
    }

    commands.get(day, lambda x: print(f'No day {x}'))(fname)


if __name__ == '__main__':
    main()

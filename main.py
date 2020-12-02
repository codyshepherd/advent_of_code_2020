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


class Passwords(object):

    def __init__(self, filename: str):
        with open(filename, 'r') as fh:
            self.input = fh.readlines()

        self.db = {}
        key = 0
        for line in self.input:
            splitline = line.split(': ')
            pwd = splitline[1]
            policy = splitline[0].split(' ')
            limits = policy[0]
            letter = policy[1]
            low = int(limits.split('-')[0])
            high = int(limits.split('-')[1])
            letter_count = pwd.count(letter)
            self.db[key] = {
                'low': low,
                'high': high,
                'letter': letter,
                'letter_count': letter_count,
                'pwd': pwd,
            }
            key += 1

    def count_valid_pwds(self)->int:
        count = 0
        for entry in self.db.values():
            if entry['letter_count'] <= entry['high'] and \
               entry['letter_count'] >= entry['low']:
                count += 1

        return count

    def count_valid_pwds_new(self)->int:
        count = 0
        for entry in self.db.values():
            pwd = entry['pwd']
            letter = entry['letter']
            low = entry['low']
            high = entry['high']
            low_valid = pwd[low-1] == letter
            high_valid = pwd[high-1] == letter
            if low_valid ^ high_valid:
                count += 1
        return count


def day_1(fname: str):
    ex = Expense(fname)
    print('Part 1: {}'.format(ex.multiply_summands(2020, 2)))
    print('Part 2: {}'.format(ex.multiply_summands(2020, 3)))


def day_2(fname: str):
    print('Day 2\n==========')
    pw = Passwords(fname)
    print('Part 1: {}'.format(pw.count_valid_pwds()))
    print('Part 2: {}'.format(pw.count_valid_pwds_new()))


@click.command()
@click.argument('day')
@click.argument('fname', type=click.Path())
def main(day, fname):

    commands = {
        '1': lambda x: day_1(fname),
        '2': lambda x: day_2(fname),
    }

    commands.get(day, lambda x: print(f'No day {x}'))(fname)


if __name__ == '__main__':
    main()

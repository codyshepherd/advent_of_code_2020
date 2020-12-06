#!/usr/bin/env python

import re

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


class SledMap(object):

    def __init__(self, filename: str):
        with open(filename, 'r') as fh:
            self.input = fh.readlines()

    def prepare_map(self, right:int, down:int):
        total_required_length = (right*down*len(self.input))+1
        newmap = []
        for line in self.input:
            newline = ''
            while len(newline) < total_required_length:
                newline += line.replace('\n', '')
            newmap.append(newline)
        return newmap

    def count_trees_on_slope(self, right: int, down: int)->int:
        slope_map = self.prepare_map(right, down)
        count = 0

        for h, line in enumerate(slope_map):
            index = (right*down*h) if right > down else (right*(h//down))
            newstring = line[:index] + '[' + line[index] + ']' + line[index+1:]
            if h != 0 and h%down == 0 and line[index] == '#':
                count += 1
        return count

    def count_several_slopes(self, slope_list: (int, int))-> list:
        trees = []
        for (right, down) in slope_list:
            trees.append(self.count_trees_on_slope(right, down))
        return trees


class Passports(object):

    eye_colors = [
        'amb',
        'blu',
        'brn',
        'gry',
        'grn',
        'hzl',
        'oth',
    ]

    validation_ops = {
        'byr': lambda x: len(x) == 4 and 1920 <= int(x) <= 2002,
        'iyr': lambda x: len(x) == 4 and 2010 <= int(x) <= 2020,
        'eyr': lambda x: len(x) == 4 and 2020 <= int(x) <= 2030,
        'hgt': lambda x: (150 <= Passports.try_cast(x[:-2]) <= 193 and re.match(r'^[0-9]+cm$', x) is not None) \
                or (59 <= Passports.try_cast(x[:-2]) <= 76 and re.match(r'^[0-9]+in$', x) is not None),
        'hcl': lambda x: re.match(r'^#[0-9a-f]{6}$', x) is not None,
        'ecl': lambda x: x in Passports.eye_colors,
        'pid': lambda x: re.match(r'^[0-9]{9}$', x) is not None,
        'cid': lambda x: True,
    }

    def __init__(self, filename: str):
        with open(filename, 'r') as fh:
            self.input = fh.read()

        raw_docs = self.input.split('\n\n')

        self.docs = []

        for doc in raw_docs:
            parsed = {}
            entries = doc.split()
            for entry in entries:
                pair = entry.split(':')
                k = pair[0]
                v = pair[1]
                parsed[k] = v
            self.docs.append(parsed)

    @staticmethod
    def try_cast(val: str)->int:
        try:
            casted = int(val)
        except ValueError:
            casted = -1
        return casted

    def is_data_valid(self, doc: dict)->bool:
        return all([Passports.validation_ops[k](v) for k, v in doc.items()])

    def is_doc_valid(self, doc: dict)->bool:
        keys = doc.keys()
        return all([k in keys or k == 'cid' for k in Passports.validation_ops.keys()])

    def count_valid_docs(self)-> int:
        count = 0
        for doc in self.docs:
            if self.is_doc_valid(doc):
                count += 1
        return count

    def count_valid_docs_improved(self)->int:
        count = 0
        for doc in self.docs:
            if self.is_doc_valid(doc) and self.is_data_valid(doc):
                count += 1
        return count


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


class BoardingPasses(object):

    def __init__(self, filename: str):
        with open(filename, 'r') as fh:
            self.input = fh.readlines()

        self.converted = []
        for line in self.input:
            part1 = line[:7]
            part2 = line[7:]

            b_string1 = part1.replace('B', '1')
            b_string1 = b_string1.replace('F', '0')

            b_string2 = part2.replace('R', '1')
            b_string2 = b_string2.replace('L', '0')

            self.converted.append((int(b_string1, 2), int(b_string2, 2)))

    def highest_seat(self)->int:
        max_found = 0
        for row, seat in self.converted:
            result = (row*8) + seat
            if result > max_found:
                max_found = result
        return max_found

    def find_seat(self)->int:
        all_results = []
        for row, seat in self.converted:
            result = (row*8) + seat
            all_results.append(result)

        for i in range(128*8):
            if i-1 in all_results and i+1 in all_results and i not in all_results:
                return i


def day_1(fname: str):
    ex = Expense(fname)
    print('Part 1: {}'.format(ex.multiply_summands(2020, 2)))
    print('Part 2: {}'.format(ex.multiply_summands(2020, 3)))


def day_2(fname: str):
    print('Day 2\n==========')
    pw = Passwords(fname)
    print('Part 1: {}'.format(pw.count_valid_pwds()))
    print('Part 2: {}'.format(pw.count_valid_pwds_new()))


def day_3(fname: str):
    print('Day 3\n==========')
    m = SledMap(fname)
    print('Part 1: {}'.format(m.count_trees_on_slope(3, 1)))
    trees = m.count_several_slopes([(1, 1), (3, 1), (5, 1), (7, 1), (1, 2)])
    print(trees)
    prod = reduce(lambda a,b: a*b, trees, 1)
    print('Part 2: {}'.format(prod))


def day_4(fname: str):
    print('Day 4\n==========')
    p = Passports(fname)
    print('Part 1: {}'.format(p.count_valid_docs()))
    print('Part 2: {}'.format(p.count_valid_docs_improved()))


def day_5(fname: str):
    print('Day 5\n==========')
    b = BoardingPasses(fname)
    print('Part 1: {}'.format(b.highest_seat()))
    print('Part 2: {}'.format(b.find_seat()))


@click.command()
@click.argument('day')
@click.argument('fname', type=click.Path())
def main(day, fname):

    commands = {
        '1': lambda x: day_1(fname),
        '2': lambda x: day_2(fname),
        '3': lambda x: day_3(fname),
        '4': lambda x: day_4(fname),
        '5': lambda x: day_5(fname),
    }

    commands.get(day, lambda x: print(f'No day {x}'))(fname)


if __name__ == '__main__':
    main()

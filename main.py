#!/usr/bin/env python

import re

import click

from functools import reduce
from itertools import combinations
from queue import Queue


class Expense(object):

    def __init__(self, filename: str):
        with open(filename, 'r') as fh:
            self.input = fh.readlines()

    def multiply_summands(self, target: int, num: int) -> int:
        combos = combinations(self.input, num)
        for combo in combos:
            summands = [int(combo[i]) for i in range(len(combo))]
            if sum(summands) == target:
                return reduce(lambda a, b: a*b, summands, 1)


class SledMap(object):

    def __init__(self, filename: str):
        with open(filename, 'r') as fh:
            self.input = fh.readlines()

    def prepare_map(self, right: int, down: int):
        total_required_length = (right*down*len(self.input))+1
        newmap = []
        for line in self.input:
            newline = ''
            while len(newline) < total_required_length:
                newline += line.replace('\n', '')
            newmap.append(newline)
        return newmap

    def count_trees_on_slope(self, right: int, down: int) -> int:
        slope_map = self.prepare_map(right, down)
        count = 0

        for h, line in enumerate(slope_map):
            index = (right*down*h) if right > down else (right*(h//down))
            if h != 0 and h % down == 0 and line[index] == '#':
                count += 1
        return count

    def count_several_slopes(self, slope_list: (int, int)) -> list:
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
        'hgt': lambda x: (150 <= Passports.try_cast(x[:-2]) <= 193 and re.match(r'^[0-9]+cm$', x) is not None)
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
    def try_cast(val: str) -> int:
        try:
            casted = int(val)
        except ValueError:
            casted = -1
        return casted

    def is_data_valid(self, doc: dict) -> bool:
        return all([Passports.validation_ops[k](v) for k, v in doc.items()])

    def is_doc_valid(self, doc: dict) -> bool:
        keys = doc.keys()
        return all([k in keys or k == 'cid' for k in
                    Passports.validation_ops.keys()])

    def count_valid_docs(self) -> int:
        count = 0
        for doc in self.docs:
            if self.is_doc_valid(doc):
                count += 1
        return count

    def count_valid_docs_improved(self) -> int:
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

    def count_valid_pwds(self) -> int:
        count = 0
        for entry in self.db.values():
            if entry['letter_count'] <= entry['high'] and \
               entry['letter_count'] >= entry['low']:
                count += 1

        return count

    def count_valid_pwds_new(self) -> int:
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

    def highest_seat(self) -> int:
        max_found = 0
        for row, seat in self.converted:
            result = (row*8) + seat
            if result > max_found:
                max_found = result
        return max_found

    def find_seat(self) -> int:
        all_results = []
        for row, seat in self.converted:
            result = (row*8) + seat
            all_results.append(result)

        for i in range(128*8):
            if i-1 in all_results and i+1 in all_results and i not in \
              all_results:
                return i


class Customs(object):

    def __init__(self, filename: str):
        with open(filename, 'r') as fh:
            self.input = fh.read()

        self.groups = self.input.split('\n\n')

    def sum_group_counts(self) -> int:
        total = 0
        for group in self.groups:
            lines = group.split()
            total += len(set(reduce(lambda a, b: set(a).union(set(b)), lines)))
        return total

    def sum_group_all_counts(self) -> int:
        total = 0
        for group in self.groups:
            lines = group.split()
            total += len(set(reduce(lambda a, b: set(a).intersection(set(b)),
                                    lines)))
        return total


class Luggage(object):

    leaf_value = 'no other'

    def __init__(self, filename: str):
        with open(filename, 'r') as fh:
            self.input = fh.readlines()

        self.dead_ends = []
        self.parents_of_gold = []
        self.digraph = {}
        self.contents = {}
        for line in self.input:
            line = line.strip('.\n')
            line = line.replace(' bags', '')
            line = line.replace(' bag', '')
            splitline = line.split(' contain ')
            key = splitline[0]
            values = splitline[1]
            values_list = values.split(', ')
            edges = []
            for value in values_list:
                if value == Luggage.leaf_value:
                    number = 0
                    edge = value
                else:
                    number = value[0]
                    edge = value[2:]
                edges.append({"number": int(number), "edge": edge})
            self.digraph[key] = edges

    def count_shiny_gold_contents(self) -> int:
        count = 0
        gold_edges = self.digraph['shiny gold']

        for edge in gold_edges:
            count += edge['number']
            count += self.count_contents(edge['edge'])*edge['number']

        return count

    def count_contents(self, key: str) -> int:
        if key in self.contents.keys():
            return self.contents[key]

        count = 0
        if key == Luggage.leaf_value:
            return 0

        edges = self.digraph[key]
        for entry in edges:
            edge = entry['edge']
            num = entry['number']
            count += num
            count += self.count_contents(edge)*num
        self.contents[key] = count
        return count

    def count_shiny_gold_containers(self) -> int:
        count = 0

        for k in self.digraph.keys():
            if self.does_key_lead_to_gold(k):
                count += 1

        return count

    def does_key_lead_to_gold(self, key: str) -> bool:
        q = Queue()
        edges = self.digraph.get(key, [])
        for e in edges:
            q.put(e)

        while not q.empty():
            entry = q.get()
            e = entry['edge']
            if e in self.parents_of_gold or e == 'shiny gold':
                return True
            elif e in self.dead_ends:
                continue
            elif self.does_key_lead_to_gold(e):
                self.parents_of_gold.append(e)
                return True
            else:
                self.dead_ends.append(e)
                continue
        return False


class Assembler(object):

    def __init__(self, filename: str):
        with open(filename, 'r') as fh:
            self.input = fh.read().split('\n')

        self.code = list(map(
            lambda a, b: (a, b), range(len(self.input)), self.input))

        self.accum = 0
        self.pc = 0

        self.executed = {}

    def reset(self):
        self.accum = 0
        self.pc = 0
        self.executed = {}

    def acc(self, x: str) -> int:
        self.accum += int(x)
        self.pc += 1
        return self.accum

    def jmp(self, x: str) -> int:
        self.pc += int(x)
        return self.accum

    def nop(self, x: str) -> int:
        self.pc += 1
        return self.accum

    def exec_next(self) -> int:
        l_num, instruction = self.code[self.pc]
        if self.executed.get(l_num, False):
            raise Exception(
                f'acc value prior to first repeated inst: {self.accum}')
        split_inst = instruction.split()
        operation = split_inst[0]
        arg = split_inst[1]

        fn = getattr(self, operation)
        to_return = fn(arg)
        self.executed[l_num] = True
        return to_return

    def exec_next_heuristic(self, limit: int) -> int:
        if self.pc >= len(self.code):
            raise Exception(f'End of instructions! Value of acc: {self.accum}')
        l_num, instruction = self.code[self.pc]
        times_executed = self.executed.get(l_num, 0)
        if times_executed > limit:
            raise Exception(
                f'acc value prior to first repeated inst: {self.accum}')
        split_inst = instruction.split()
        operation = split_inst[0]
        arg = split_inst[1]

        fn = getattr(self, operation)
        to_return = fn(arg)
        self.executed[l_num] = times_executed + 1
        return to_return

    def accumulator_prior_to_loop(self) -> int:
        while True:
            try:
                self.exec_next()
            except Exception as e:
                return int(str(e).split()[-1])

    def find_terminating_program(self) -> int:
        nops = [x for x in self.code if 'nop' in x[1]]
        jmps = [x for x in self.code if 'jmp' in x[1]]

        for nop in nops:
            self.reset()
            nop_ind = nop[0]
            nop_inst = nop[1]
            self.code[nop_ind] = (nop_ind, nop_inst.replace('nop', 'jmp'))
            loop = False
            try:
                while self.pc < len(self.code):
                    return_value = self.exec_next_heuristic(1000)
            except Exception as e:
                if 'acc value' not in str(e):
                    raise e
                else:
                    loop = True
            if not loop:
                return return_value
            else:
                self.code[nop_ind] = nop
                print(f'Failed: {nop}, accum: {self.accum}')

        for nop in jmps:
            self.reset()
            nop_ind = nop[0]
            nop_inst = nop[1]
            self.code[nop_ind] = (nop_ind, nop_inst.replace('jmp', 'nop'))
            loop = False
            try:
                while True:
                    return_value = self.exec_next_heuristic(1000)
            except Exception as e:
                if 'acc value' not in str(e):
                    raise e
                else:
                    loop = True
            if not loop:
                return return_value
            else:
                self.code[nop_ind] = nop
                print(f'Failed: {nop}, accum: {self.accum}')

        raise Exception('None found!')


class Xmas(object):

    def __init__(self, filename: str):
        with open(filename, 'r') as fh:
            self.input = fh.read().split('\n')

        self.values = []
        for s in self.input:
            self.values.append(int(s))

    def is_sum(self, val: int, possible: list) -> bool:
        combos = combinations(possible, 2)
        sums = [x + y for (x, y) in combos]
        return val in sums

    def not_a_sum(self) -> int:
        for i, val in enumerate(self.values):
            if i < 25:
                continue
            if not self.is_sum(int(val), self.values[i-25:i]):
                return val

        raise Exception("None found!")

    def find_summands(self, val: int) -> tuple:
        window_sizes = range(2, len(self.values))
        for window in window_sizes:
            for j in range(len(self.values)):
                if j + window > len(self.values):
                    break
                window_values = self.values[j:j+window]
                if sum(window_values) == val:
                    return min(window_values) + max(window_values)
        raise Exception("No windows found!")


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
    prod = reduce(lambda a, b: a*b, trees, 1)
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


def day_6(fname: str):
    print('Day 6\n==========')
    c = Customs(fname)
    print('Part 1: {}'.format(c.sum_group_counts()))
    print('Part 2: {}'.format(c.sum_group_all_counts()))


def day_7(fname: str):
    print('Day 7\n==========')
    l = Luggage(fname)
    print('Part 1: {}'.format(l.count_shiny_gold_containers()))
    print('Part 2: {}'.format(l.count_shiny_gold_contents()))


def day_8(fname: str):
    print('Day 8\n==========')
    a = Assembler(fname)
    print('Part 1: {}'.format(a.accumulator_prior_to_loop()))
    b = Assembler(fname)
    print('Part 2: {}'.format(b.find_terminating_program()))


def day_9(fname: str):
    print('Day 9\n==========')
    x = Xmas(fname)
    invalid_sum = x.not_a_sum()
    print(f'Part 1: {invalid_sum}')
    print('Part 2: {}'.format(x.find_summands(invalid_sum)))


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
        '6': lambda x: day_6(fname),
        '7': lambda x: day_7(fname),
        '8': lambda x: day_8(fname),
        '9': lambda x: day_9(fname),
    }

    commands.get(day, lambda x: print(f'No day {x}'))(fname)


if __name__ == '__main__':
    main()

import sys
import os
import json
import math
from csv import DictReader
from fairness_data import FairnessProblem
import solver as solve


def process_line(filters, headers_integer_values, line):
    for f in filters:
        if f.get('values', None):
            if line[f['header']] in f['values']:
                return
        if f.get('range', None):
            lt = float(f['range'].get('lt', -math.inf))
            gt = float(f['range'].get('gt', math.inf))
            if f['range']['type'].lower() == 'and':
                if float(line[f['header']]) < lt and float(line[f['header']]) > gt:
                    return
            else:
                if float(line[f['header']]) < lt or float(line[f['header']]) > gt:
                    return

    line_copy = line.copy()
    for h in headers_integer_values:
        if line[h['header']] not in h.keys():
            return
        line_copy[h['header']] = h[line[h['header']]]

    return line_copy


def load_problem(options):
    data_file = open(os.path.dirname(__file__) + '/datasets/' + options['data_set'] + '/' + options['file'])
    headers = options['data_headers'].split(',')
    protected = options['protected']
    tag = options['tag']
    fp = bool(options['fp'])
    fn = bool(options['fn'])
    weight_gt = float(options['weight']['gt'])
    weight_lt = float(options['weight']['lt'])
    gamma_gt = float(options['gamma']['gt'])
    gamma_lt = float(options['gamma']['lt'])
    weight_res = int(options['weight_res'])
    gamma_res = int(options['gamma_res'])
    filters = options['filters']

    headers_integer_values = options['headers_integer_values']

    file_reader = DictReader(data_file)

    protected_index = headers.index(protected)
    x = []
    y = []
    for line in file_reader:
        processed_line = process_line(filters, headers_integer_values, line)
        if processed_line:
            x.append([float(processed_line[h]) for h in headers])
            y.append(int(processed_line[tag]))

    return FairnessProblem(
        description=options['description'],
        x=x,
        y=y,
        protected_index=protected_index,
        gamma_gt=gamma_gt,
        gamma_lt=gamma_lt,
        weight_gt=weight_gt,
        weight_lt=weight_lt,
        fp=fp,
        fn=fn,
        weight_res=weight_res,
        gamma_res=gamma_res
    )


def main(options_file):
    options = json.load(open(options_file))
    problem = load_problem(options)
    solve.fairness(problem)


if __name__ == "__main__":
    main(sys.argv[1])

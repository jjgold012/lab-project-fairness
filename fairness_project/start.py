import sys
import os
import json
import math
from csv import DictReader
from fairness_data import *
import solver as solve


def process_line(filters, headers_integer_values, headers_operation, line):
    for f in filters:
        if f.get('values', None):
            if line[f['header']] in f['values']:
                return
        if f.get('range', None):
            lt = float(f['range'].get('lt', -float('inf')))
            gt = float(f['range'].get('gt', float('inf')))
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

    for o in headers_operation:
        if o.get('divide', None):
            line_copy[o['header']] = float(line[o['header']])/float(o['divide'])

    return line_copy


def load_problem_from_options(options_file):
    file = open(options_file).read()
    print('Setting up the problem according to this option file:')
    print(file)
    options = json.loads(file)

    data_file = open(os.path.dirname(__file__) + '/datasets/' + options['data_set'] + '/' + options['file'])
    headers = options['data_headers'].split(',')
    protected = options['protected']
    tag = options['tag']
    fp = bool(options['fp'])
    fn = bool(options['fn'])
    objective_weight = float(options['objective_weight'])
    weight_gt = float(options['weight']['gt'])
    weight_lt = float(options['weight']['lt'])
    gamma_gt = float(options['gamma']['gt'])
    gamma_lt = float(options['gamma']['lt'])
    weight_res = int(options['weight_res'])
    gamma_res = int(options['gamma_res'])
    test_size = float(options['test_size'])
    val_size = float(options['val_size'])
    num_of_folds = int(options['num_of_folds'])
    num_of_runs = int(options['num_of_runs'])

    filters = options['filters']

    headers_integer_values = options['headers_integer_values']
    headers_operation = options['headers_operation']
    file_reader = DictReader(data_file)

    protected_index = headers.index(protected)
    x = []
    y = []
    number_of_lines = 0
    positive_data = 0
    for line in file_reader:
        number_of_lines += 1
        processed_line = process_line(filters, headers_integer_values, headers_operation, line)
        if processed_line:
            line_data = [float(processed_line[h]) for h in headers]
            line_data.append(1.)
            x.append(line_data)
            line_tag = int(processed_line[tag])
            y.append(line_tag)
            if line_tag == 1:
                positive_data += 1

    print('\ndataset size: ' + str(number_of_lines) + '\ndataset used: ' + str(len(x)))
    print('positive examples: ' + str(positive_data) + '\nnegative examples: ' + str(len(x) - positive_data) + '\n')
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
        objective_weight=objective_weight,
        weight_res=weight_res,
        gamma_res=gamma_res,
        test_size=test_size,
        val_size=val_size,
        num_of_folds=num_of_folds,
        num_of_runs=num_of_runs,
        original_options=file
    )


def main(options):
    synthetic = False
    if options.endswith('json'):
        problem = load_problem_from_options(options)
    else:
        epsilon = float(options)
        problem = create_synthetic_problem(epsilon)
        synthetic = True

    solve.solve_problem_for_fairness(problem, synthetic)

if __name__ == "__main__":
    main(sys.argv[1])

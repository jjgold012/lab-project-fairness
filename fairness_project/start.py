import sys
import os
import json
from csv import DictReader
from fairness_project.fairness_data import FairnessProblem
import fairness_project.solver as solve


def process_line(filters, headers_integer_values, line):
    for f in filters:
        if line[f['header']] in f['values']:
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
    fp_weight = float(options['fp_weight'])
    fn_weight = float(options['fn_weight'])
    gamma_gt = float(options['gamma']['gt'])
    gamma_lt = float(options['gamma']['lt'])
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

    return FairnessProblem(options['description'], x, y, protected_index, gamma_gt, gamma_lt, fp_weight, fn_weight)


def main(options_file):
    options = json.load(open(options_file))
    problem = load_problem(options)
    solve.fairness(problem)


if __name__ == "__main__":
    main(sys.argv[1])

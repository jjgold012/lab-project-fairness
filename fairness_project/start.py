import sys
from csv import DictReader
import json
#from pprint import pprint
from fairness_project.problem import FairnessProblem


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

def main(option_file):
    option = json.load(open(option_file))
    data_file = open('./datasets/' + option['data_set'] + '/' + option['file'])
    headers = option['data_headers']
    protected = option['protected']
    tag = option['tag']
    filters = option['filters']
    headers_integer_values = option['headers_integer_values']

    file_reader = DictReader(data_file)

    protected_index = headers.index(protected)
    x = []
    y = []
    for line in file_reader:
        processed_line = process_line(filters, headers_integer_values, line)
        if processed_line:
            x.append([float(processed_line[h]) for h in headers])
            y.append(int(processed_line[tag]))

    problem = FairnessProblem(option['description'], x, y, protected_index)


if __name__ == "__main__":
    main(sys.argv[1])

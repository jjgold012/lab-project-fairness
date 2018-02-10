import os
import json
import subprocess
from timeit import default_timer


def run(description, argument):
    start_path = os.path.dirname(__file__) + 'fairness_project/start.py'

    dir_name = os.path.dirname(__file__) + 'results/' + description
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    out_file = dir_name + '/' + description + '.out'
    start = default_timer()
    subprocess.call(('python ' + start_path + ' ' + argument + ' > ' + out_file), shell=True)
    end = default_timer()
    print(description + ' finished. Results inside the directory: ' + dir_name)
    print(description + ' run for: ' + str(end - start))

print('Starting: synthetic_data_with_epsilon_0.1')
run('synthetic_data_with_epsilon_0.1', '0.1')

for file in os.listdir(os.path.dirname(__file__) + 'fairness_project/options'):
    if file.endswith('json'):
        file_path = os.path.dirname(__file__) + 'fairness_project/options/' + file
        options = json.load(open(file_path))
        print('Starting: ' + options['description'])
        run(options['description'], file_path)

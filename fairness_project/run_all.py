import os
import json
import subprocess

start_path = os.path.dirname(__file__) + '/start.py'

dir_name = os.path.dirname(__file__) + '/../results/all/synthetic_data_with_epsilon_0.1'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
out_file = dir_name + '/' + 'synthetic_data_with_epsilon_0.1.out'
subprocess.call(('python ' + start_path + ' 0.1' + ' > ' + out_file), shell=True)

for file in os.listdir(os.path.dirname(__file__) + '/options'):
    file_path = os.path.dirname(__file__) + '/options/' + file

    options = json.load(open(file_path))
    dir_name = os.path.dirname(__file__)  + '/../results/all/' + options['description']
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    out_file = dir_name + '/' + options['description'] + '.out'
    subprocess.call(('python ' + start_path + ' ' + file_path + ' > ' + out_file), shell=True)


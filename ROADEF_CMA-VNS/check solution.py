from function_checker import main_checker
from json_reader import read_json


penalty_koef = 300
instance_name = 'A_15'
f = open('current_best/' + instance_name + ".txt", 'r')
lines = []
solution = []
for line in f:
    lines.append(line.split())
    solution.append(int(lines[-1][1]))

instance = read_json(instance_name + '.json')

value_best, value_penalty, value_pen_tup = main_checker(instance, solution, penalty_koef)
print(value_best, value_penalty)

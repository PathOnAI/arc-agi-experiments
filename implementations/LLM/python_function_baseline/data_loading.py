from   matplotlib import colors
import json

base_path='../../data/arc-prize-2024/'
# Loading JSON data
def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

# Reading files
training_challenges =  load_json(base_path +'arc-agi_training_challenges.json')
training_solutions =   load_json(base_path +'arc-agi_training_solutions.json')

evaluation_challenges =load_json(base_path +'arc-agi_evaluation_challenges.json')
evaluation_solutions = load_json(base_path +'arc-agi_evaluation_solutions.json')

test_challenges =  load_json(base_path +'arc-agi_test_challenges.json')

print(f'Number of training challenges = {len(training_challenges)}')
print(f'Number of training solutions = {len(training_solutions)}')
print(f'Number of evaluation challenges = {len(evaluation_challenges)}')
print(f'Number of evaluation solutions = {len(evaluation_solutions)}')
print(f'Number of test challenges = {len(test_challenges)}')

# 0:black, 1:blue, 2:red, 3:greed, 4:yellow, # 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown

cmap = colors.ListedColormap(
    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)

t = '447fd412'
task=training_challenges[t]
task_solution = training_solutions[t][0]
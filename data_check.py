import os

def check_data_distribution(data_dir):
    class_dirs = ['Certificate', 'Scoreboard']
    for class_dir in class_dirs:
        path = os.path.join(data_dir, class_dir)
        print(f'{class_dir}: {len(os.listdir(path))} images')

current_dir = os.getcwd()
print("current dir", current_dir)

data_dir = os.path.join(current_dir, 'train')
print("data_dir", data_dir)

check_data_distribution(data_dir)

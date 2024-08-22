import os

def check_data_distribution(data_dir):
    class_dirs = ['Certificate', 'Scoreboard']
    for class_dir in class_dirs:
        path = os.path.join(data_dir, class_dir)
        print(f'{class_dir}: {len(os.listdir(path))} images')

data_dir = r'C:\Users\huy\Desktop\Cuong-Classification\train'
check_data_distribution(data_dir)

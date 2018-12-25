import os


def list_all(root, output_name):
    file_name = []
    file_name_r = []
    for item in os.listdir(root):
        if os.path.isdir(root + item):
            for name in os.listdir(root + item):
                if 'r' in name:
                    file_name_r.append(item + '/' + name)
                else:
                    file_name.append(item + '/' + name)
    assert(len(file_name) == len(file_name_r))
    with open(output_name, 'w') as f:
        for i in file_name:
            f.writelines(i+','+i.split('.')[0]+'_r.txt\n')


if __name__ == "__main__":
    list_all('./data/a/', 'list_a.txt')

import os


def rename(root):
    count = 0
    for i in os.listdir(root):
        count += 1
        print(count)
        # os.rename(root + i, root + i[:6] + i.split('-')[-1])
        os.rename(root + i, root + i[1:])


if __name__ == "__main__":
    rename('../data/a/')

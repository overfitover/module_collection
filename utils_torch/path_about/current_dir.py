import os

def current_path():
    """
    :return: current file's path and its parents' path
    """
    import os.path as osp
    this_dir = osp.dirname(__file__)
    parents_path = osp.split(this_dir)[0]
    # print(this_dir)
    # print(parents_path)

    return this_dir, parents_path


if __name__=="__main__":
    print(current_path())
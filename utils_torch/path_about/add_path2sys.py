import sys
def add_top_path2sys():
    """
    description: add this file's dir and its parents' dir to sys.path
    :return:
    """
    import os.path as osp
    import sys
    this_dir = osp.dirname(__file__)
    parents_path = osp.split(this_dir)[0]
    if this_dir not in sys.path:
        sys.path.insert(0, this_dir)
    if parents_path not in sys.path:
        sys.path.insert(0, parents_path)

if __name__ == '__main__':
    print(sys.path)
    # add_top_path2sys()
    print(sys.path)
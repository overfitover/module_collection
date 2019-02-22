import os
output_dir = './input/output'

def make_filedir(file_dir):
    import os
    """
    description: no dir make a dir , yes pass
    :param file_dir: file path
    :return: 
    """
    if os.path.exists(file_dir):
        print("{} exists".format(file_dir))

    else:
        os.makedirs(file_dir)

make_filedir(output_dir)
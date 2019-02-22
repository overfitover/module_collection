import os.path as osp

def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))
    else:
        print("check out")
        pass

if __name__=='__main__':
    check_file_exist('__init__.py')
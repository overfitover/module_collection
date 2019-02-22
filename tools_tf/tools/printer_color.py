
class Color:
    underline = '\033[4m'
    darkcyan  = '\033[36m'
    purple    = '\033[95m'
    yellow    = '\033[93m'
    green     = '\033[36m'
    cyan      = '\033[96m'
    blue      = '\033[94m'
    bold      = '\033[1m'
    red       = '\033[91m'
    end       = '\033[0m'

def underline(string):
    return Color.underline + string + Color.end

def darkcyan(string):
    return Color.darkcyan + string + Color.end

def purple(string):
    return Color.purple + string + Color.end

def yellow(string):
    return Color.yellow + string + Color.end

def green(string):
    return Color.green + string + Color.end

def cyan(string):
    return Color.cyan + string + Color.end

def blue(string):
    return Color.blue + string + Color.end

def bold(string):
    return Color.bold + string + Color.end

def red(string):
    return Color.red + string + Color.end


if __name__=="__main__":
    head='Table:Test of ApoxelNet'
    print(underline(head))
    print(darkcyan(head))
    print(purple(head))
    print(yellow(head))
    print(green(head))
    print(cyan(head))
    print(bold(head))
    print blue(head)
    print red(head)


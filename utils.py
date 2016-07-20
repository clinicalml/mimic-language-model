class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


def print_color(s, color=None):
    if color:
        print color + str(s) + Colors.ENDC,
    else:
        print s,


def inspection_decide_color(diff):
    if diff < -0.1:
        return Colors.FAIL
    elif diff > 0.2:
        return Colors.OKGREEN
    elif diff > 0.1:
        return Colors.WARNING
    else:
        return None

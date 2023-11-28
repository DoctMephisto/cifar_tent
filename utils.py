import torch

import os
import sys
import time

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    fmt_time = []
    if days > 0:
        fmt_time.append(f"{days}D")
    if hours > 0:
        fmt_time.append(f"{hours}h")
    if minutes > 0:
        fmt_time.append(f"{minutes}m")
    if secondsf > 0:
        fmt_time.append(f"{secondsf}s")
    if millis > 0:
        fmt_time.append(f"{millis}ms")

    str = ":".join(fmt_time)
    if str == "":
        str = '0ms'
    return str

last_time = time.time()
begin_time = last_time
TOTAL_BAR_LENGTH = 50
_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

def progress_bar(current, total, msg = None):
    global begin_time, last_time
    if current == 0:
        begin_time = time.time()

    cur_len = int(TOTAL_BAR_LENGTH*(current/total))
    rest_len = TOTAL_BAR_LENGTH - cur_len

    cur_time = time.time()
    step_time = cur_time - last_time
    tot_time = cur_time - begin_time
    last_time = cur_time

    sys.stdout.write('[')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    msg_arr = []
    msg_arr.append(f' Step: {format_time(step_time)}')
    msg_arr.append(f' | Tot: {format_time(tot_time)}')
    if msg:
        msg_arr.append(' | ' + msg)

    msg = ''.join(msg_arr)

    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def test():
    len = eval(input())
    for i in range(len):
        progress_bar(i, len+1)
        time.sleep(0.5)

if __name__ == '__main__':
    test()

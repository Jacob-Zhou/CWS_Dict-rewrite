import os
import codecs


def max_len(filename):
    count = 0
    max_l = 0
    with codecs.open(filename, 'r', 'utf-8') as f:
        for line in f:
            line_len = len("".join(line.split()))
            if line_len > max_l:
                max_l = line_len
            if line_len > 510:
                count += 1
    return max_l, count


if __name__ == "__main__":
    files = list(os.walk("./"))[0][2]
    for file in files:
        if file == "statistic.py":
            continue
        max_l, count = max_len(file)
        print("----------------------------------------------------------")
        print("%s" % file)
        print("max line length: %d" % max_l)
        print("longer than 512: %d" % count)
        print("----------------------------------------------------------")

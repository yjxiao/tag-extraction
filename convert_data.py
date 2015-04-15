import sys
import gc


def convert_raw_data(in_path, out_path):
    """Convert raw csv data file to clear newlines in entries"""
    f_src = open(in_path, 'rb')
    f_dest = open(out_path, 'wb')
    for i, line in enumerate(f_src.xreadlines()):
        if line[-2:] == '\r\n':
            f_dest.write(line[:-2] + '\n')
        else:
            f_dest.write(line[:-1] + ' ')
        if (i + 1) % 1000000 == 0:
            f_dest.flush()
            gc.collect()


def main(args):
    convert_raw_data(str(args[0]), str(args[1]))


if __name__ == '__main__':
    main(sys.argv[1:])

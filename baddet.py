import sys
import argparse
from setup import set_args

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    set_args(parser)
    args = parser.parse_args()
import sys
import argparse
from .masking import add_masking_parser
from .regrid import add_regrid_parser

def main():
    parser = argparse.ArgumentParser(prog="ptopo")
    subparsers = parser.add_subparsers(dest="command")

    add_regrid_parser(subparsers)
    add_masking_parser(subparsers)

    args = parser.parse_args()
    if args.command:
        args.cmdline = " ".join(sys.argv)
        args.func(args)
    else:
        parser.print_help()

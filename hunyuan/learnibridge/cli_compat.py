import inspect
import sys


def parse_native_arguments(parse_function, arguments):
    """Parse a native CLI at process startup while preserving the caller's argv."""
    parameters = inspect.signature(parse_function).parameters
    if "args" in parameters:
        return parse_function(args=list(arguments))
    previous = sys.argv
    try:
        sys.argv = [previous[0], *arguments]
        return parse_function()
    finally:
        sys.argv = previous

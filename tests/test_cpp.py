"""
Run the C++ tests.

A very simple, inelegant way to run the basic C++ tests.
We should probably invest some time into writing some proper
C++ unit tests.

"""
import subprocess
import os
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpp")


def test_cpp():
    """Run the C++ tests."""
    nerr = subprocess.check_call(["./test"], cwd=PATH)
    assert(nerr == 0)
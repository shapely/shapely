import os
import unittest

try:
    import pkg_resources
    pkg_resources.require('Shapely')
except:
    pass

suite = unittest.TestSuite()
load = unittest.defaultTestLoader.loadTestsFromModule

tests = os.listdir(os.curdir)
tests = [n[:-3] for n in tests if n.startswith('test') and n.endswith('.py')]

for test in tests:
    m = __import__(test)
    suite.addTest(load(m))

if __name__ == '__main__':
    import getopt
    import sys

    verbosity = 0
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'v')
        if opts[0][0] == '-v':
            verbosity = verbosity + 1
    except:
        pass

    runner = unittest.TextTestRunner(verbosity=verbosity)
    runner.run(suite)


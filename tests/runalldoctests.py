
try:
    import pkg_resources
    pkg_resources.require('Shapely')
except:
    pass

if __name__ == '__main__':
    import doctest
    import getopt
    import glob
    import sys
    
    verbosity = 0
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'v')
        if opts[0][0] == '-v':
            verbosity = verbosity + 1
    except:
        pass

    for file in [
        'Array.txt',
        'Operations.txt',
        'Persist.txt',
        'Predicates.txt'
        ]:
        doctest.testfile(file, verbose=verbosity)
    

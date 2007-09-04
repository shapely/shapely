
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
    pattern = None

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'vt:')
        for o, a in opts:
            if o == '-v':
                verbosity = verbosity + 1
            if o == '-t':
                pattern = a
    except:
        pass

    docfiles = [
        'Array.txt',
        'GeoInterface.txt',
        'IterOps.txt',
        'LineString.txt',
        'MultiLineString.txt',
        'MultiPoint.txt',
        'MultiPolygon.txt',
        'Operations.txt',
        'Persist.txt',
        'Point.txt',
        'Polygon.txt',
        'Predicates.txt'
        ]
        
    if pattern:
        tests = [f for f in docfiles if f.find(pattern) == 0]
    else:
        tests = docfiles
        
    for file in tests:
        doctest.testfile(file, verbose=verbosity)


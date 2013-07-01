import threading

def main():
    num_threads = 10
    use_threads = True
    
    if not use_threads:
        # Run core code
        runShapelyBuilding()
    else:
        threads = [threading.Thread(target=runShapelyBuilding, name=str(i), args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
def runShapelyBuilding(num):
    print "%s: Running shapely tests on wkb" % num
    import shapely.geos
    print "%s GEOS Handle: %s" % (num, shapely.geos.lgeos.geos_handle)
    import shapely.wkt
    import shapely.wkb
    p = shapely.wkt.loads("POINT (0 0)")
    print "%s WKT: %s" % (num, shapely.wkt.dumps(p))
    wkb = shapely.wkb.dumps(p)
    print "%s WKB: %s" % (num, wkb.encode('hex'))
    
    for i in xrange(10):
        obj = shapely.wkb.loads(wkb)
    
    print "%s GEOS Handle: %s" % (num, shapely.geos.lgeos.geos_handle)
    print "Done %s" % num
    
# Run main
if __name__ == '__main__':
   main()   


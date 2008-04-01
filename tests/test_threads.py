import threading

def main():
   num_threads = 4
   use_threads = True
   
   if not use_threads:
      # Run core code
      runShapelyBuilding()
   else:
      threads = [threading.Thread(target=runShapelyBuilding) for x in range(num_threads)]
      for t in threads:
         t.start()
      for t in threads:
         t.join()
         
def runShapelyBuilding():
   import shapely.wkt
   import shapely.wkb
   
   print "Running shapely tests on wkb"
   
   wkb = shapely.wkt.loads("POINT (0 0)").wkb
   
   for i in xrange(1000):
      obj = shapely.wkb.loads(wkb)
   
   print "Done"
   
# Run main
if __name__ == '__main__':
   main()   


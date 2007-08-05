
from point import PointAdapter

def asShape(context):
    try:
        if hasattr(context, "__geo_interface__"):
            ob = context.__geo_interface__
        else:
            ob = context
        if ob.get("type").lower() == "point":
            return PointAdapter(ob["coordinates"])
        else:
            raise NotImplementedError
    except:
        raise


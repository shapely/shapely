"""svgtree.py: ElementTree based geometry SVG"""

import itertools
from xml.etree.ElementTree import Element, ElementTree, SubElement

import numpy


class SVGTree(ElementTree):
    @classmethod
    def path(cls, coords):
        return "M {} L {}".format(
            f"{coords[0][0]},{coords[0][1]}",
            " L ".join(f"{xy[0]},{xy[1]}" for xy in coords[1:]),
        )

    @classmethod
    def fromgeom(cls, geom):
        """Returns Element"""
        if geom.geom_type == "Point":
            return cls.frompoint(geom)
        elif geom.geom_type == "LineString":
            return cls.fromlinestring(geom)
        elif geom.geom_type == "LinearRing":
            return cls.fromlinearring(geom)
        elif geom.geom_type == "Polygon":
            return cls.frompolygon(geom)
        else:
            raise NotImplementedError

    @classmethod
    def frompoint(cls, geom):
        group = Element("g")
        SubElement(group, "circle", cx=str(geom.x), cy=str(geom.y), r="1.0")
        return group

    @classmethod
    def fromlinestring(cls, geom):
        group = Element("g")        
        SubElement(group, "path", d=cls.path(geom.coords))
        return group

    @classmethod
    def fromlinearring(cls, geom):
        group = Element("g")        
        SubElement(group, "path", d="{} Z".format(cls.path(geom.coords[:-1])))
        return group

    @classmethod
    def frompolygon(cls, geom):
        group = Element("g")        
        path = "  ".join(
            "{} Z".format(cls.path(ring.coords[:-1]))
            for ring in itertools.chain([geom.exterior], geom.interiors)
        )
        SubElement(group, "path", d=path)
        return group
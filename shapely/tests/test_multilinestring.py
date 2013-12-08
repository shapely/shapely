from . import unittest, numpy
from shapely.geometry import LineString, MultiLineString, asMultiLineString
from shapely.geometry.base import dump_coords


class MultiLineStringTestCase(unittest.TestCase):

    def test_multipoint(self):

        # From coordinate tuples
        geom = MultiLineString((((1.0, 2.0), (3.0, 4.0)),))
        self.assertIsInstance(geom, MultiLineString)
        self.assertEqual(len(geom.geoms), 1)
        self.assertEqual(dump_coords(geom), [[(1.0, 2.0), (3.0, 4.0)]])

        # From lines
        a = LineString(((1.0, 2.0), (3.0, 4.0)))
        ml = MultiLineString([a])
        self.assertEqual(len(ml.geoms), 1)
        self.assertEqual(dump_coords(ml), [[(1.0, 2.0), (3.0, 4.0)]])

        # From another multi-line
        ml2 = MultiLineString(ml)
        self.assertEqual(len(ml2.geoms), 1)
        self.assertEqual(dump_coords(ml2), [[(1.0, 2.0), (3.0, 4.0)]])

        # Sub-geometry Access
        geom = MultiLineString([(((0.0, 0.0), (1.0, 2.0)))])
        self.assertIsInstance(geom[0], LineString)
        self.assertEqual(dump_coords(geom[0]), [(0.0, 0.0), (1.0, 2.0)])
        with self.assertRaises(IndexError):  # index out of range
            geom.geoms[1]

        # Geo interface
        self.assertEqual(geom.__geo_interface__,
                         {'type': 'MultiLineString',
                          'coordinates': (((0.0, 0.0), (1.0, 2.0)),)})

    @unittest.skipIf(not numpy, 'Numpy required')
    def test_numpy(self):

        from numpy import array
        from numpy.testing import assert_array_equal

        # Construct from a numpy array
        geom = MultiLineString([array(((0.0, 0.0), (1.0, 2.0)))])
        self.assertIsInstance(geom, MultiLineString)
        self.assertEqual(len(geom.geoms), 1)
        self.assertEqual(dump_coords(geom), [[(0.0, 0.0), (1.0, 2.0)]])

        # Adapt a sequence of Numpy arrays to a multilinestring
        a = [array(((1.0, 2.0), (3.0, 4.0)))]
        geoma = asMultiLineString(a)
        assert_array_equal(geoma.context, [array([[1., 2.], [3., 4.]])])
        self.assertEqual(dump_coords(geoma), [[(1.0, 2.0), (3.0, 4.0)]])

        # TODO: is there an inverse?


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(MultiLineStringTestCase)

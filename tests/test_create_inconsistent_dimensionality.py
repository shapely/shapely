"""
When a "context" passed to shape/asShape has a coordinate
which is missing a dimension we should raise a descriptive error.

When we use mixed dimensions in a WKT geometry, the parser strips
any dimension which is not present in every coordinate.
"""


import pytest

from shapely import wkt
from shapely.geometry import shape, LineString, Polygon

from tests.conftest import shapely20_todo


geojson_cases = [
    {"type": "LineString", "coordinates": [[1, 1, 1], [2, 2]]},
    # Specific test case from #869
    {"type": "Polygon", "coordinates": [[[55.12916764533149, 24.980385694214384, 2.5], [55.13098248044217, 24.979828079961905], [55.13966519231666, 24.97801442415322], [55.13966563924936, 24.97801442415322], [55.14139286840762, 24.982307444496097], [55.14169331277646, 24.983717465495562], [55.14203489144224, 24.985419446276566, 2.5], [55.14180327151276, 24.98428602667792, 2.5], [55.14170091915952, 24.984242720177235, 2.5], [55.14122966992623, 24.984954809433702, 2.5], [55.14134021791831, 24.985473928648396, 2.5], [55.141405876161286, 24.986090184809793, 2.5], [55.141361358941225, 24.986138101357326, 2.5], [55.14093322994411, 24.986218753894093, 2.5], [55.140897653420964, 24.986214283545635, 2.5], [55.14095492976058, 24.9863027591922, 2.5], [55.140900447388745, 24.98628436557094, 2.5], [55.140867059473706, 24.98628869622101, 2.5], [55.14089155325796, 24.986402364143782, 2.5], [55.14090938808566, 24.986479011993385, 2.5], [55.140943893587824, 24.986471188883584, 2.5], [55.1410161176551, 24.9864174050037, 2.5], [55.140996932409635, 24.986521806266644, 2.5], [55.14163554031332, 24.986910400619593, 2.5], [55.14095781686062, 24.987033474900578, 2.5], [55.14058258698692, 24.98693261266349, 2.5], [55.14032624044253, 24.98747538747211, 2.5], [55.14007240846915, 24.988001119077232, 2.5], [55.14013122149105, 24.98831115636925, 2.5], [55.13991827457961, 24.98834356639557, 2.5], [55.139779460946755, 24.988254625087706, 2.5], [55.13974742344948, 24.988261377176524, 2.5], [55.139515198160304, 24.98841811876934, 2.5], [55.13903617238334, 24.98817914139135, 2.5], [55.1391330764994, 24.988660542040925, 2.5], [55.13914369357698, 24.989438289540374, 2.5], [55.136431216517785, 24.98966711550207, 2.0], [55.13659028641709, 24.99041706302204, 2.0], [55.1355852030721, 24.990933481401207, 2.5], [55.13535549235394, 24.99110470506038, 2.5], [55.13512578163577, 24.99127592871955, 2.5], [55.129969653784556, 24.991440074326995, 2.5], [55.130221623112746, 24.988070688875112, 2.5], [55.130451333830905, 24.98789946521594, 2.5], [55.13089208224919, 24.98742639990359, 2.5], [55.132177586827666, 24.989003408454433, 2.5], [55.13238862452779, 24.988701566801254, 2.5], [55.132482594977674, 24.988501518707757, 2.5], [55.132525994610624, 24.988048802794115, 2.5], [55.13249018525683, 24.987180623870653, 2.5], [55.13253358488978, 24.986727907957015, 2.5], [55.1322761673244, 24.985827132742713, 2.5], [55.13163341503516, 24.98503862846729, 2.5], [55.131514764536504, 24.984469124700183, 2.5], [55.131275600894, 24.983796337257242, 2.0], [55.13066865795855, 24.98387601190528, 2.0], [55.13026930682963, 24.981537228037503, 2.0], [55.130260412698846, 24.981495691049748, 2.0], [55.13025151856806, 24.981454154061993, 2.0], [55.13022925995803, 24.98096497686874, 2.5], [55.12984453059386, 24.9804285816199, 2.5], [55.129998291954365, 24.98021419115843, 2.5], [55.12916764533149, 24.980385694214384, 2.5]]]},
]


direct_cases = [
    (LineString, [[[0, 0, 0], [1, 1]]]),
    (Polygon, [[[0, 0, 0], [1, 1, 0], [1, 1], [0, 1, 0], [0, 0, 0]]]),
    # Specific test case from #869
    (Polygon, [[[55.12916764533149, 24.980385694214384, 2.5], [55.13098248044217, 24.979828079961905], [55.13966519231666, 24.97801442415322], [55.13966563924936, 24.97801442415322], [55.14139286840762, 24.982307444496097], [55.14169331277646, 24.983717465495562], [55.14203489144224, 24.985419446276566, 2.5], [55.14180327151276, 24.98428602667792, 2.5], [55.14170091915952, 24.984242720177235, 2.5], [55.14122966992623, 24.984954809433702, 2.5], [55.14134021791831, 24.985473928648396, 2.5], [55.141405876161286, 24.986090184809793, 2.5], [55.141361358941225, 24.986138101357326, 2.5], [55.14093322994411, 24.986218753894093, 2.5], [55.140897653420964, 24.986214283545635, 2.5], [55.14095492976058, 24.9863027591922, 2.5], [55.140900447388745, 24.98628436557094, 2.5], [55.140867059473706, 24.98628869622101, 2.5], [55.14089155325796, 24.986402364143782, 2.5], [55.14090938808566, 24.986479011993385, 2.5], [55.140943893587824, 24.986471188883584, 2.5], [55.1410161176551, 24.9864174050037, 2.5], [55.140996932409635, 24.986521806266644, 2.5], [55.14163554031332, 24.986910400619593, 2.5], [55.14095781686062, 24.987033474900578, 2.5], [55.14058258698692, 24.98693261266349, 2.5], [55.14032624044253, 24.98747538747211, 2.5], [55.14007240846915, 24.988001119077232, 2.5], [55.14013122149105, 24.98831115636925, 2.5], [55.13991827457961, 24.98834356639557, 2.5], [55.139779460946755, 24.988254625087706, 2.5], [55.13974742344948, 24.988261377176524, 2.5], [55.139515198160304, 24.98841811876934, 2.5], [55.13903617238334, 24.98817914139135, 2.5], [55.1391330764994, 24.988660542040925, 2.5], [55.13914369357698, 24.989438289540374, 2.5], [55.136431216517785, 24.98966711550207, 2.0], [55.13659028641709, 24.99041706302204, 2.0], [55.1355852030721, 24.990933481401207, 2.5], [55.13535549235394, 24.99110470506038, 2.5], [55.13512578163577, 24.99127592871955, 2.5], [55.129969653784556, 24.991440074326995, 2.5], [55.130221623112746, 24.988070688875112, 2.5], [55.130451333830905, 24.98789946521594, 2.5], [55.13089208224919, 24.98742639990359, 2.5], [55.132177586827666, 24.989003408454433, 2.5], [55.13238862452779, 24.988701566801254, 2.5], [55.132482594977674, 24.988501518707757, 2.5], [55.132525994610624, 24.988048802794115, 2.5], [55.13249018525683, 24.987180623870653, 2.5], [55.13253358488978, 24.986727907957015, 2.5], [55.1322761673244, 24.985827132742713, 2.5], [55.13163341503516, 24.98503862846729, 2.5], [55.131514764536504, 24.984469124700183, 2.5], [55.131275600894, 24.983796337257242, 2.0], [55.13066865795855, 24.98387601190528, 2.0], [55.13026930682963, 24.981537228037503, 2.0], [55.130260412698846, 24.981495691049748, 2.0], [55.13025151856806, 24.981454154061993, 2.0], [55.13022925995803, 24.98096497686874, 2.5], [55.12984453059386, 24.9804285816199, 2.5], [55.129998291954365, 24.98021419115843, 2.5], [55.12916764533149, 24.980385694214384, 2.5]]]),
]


wkt_cases = [
    ('LINESTRING (1 1 1, 2 2)', 'LINESTRING (1.0000000000000000 1.0000000000000000, 2.0000000000000000 2.0000000000000000)'),
    ('POLYGON ((0 0 0, 1 0 0, 1 1, 0 1 0, 0 0 0))', 'POLYGON ((0.0000000000000000 0.0000000000000000, 1.0000000000000000 0.0000000000000000, 1.0000000000000000 1.0000000000000000, 0.0000000000000000 1.0000000000000000, 0.0000000000000000 0.0000000000000000))')
]


@pytest.mark.parametrize('geojson', geojson_cases)
def test_create_from_geojson(geojson):
    with pytest.raises(ValueError) as exc:
        wkt = shape(geojson).wkt
    assert exc.match("Inconsistent coordinate dimensionality|Input operand 0 does not have enough dimensions")


@pytest.mark.parametrize('constructor, args', direct_cases)
def test_create_directly(constructor, args):
    with pytest.raises(ValueError) as exc:
        geom = constructor(*args)
    assert exc.match("Inconsistent coordinate dimensionality|Input operand 0 does not have enough dimensions")


# TODO(shapely-2.0) pygeos adds missing z coordinate instead of dropping
@shapely20_todo
@pytest.mark.parametrize('wkt_geom,expected', wkt_cases)
def test_create_from_wkt(wkt_geom, expected):
    geom = wkt.loads(wkt_geom)
    assert geom.wkt == expected

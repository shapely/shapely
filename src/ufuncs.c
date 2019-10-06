#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <math.h>

#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC
#define PY_ARRAY_UNIQUE_SYMBOL pygeos_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL pygeos_UFUNC_API
#include <numpy/ndarraytypes.h>
#include <numpy/ufuncobject.h>
#include <numpy/npy_3kcompat.h>

#include "geos.h"
#include "pygeom.h"
#include "fast_loop_macros.h"


#define RAISE_NO_MALLOC PyErr_Format(PyExc_MemoryError, "Could not allocate memory")
#define CREATE_COORDSEQ(SIZE, NDIM)\
    void *coord_seq = GEOSCoordSeq_create_r(context_handle, SIZE, NDIM);\
    if (coord_seq == NULL) {\
        return;\
    }

#define SET_COORD(N, DIM)\
    if (!GEOSCoordSeq_setOrdinate_r(context_handle, coord_seq, N, DIM, coord)) {\
        GEOSCoordSeq_destroy_r(context_handle, coord_seq);\
        RAISE_ILLEGAL_GEOS;\
        return;\
    }

#define OUTPUT_Y\
    PyObject *ret = GeometryObject_FromGEOS(&GeometryType, ret_ptr);\
    PyObject **out = (PyObject **)op1;\
    Py_XDECREF(*out);\
    *out = ret

/* Define the geom -> bool functions (Y_b) */
static void *is_empty_data[1] = {GEOSisEmpty_r};
/* the GEOSisSimple_r function fails on geometrycollections */
static char GEOSisSimpleAllTypes_r(void *context, void *geom) {
    int type = GEOSGeomTypeId_r(context, geom);
    if (type == -1) {
        RAISE_ILLEGAL_GEOS;
        return 2;
    } else if (type == 7) {
        return 0;
    } else {
        return GEOSisSimple_r(context, geom);
    }
}
static void *is_simple_data[1] = {GEOSisSimpleAllTypes_r};
static void *is_ring_data[1] = {GEOSisRing_r};
static void *has_z_data[1] = {GEOSHasZ_r};
/* the GEOSisClosed_r function fails on non-linestrings */
static char GEOSisClosedAllTypes_r(void *context, void *geom) {
    int type = GEOSGeomTypeId_r(context, geom);
    if (type == -1) {
        /* do not SetErr: trust HandleGEOSError */
        return 2;
    } else if ((type == 1) | (type == 5)) {
        return GEOSisClosed_r(context, geom);
    } else {
        return 0;
    }
}
static void *is_closed_data[1] = {GEOSisClosedAllTypes_r};
static void *is_valid_data[1] = {GEOSisValid_r};
typedef char FuncGEOS_Y_b(void *context, void *a);
static char Y_b_dtypes[2] = {NPY_OBJECT, NPY_BOOL};
static void Y_b_func(char **args, npy_intp *dimensions,
                     npy_intp *steps, void *data)
{
    FuncGEOS_Y_b *func = (FuncGEOS_Y_b *)data;
    void *context_handle = geos_context[0];
    GEOSGeometry *in1;
    char ret;

    UNARY_LOOP {
        /* get the geometry; return on error */
        if (!get_geom(*(GeometryObject **)ip1, &in1)) { return; }
        if (in1 == NULL) {
            /* in case of a missing value: return 0 (False) */
            ret = 0;
        } else {
            /* call the GEOS function */
            ret = func(context_handle, in1);
            /* return for illegal values (trust HandleGEOSError for SetErr) */
            if (ret == 2) { return; }
        }
        *(npy_bool *)op1 = ret;
    }
}
static PyUFuncGenericFunction Y_b_funcs[1] = {&Y_b_func};

/* Define the object -> bool functions (O_b) which do not raise on non-geom objects*/
static char IsMissing(void *context, PyObject *obj) {
    return ((PyObject *) obj == Py_None);
}
static void *is_missing_data[1] = {IsMissing};
static char IsGeometry(void *context, PyObject *obj) {
    return (PyObject_IsInstance(obj, (PyObject *) &GeometryType));
}
static void *is_geometry_data[1] = {IsGeometry};
static char IsValidInput(void *context, PyObject *obj) {
    return (IsGeometry(context, obj) | IsMissing(context, obj));
}
static void *is_valid_input_data[1] = {IsValidInput};
typedef char FuncGEOS_O_b(void *context, PyObject *obj);
static char O_b_dtypes[2] = {NPY_OBJECT, NPY_BOOL};
static void O_b_func(char **args, npy_intp *dimensions,
                     npy_intp *steps, void *data)
{
    FuncGEOS_O_b *func = (FuncGEOS_O_b *)data;
    void *context_handle = geos_context[0];

    UNARY_LOOP {
        *(npy_bool *)op1 = func(context_handle, *(PyObject **)ip1);
    }
}
static PyUFuncGenericFunction O_b_funcs[1] = {&O_b_func};


/* Define the geom, geom -> bool functions (YY_b) */
static void *disjoint_data[1] = {GEOSDisjoint_r};
static void *touches_data[1] = {GEOSTouches_r};
static void *intersects_data[1] = {GEOSIntersects_r};
static void *crosses_data[1] = {GEOSCrosses_r};
static void *within_data[1] = {GEOSWithin_r};
static void *contains_data[1] = {GEOSContains_r};
static void *overlaps_data[1] = {GEOSOverlaps_r};
static void *equals_data[1] = {GEOSEquals_r};
static void *covers_data[1] = {GEOSCovers_r};
static void *covered_by_data[1] = {GEOSCoveredBy_r};
typedef char FuncGEOS_YY_b(void *context, void *a, void *b);
static char YY_b_dtypes[3] = {NPY_OBJECT, NPY_OBJECT, NPY_BOOL};
static void YY_b_func(char **args, npy_intp *dimensions,
                      npy_intp *steps, void *data)
{
    FuncGEOS_YY_b *func = (FuncGEOS_YY_b *)data;
    void *context_handle = geos_context[0];
    GEOSGeometry *in1, *in2;
    char ret;

    BINARY_LOOP {
        /* get the geometries: return on error */
        if (!get_geom(*(GeometryObject **)ip1, &in1)) { return; }
        if (!get_geom(*(GeometryObject **)ip2, &in2)) { return; }
        if ((in1 == NULL) | (in2 == NULL)) {
            /* in case of a missing value: return 0 (False) */
            ret = 0;
        } else {
            /* call the GEOS function */
            ret = func(context_handle, in1, in2);
            /* return for illegal values (trust HandleGEOSError for SetErr) */
            if (ret == 2) { return; }
        }
        *(npy_bool *)op1 = ret;
    }
}
static PyUFuncGenericFunction YY_b_funcs[1] = {&YY_b_func};

/* Define the geom -> geom functions (Y_Y) */
static void *envelope_data[1] = {GEOSEnvelope_r};
static void *convex_hull_data[1] = {GEOSConvexHull_r};
static void *GEOSBoundaryAllTypes_r(void *context, void *geom) {
    char typ = GEOSGeomTypeId_r(context, geom);
    if (typ > 3) {
        /* return NaG for geometrycollections */
        return NULL;
    } else {
        return GEOSBoundary_r(context, geom);
    }
}
static void *boundary_data[1] = {GEOSBoundaryAllTypes_r};
static void *unary_union_data[1] = {GEOSUnaryUnion_r};
static void *point_on_surface_data[1] = {GEOSPointOnSurface_r};
static void *centroid_data[1] = {GEOSGetCentroid_r};
static void *line_merge_data[1] = {GEOSLineMerge_r};
static void *extract_unique_points_data[1] = {GEOSGeom_extractUniquePoints_r};
static void *GetExteriorRing(void *context, void *geom) {
    char typ = GEOSGeomTypeId_r(context, geom);
    if (typ != 3) {
        return NULL;
    }
    void *ret = (void *) GEOSGetExteriorRing_r(context, geom);
    /* Create a copy of the obtained geometry */
    if (ret != NULL) {
        ret = GEOSGeom_clone_r(context, ret);
    }
    return ret;
}
static void *get_exterior_ring_data[1] = {GetExteriorRing};
/* a linear-ring to polygon conversion function */
static void *GEOSLinearRingToPolygon(void *context, void *geom) {
    void *shell = GEOSGeom_clone_r(context, geom);
    if (shell == NULL) { return NULL; }
    return GEOSGeom_createPolygon_r(context, shell, NULL, 0);
}
static void *polygons_without_holes_data[1] = {GEOSLinearRingToPolygon};
typedef void *FuncGEOS_Y_Y(void *context, void *a);
static char Y_Y_dtypes[2] = {NPY_OBJECT, NPY_OBJECT};
static void Y_Y_func(char **args, npy_intp *dimensions,
                     npy_intp *steps, void *data)
{
    FuncGEOS_Y_Y *func = (FuncGEOS_Y_Y *)data;
    void *context_handle = geos_context[0];
    GEOSGeometry *in1, *ret_ptr;

    UNARY_LOOP {
        /* get the geometry: return on error */
        if (!get_geom(*(GeometryObject **)ip1, &in1)) { return; }
        if (in1 == NULL) {
            /* in case of a missing value: return NULL (NaG) */
            ret_ptr = NULL;
        } else {
            ret_ptr = func(context_handle, in1);
            /* trust that GEOS calls HandleGEOSError on error */
        }
        OUTPUT_Y;
    }
}
static PyUFuncGenericFunction Y_Y_funcs[1] = {&Y_Y_func};

/* Define the geom, double -> geom functions (Yd_Y) */
static void *interpolate_data[1] = {GEOSInterpolate_r};
static void *interpolate_normalized_data[1] = {GEOSInterpolateNormalized_r};
static void *simplify_data[1] = {GEOSSimplify_r};
static void *simplify_preserve_topology_data[1] = {GEOSTopologyPreserveSimplify_r};
typedef void *FuncGEOS_Yd_Y(void *context, void *a, double b);
static char Yd_Y_dtypes[3] = {NPY_OBJECT, NPY_DOUBLE, NPY_OBJECT};
static void Yd_Y_func(char **args, npy_intp *dimensions,
                      npy_intp *steps, void *data)
{
    FuncGEOS_Yd_Y *func = (FuncGEOS_Yd_Y *)data;
    void *context_handle = geos_context[0];
    GEOSGeometry *in1, *ret_ptr;

    BINARY_LOOP {
        /* get the geometry: return on error */
        if (!get_geom(*(GeometryObject **)ip1, &in1)) { return; }
        double in2 = *(double *)ip2;
        if ((in1 == NULL) | (npy_isnan(in2))) {
            ret_ptr = NULL;
        } else {
            ret_ptr = func(context_handle, in1, in2);
            /* trust that GEOS calls HandleGEOSError on error */
        }
        OUTPUT_Y;
    }
}
static PyUFuncGenericFunction Yd_Y_funcs[1] = {&Yd_Y_func};

/* Define the geom, int -> geom functions (Yi_Y) */
/* We add bound and type checking to the various indexing functions */
static void *GetPointN(void *context, void *geom, int n) {
    char typ = GEOSGeomTypeId_r(context, geom);
    int size, i;
    if ((typ != 1) & (typ != 2)) {
        return NULL;
    }
    size = GEOSGeomGetNumPoints_r(context, geom);
    if (size == -1) {
        return NULL;
    }
    if (n < 0) {
        /* Negative indexing: we get it for free */
        i = size + n;
    } else {
        i = n;
    }
    if ((i < 0) | (i >= size)) {
        /* Important, could give segfaults else */
        return NULL;
    }
    return GEOSGeomGetPointN_r(context, geom, i);
}
static void *get_point_data[1] = {GetPointN};
static void *GetInteriorRingN(void *context, void *geom, int n) {
    char typ = GEOSGeomTypeId_r(context, geom);
    int size, i;
    if (typ != 3) {
        return NULL;
    }
    size = GEOSGetNumInteriorRings_r(context, geom);
    if (size == -1) {
        return NULL;
    }
    if (n < 0) {
        /* Negative indexing: we get it for free */
        i = size + n;
    } else {
        i = n;
    }
    if ((i < 0) | (i >= size)) {
        /* Important, could give segfaults else */
        return NULL;
    }
    void *ret = (void *) GEOSGetInteriorRingN_r(context, geom, i);
    /* Create a copy of the obtained geometry */
    if (ret != NULL) {
        ret = GEOSGeom_clone_r(context, ret);
    }
    return ret;
}
static void *get_interior_ring_data[1] = {GetInteriorRingN};
static void *GetGeometryN(void *context, void *geom, int n) {
    int size, i;
    size = GEOSGetNumGeometries_r(context, geom);
    if (size == -1) {
        return NULL;
    }
    if (n < 0) {
        /* Negative indexing: we get it for free */
        i = size + n;
    } else {
        i = n;
    }
    if ((i < 0) | (i >= size)) {
        /* Important, could give segfaults else */
        return NULL;
    }
    void *ret = (void *) GEOSGetGeometryN_r(context, geom, i);
    /* Create a copy of the obtained geometry */
    if (ret != NULL) {
        ret = GEOSGeom_clone_r(context, ret);
    }
    return ret;
}
static void *get_geometry_data[1] = {GetGeometryN};
/* the set srid funcion acts inplace */
static void *GEOSSetSRID_r_with_clone(void *context, void *geom, int srid) {
    void *ret = GEOSGeom_clone_r(context, geom);
    if (ret == NULL) { return NULL; }
    GEOSSetSRID_r(context, ret, srid);
    return ret;
}
static void *set_srid_data[1] = {GEOSSetSRID_r_with_clone};
typedef void *FuncGEOS_Yi_Y(void *context, void *a, int b);
static char Yi_Y_dtypes[3] = {NPY_OBJECT, NPY_INT, NPY_OBJECT};
static void Yi_Y_func(char **args, npy_intp *dimensions,
                      npy_intp *steps, void *data)
{
    FuncGEOS_Yi_Y *func = (FuncGEOS_Yi_Y *)data;
    void *context_handle = geos_context[0];
    GEOSGeometry *in1, *ret_ptr;

    BINARY_LOOP {
        /* get the geometry: return on error */
        if (!get_geom(*(GeometryObject **)ip1, &in1)) { return; }
        int in2 = *(int *) ip2;
        if (in1 == NULL) {
            ret_ptr = NULL;
        } else {
            ret_ptr = func(context_handle, in1, in2);
            /* trust that GEOS calls HandleGEOSError on error */
        }
        OUTPUT_Y;
    }
}
static PyUFuncGenericFunction Yi_Y_funcs[1] = {&Yi_Y_func};

/* Define the geom, geom -> geom functions (YY_Y) */
static void *intersection_data[1] = {GEOSIntersection_r};
static void *difference_data[1] = {GEOSDifference_r};
static void *symmetric_difference_data[1] = {GEOSSymDifference_r};
static void *union_data[1] = {GEOSUnion_r};
static void *shared_paths_data[1] = {GEOSSharedPaths_r};
typedef void *FuncGEOS_YY_Y(void *context, void *a, void *b);
static char YY_Y_dtypes[3] = {NPY_OBJECT, NPY_OBJECT, NPY_OBJECT};
static void YY_Y_func(char **args, npy_intp *dimensions,
                      npy_intp *steps, void *data)
{
    FuncGEOS_YY_Y *func = (FuncGEOS_YY_Y *)data;
    void *context_handle = geos_context[0];
    GEOSGeometry *in1, *in2, *ret_ptr;

    BINARY_LOOP {
        /* get the geometries: return on error */
        if (!get_geom(*(GeometryObject **)ip1, &in1)) { return; }
        if (!get_geom(*(GeometryObject **)ip2, &in2)) { return; }
        if ((in1 == NULL) | (in2 == NULL)) {
            /* in case of a missing value: return NULL (NaG) */
            ret_ptr = NULL;
        } else {
            ret_ptr = func(context_handle, in1, in2);
            /* trust that GEOS calls HandleGEOSError on error */
        }
        OUTPUT_Y;
    }
}
static PyUFuncGenericFunction YY_Y_funcs[1] = {&YY_Y_func};

/* Define the geom -> double functions (Y_d) */
static int GetX(void *context, void *a, double *b) {
    char typ = GEOSGeomTypeId_r(context, a);
    if (typ != 0) {
        *(double *)b = NPY_NAN;
        return 1;
    } else {
        return GEOSGeomGetX_r(context, a, b);
    }
}
static void *get_x_data[1] = {GetX};
static int GetY(void *context, void *a, double *b) {
    char typ = GEOSGeomTypeId_r(context, a);
    if (typ != 0) {
        *(double *)b = NPY_NAN;
        return 1;
    } else {
        return GEOSGeomGetY_r(context, a, b);
    }
}
static void *get_y_data[1] = {GetY};
static void *area_data[1] = {GEOSArea_r};
static void *length_data[1] = {GEOSLength_r};
typedef int FuncGEOS_Y_d(void *context, void *a, double *b);
static char Y_d_dtypes[2] = {NPY_OBJECT, NPY_DOUBLE};
static void Y_d_func(char **args, npy_intp *dimensions,
                     npy_intp *steps, void *data)
{
    FuncGEOS_Y_d *func = (FuncGEOS_Y_d *)data;
    void *context_handle = geos_context[0];
    GEOSGeometry *in1;

    UNARY_LOOP {
        /* get the geometry: return on error */
        if (!get_geom(*(GeometryObject **)ip1, &in1)) { return; }
        if (in1 == NULL) {
            *(double *)op1 = NPY_NAN;
        } else {
            /* let the GEOS function set op1; return on error */
            if (func(context_handle, in1, (npy_double *) op1) == 0) { return; }
        }
    }
}
static PyUFuncGenericFunction Y_d_funcs[1] = {&Y_d_func};

/* Define the geom -> int functions (Y_i) */
static void *get_type_id_data[1] = {GEOSGeomTypeId_r};
static void *get_dimensions_data[1] = {GEOSGeom_getDimensions_r};
static void *get_coordinate_dimensions_data[1] = {GEOSGeom_getCoordinateDimension_r};
static void *get_srid_data[1] = {GEOSGetSRID_r};
static int GetNumPoints(void *context, void *geom, int n) {
    char typ = GEOSGeomTypeId_r(context, geom);
    if ((typ == 1) | (typ == 2)) {  /* Linestring & Linearring */
        return GEOSGeomGetNumPoints_r(context, geom);
    } else {
        return 0;
    }
}
static void *get_num_points_data[1] = {GetNumPoints};
static int GetNumInteriorRings(void *context, void *geom, int n) {
    char typ = GEOSGeomTypeId_r(context, geom);
    if (typ == 3) {   /* Polygon */
        return GEOSGetNumInteriorRings_r(context, geom);
    } else {
        return 0;
    }
}
static void *get_num_interior_rings_data[1] = {GetNumInteriorRings};
static void *get_num_geometries_data[1] = {GEOSGetNumGeometries_r};
static void *get_num_coordinates_data[1] = {GEOSGetNumCoordinates_r};
typedef int FuncGEOS_Y_i(void *context, void *a);
static char Y_i_dtypes[2] = {NPY_OBJECT, NPY_INT};
static void Y_i_func(char **args, npy_intp *dimensions,
                     npy_intp *steps, void *data)
{
    FuncGEOS_Y_i *func = (FuncGEOS_Y_i *)data;
    void *context_handle = geos_context[0];
    GEOSGeometry *in1;

    UNARY_LOOP {
        /* get the geometry: return on error */
        if (!get_geom(*(GeometryObject **)ip1, &in1)) { return; }
        if (in1 == NULL) {
            /* NaG results in -1 */
            *(npy_int *)op1 = -1;
        } else {
            /* trust that GEOS calls the error handler if necessary */
            *(npy_int *)op1 = func(context_handle, in1);
        }
    }
}
static PyUFuncGenericFunction Y_i_funcs[1] = {&Y_i_func};

/* Define the geom, geom -> double functions (YY_d) */
static void *distance_data[1] = {GEOSDistance_r};
static void *hausdorff_distance_data[1] = {GEOSHausdorffDistance_r};
/* Project and ProjectNormalize don't return error codes. wrap them. */
static int GEOSProjectWrapped_r(void *context, void *a,  void *b, double *c) {
    *c = GEOSProject_r(context, a, b);
    return 1;
}
static void *project_data[1] = {GEOSProjectWrapped_r};
static int GEOSProjectNormalizedWrapped_r(void *context, void *a,  void *b, double *c) {
    *c = GEOSProjectNormalized_r(context, a, b);
    return 1;
}
static void *project_normalized_data[1] = {GEOSProjectNormalizedWrapped_r};
typedef int FuncGEOS_YY_d(void *context, void *a,  void *b, double *c);
static char YY_d_dtypes[3] = {NPY_OBJECT, NPY_OBJECT, NPY_DOUBLE};
static void YY_d_func(char **args, npy_intp *dimensions,
                      npy_intp *steps, void *data)
{
    FuncGEOS_YY_d *func = (FuncGEOS_YY_d *)data;
    void *context_handle = geos_context[0];
    GEOSGeometry *in1, *in2;

    BINARY_LOOP {
        /* get the geometries: return on error */
        if (!get_geom(*(GeometryObject **)ip1, &in1)) { return; }
        if (!get_geom(*(GeometryObject **)ip2, &in2)) { return; }
        if ((in1 == NULL) | (in2 == NULL)) {
            /* in case of a missing value: return NaN */
            *(double *)op1 = NPY_NAN;
        } else {
            /* let the GEOS function  set op1; return on error */
            if (func(context_handle, in1, in2, (double *) op1) == 0) { return; }
            /* incase the outcome is 0.0, check the inputs for emptyness */
            if (*op1 == 0.0) {
                if (GEOSisEmpty_r(context_handle, in1) | GEOSisEmpty_r(context_handle, in2)) {
                    *(double *)op1 = NPY_NAN;
                }
            }
        }
    }
}
static PyUFuncGenericFunction YY_d_funcs[1] = {&YY_d_func};

/* Define functions with unique call signatures */
static void *null_data[1] = {NULL};
static char buffer_inner(void *context_handle, GEOSBufferParams *params, void *ip1, void *ip2, void *op1) {
    GEOSGeometry *in1, *ret_ptr;

    /* get the geometry: return on error */
    if (!get_geom(*(GeometryObject **)ip1, &in1)) { return 0; }
    double in2 = *(double *) ip2;
     /* handle NULL geometries or NaN buffer width */
    if ((in1 == NULL) | npy_isnan(in2)) {
        ret_ptr = NULL;
    } else {
        ret_ptr = GEOSBufferWithParams_r(context_handle, in1, params, in2);
        if (ret_ptr == NULL) { return 0; }
    }
    OUTPUT_Y;
    return 1;
}

static char buffer_dtypes[8] = {NPY_OBJECT, NPY_DOUBLE, NPY_INT, NPY_INT, NPY_INT, NPY_DOUBLE, NPY_BOOL, NPY_OBJECT};
static void buffer_func(char **args, npy_intp *dimensions,
                        npy_intp *steps, void *data)
{
    void *context_handle = geos_context[0];
    char *ip1 = args[0], *ip2 = args[1], *ip3 = args[2], *ip4 = args[3], *ip5 = args[4], *ip6 = args[5], *ip7 = args[6], *op1 = args[7];
    npy_intp is1 = steps[0], is2 = steps[1], is3 = steps[2], is4 = steps[3], is5 = steps[4], is6 = steps[5], is7 = steps[6], os1 = steps[7];
    npy_intp n = dimensions[0];
    npy_intp i;

    if ((is3 != 0) | (is4 != 0) | (is5 != 0) | (is6 != 0) | (is7 != 0)) {
        PyErr_Format(PyExc_ValueError, "Buffer function called with non-scalar parameters");
        return;
    }

    GEOSBufferParams *params = GEOSBufferParams_create_r(context_handle);
    if (params == 0) { return; }
    if (!GEOSBufferParams_setQuadrantSegments_r(context_handle, params, *(int *) ip3)) {
        goto fail;
    }
    if (!GEOSBufferParams_setEndCapStyle_r(context_handle, params, *(int *) ip4)) {
        goto fail;
    }
    if (!GEOSBufferParams_setJoinStyle_r(context_handle, params, *(int *) ip5)) {
        goto fail;
    }
    if (!GEOSBufferParams_setMitreLimit_r(context_handle, params, *(double *) ip6)) {
        goto fail;
    }
    if (!GEOSBufferParams_setSingleSided_r(context_handle, params, *(npy_bool *) ip7)) {
        goto fail;
    }

    for(i = 0; i < n; i++, ip1 += is1, ip2 += is2, op1 += os1) {
        if (!buffer_inner(context_handle, params, ip1, ip2, op1)) { goto fail; }
    }

    fail:
        GEOSBufferParams_destroy_r(context_handle, params);
}
static PyUFuncGenericFunction buffer_funcs[1] = {&buffer_func};


static char snap_dtypes[4] = {NPY_OBJECT, NPY_OBJECT, NPY_DOUBLE, NPY_OBJECT};
static void snap_func(char **args, npy_intp *dimensions,
                      npy_intp *steps, void *data)
{
    void *context_handle = geos_context[0];
    GEOSGeometry *in1, *in2, *ret_ptr;

    TERNARY_LOOP {
        /* get the geometries: return on error */
        if (!get_geom(*(GeometryObject **)ip1, &in1)) { return; }
        if (!get_geom(*(GeometryObject **)ip2, &in2)) { return; }
        double in3 = *(double *) ip3;
        if ((in1 == NULL) | (in2 == NULL) | npy_isnan(in3)) {
            /* in case of a missing value: return NULL (NaG) */
            ret_ptr = NULL;
        } else {
            ret_ptr = GEOSSnap_r(context_handle, in1, in2, in3);
        }
        OUTPUT_Y;
    }
}
static PyUFuncGenericFunction snap_funcs[1] = {&snap_func};

static char equals_exact_dtypes[4] = {NPY_OBJECT, NPY_OBJECT, NPY_DOUBLE, NPY_BOOL};
static void equals_exact_func(char **args, npy_intp *dimensions,
                      npy_intp *steps, void *data)
{
    void *context_handle = geos_context[0];
    GEOSGeometry *in1, *in2;
    npy_bool ret;

    TERNARY_LOOP {
        /* get the geometries: return on error */
        if (!get_geom(*(GeometryObject **)ip1, &in1)) { return; }
        if (!get_geom(*(GeometryObject **)ip2, &in2)) { return; }
        double in3 = *(double *) ip3;
        if ((in1 == NULL) | (in2 == NULL) | npy_isnan(in3)) {
            /* return 0 (False) for missing values */
            ret = 0;
        } else {
            ret = GEOSEqualsExact_r(context_handle, in1, in2, in3);
            /* return for illegal values (trust HandleGEOSError for SetErr) */
            if ((ret != 0) & (ret != 1)) { return; }
        }
        *(npy_bool *)op1 = ret;
    }
}
static PyUFuncGenericFunction equals_exact_funcs[1] = {&equals_exact_func};


static char haussdorf_distance_densify_dtypes[4] = {NPY_OBJECT, NPY_OBJECT, NPY_DOUBLE, NPY_DOUBLE};
static void haussdorf_distance_densify_func(char **args, npy_intp *dimensions,
                                            npy_intp *steps, void *data)
{
    void *context_handle = geos_context[0];
    GEOSGeometry *in1, *in2;

    TERNARY_LOOP {
        /* get the geometries: return on error */
        if (!get_geom(*(GeometryObject **)ip1, &in1)) { return; }
        if (!get_geom(*(GeometryObject **)ip2, &in2)) { return; }
        double in3 = *(double *) ip3;
        if ((in1 == NULL) | (in2 == NULL) | npy_isnan(in3)) {
            *(double *)op1 = NPY_NAN;
        } else {
            /* let the GEOS function set op1; return on error */
            if (GEOSHausdorffDistanceDensify_r(context_handle, in1, in2, in3, (double *) op1) == 0) { return ; }
            if (*op1 == 0.0) {
                if (GEOSisEmpty_r(context_handle, in1) | GEOSisEmpty_r(context_handle, in2)) {
                    *(double *)op1 = NPY_NAN;
                }
            }
        }
    }
}
static PyUFuncGenericFunction haussdorf_distance_densify_funcs[1] = {&haussdorf_distance_densify_func};


static char delaunay_triangles_dtypes[4] = {NPY_OBJECT, NPY_DOUBLE, NPY_BOOL, NPY_OBJECT};
static void delaunay_triangles_func(char **args, npy_intp *dimensions,
                                    npy_intp *steps, void *data)
{
    void *context_handle = geos_context[0];
    GEOSGeometry *in1, *ret_ptr;

    TERNARY_LOOP {
        /* get the geometry: return on error */
        if (!get_geom(*(GeometryObject **)ip1, &in1)) { return; }
        double in2 = *(double *) ip2;
        npy_bool in3 = *(npy_bool *) ip3;
        if ((in1 == NULL) | npy_isnan(in2)) {
            /* propagate NULL geometries */
            ret_ptr = NULL;
        } else {
            ret_ptr = GEOSDelaunayTriangulation_r(context_handle, in1, in2, (int) in3);
            /* trust that GEOS calls HandleGEOSError on error */
        }
        OUTPUT_Y;
    }
}
static PyUFuncGenericFunction delaunay_triangles_funcs[1] = {&delaunay_triangles_func};


static char voronoi_polygons_dtypes[5] = {NPY_OBJECT, NPY_DOUBLE, NPY_OBJECT, NPY_BOOL, NPY_OBJECT};
static void voronoi_polygons_func(char **args, npy_intp *dimensions,
                                  npy_intp *steps, void *data)
{
    void *context_handle = geos_context[0];
    GEOSGeometry *in1, *in3, *ret_ptr;

    QUATERNARY_LOOP {
        /* get the geometry: return on error */
        if (!get_geom(*(GeometryObject **)ip1, &in1)) { return; }
        double in2 = *(double *) ip2;
        if (!get_geom(*(GeometryObject **)ip3, &in3)) { return; }
        npy_bool in4 = *(npy_bool *) ip4;
        if ((in1 == NULL) | npy_isnan(in2)) {
            /* propagate NULL geometries; in3 = NULL is actually supported */
            ret_ptr = NULL;
        } else {
            ret_ptr = GEOSVoronoiDiagram_r(context_handle, in1, in3, in2, (int) in4);
            /* trust that GEOS calls HandleGEOSError on error */
        }
        OUTPUT_Y;
    }
}
static PyUFuncGenericFunction voronoi_polygons_funcs[1] = {&voronoi_polygons_func};

static char is_valid_reason_dtypes[2] = {NPY_OBJECT, NPY_OBJECT};
static void is_valid_reason_func(char **args, npy_intp *dimensions,
                                 npy_intp *steps, void *data)
{
    void *context_handle = geos_context[0];
    char *reason;
    GEOSGeometry *in1;

    UNARY_LOOP {
        PyObject **out = (PyObject **)op1;
        /* get the geometry return on error */
        if (!get_geom(*(GeometryObject **)ip1, &in1)) { return; }
        if ((in1 == NULL)) {
            /* Missing geometries give None */
            Py_XDECREF(*out);
            Py_INCREF(Py_None);
            *out = Py_None;
        } else {
            reason = GEOSisValidReason_r(context_handle, in1);
            /* handle NULL return value; GEOS will call SetErr */
            if (reason == NULL) { return ; }
            /* convert to python string and set to out */
            Py_XDECREF(*out);
            *out = PyUnicode_FromString(reason);
            GEOSFree_r(context_handle, reason);
        }
    }
}
static PyUFuncGenericFunction is_valid_reason_funcs[1] = {&is_valid_reason_func};

/* define double -> geometry construction functions */
static char points_dtypes[2] = {NPY_DOUBLE, NPY_OBJECT};
static void points_func(char **args, npy_intp *dimensions,
                        npy_intp *steps, void *data)
{
    void *context_handle = geos_context[0];
    SINGLE_COREDIM_LOOP_OUTER {
        CREATE_COORDSEQ(1, n_c1);
        SINGLE_COREDIM_LOOP_INNER {
            double coord = *(double *) cp1;
            SET_COORD(0, i_c1);
        }
        GEOSGeometry *ret_ptr = GEOSGeom_createPoint_r(context_handle, coord_seq);
        OUTPUT_Y;
    }
}
static PyUFuncGenericFunction points_funcs[1] = {&points_func};


static char linestrings_dtypes[2] = {NPY_DOUBLE, NPY_OBJECT};
static void linestrings_func(char **args, npy_intp *dimensions,
                              npy_intp *steps, void *data)
{
    void *context_handle = geos_context[0];
    DOUBLE_COREDIM_LOOP_OUTER {
        CREATE_COORDSEQ(n_c1, n_c2);
        DOUBLE_COREDIM_LOOP_INNER_1 {
            DOUBLE_COREDIM_LOOP_INNER_2 {
                double coord = *(double *) cp2;
                SET_COORD(i_c1, i_c2);
            }
        }
        GEOSGeometry *ret_ptr = GEOSGeom_createLineString_r(context_handle, coord_seq);
        OUTPUT_Y;
    }
}
static PyUFuncGenericFunction linestrings_funcs[1] = {&linestrings_func};


static char linearrings_dtypes[2] = {NPY_DOUBLE, NPY_OBJECT};
static void linearrings_func(char **args, npy_intp *dimensions,
                             npy_intp *steps, void *data)
{
    void *context_handle = geos_context[0];
    DOUBLE_COREDIM_LOOP_OUTER {
        /* check if first and last coords are equal; duplicate if necessary */
        char ring_closure = 0;
        DOUBLE_COREDIM_LOOP_INNER_2 {
            double first_coord = *(double *) (ip1 + i_c2 * cs2);
            double last_coord = *(double *) (ip1 + (n_c1 - 1) * cs1 + i_c2 * cs2);
            if (first_coord != last_coord) {
                ring_closure = 1;
                break;
            }
        }
        /* fill the coordinate sequence */
        CREATE_COORDSEQ(n_c1 + ring_closure, n_c2);
        DOUBLE_COREDIM_LOOP_INNER_1 {
            DOUBLE_COREDIM_LOOP_INNER_2 {
                double coord = *(double *) cp2;
                SET_COORD(i_c1, i_c2);
            }
        }
        /* add the closing coordinate if necessary */
        if (ring_closure) {
            DOUBLE_COREDIM_LOOP_INNER_2 {
                double coord = *(double *) (ip1 + i_c2 * cs2);
                SET_COORD(n_c1, i_c2);
            }
        }
        GEOSGeometry *ret_ptr = GEOSGeom_createLinearRing_r(context_handle, coord_seq);
        OUTPUT_Y;
    }
}
static PyUFuncGenericFunction linearrings_funcs[1] = {&linearrings_func};


static char polygons_with_holes_dtypes[3] = {NPY_OBJECT, NPY_OBJECT, NPY_OBJECT};
static void polygons_with_holes_func(char **args, npy_intp *dimensions,
                                     npy_intp *steps, void *data)
{
    void *context_handle = geos_context[0];
    GEOSGeometry *hole, *shell, *hole_copy, *shell_copy;
    int n_holes;
    GEOSGeometry **holes = malloc(sizeof(void *) * dimensions[1]);
    if (holes == NULL) {
        RAISE_NO_MALLOC;
        goto finish;
    }

    BINARY_SINGLE_COREDIM_LOOP_OUTER {
        if (!get_geom(*(GeometryObject **)ip1, &shell)) { goto finish; }
        shell_copy = GEOSGeom_clone_r(context_handle, shell);
        if (shell_copy == NULL) { goto finish; }
        n_holes = 0;
        cp1 = ip2;
        BINARY_SINGLE_COREDIM_LOOP_INNER {
            if (!get_geom(*(GeometryObject **)cp1, &hole)) { goto finish; }
            if (hole == NULL) { continue; }
            hole_copy = GEOSGeom_clone_r(context_handle, hole);
            if (hole_copy == NULL) { goto finish; }
            holes[n_holes] = hole_copy;
            n_holes++;
        }
        GEOSGeometry *ret_ptr = GEOSGeom_createPolygon_r(context_handle, shell_copy, holes, n_holes);
        OUTPUT_Y;
    }

    finish:
        if (holes != NULL) { free(holes); }
}
static PyUFuncGenericFunction polygons_with_holes_funcs[1] = {&polygons_with_holes_func};


static char create_collection_dtypes[3] = {NPY_OBJECT, NPY_INT, NPY_OBJECT};
static void create_collection_func(char **args, npy_intp *dimensions,
                                   npy_intp *steps, void *data)
{
    void *context_handle = geos_context[0];
    GEOSGeometry *g, *g_copy;
    int n_geoms;
    GEOSGeometry **geoms = malloc(sizeof(void *) * dimensions[1]);
    if (geoms == NULL) {
        RAISE_NO_MALLOC;
        goto finish;
    }
    int type;

    BINARY_SINGLE_COREDIM_LOOP_OUTER {
        type = *(int *) ip2;
        n_geoms = 0;
        cp1 = ip1;
        BINARY_SINGLE_COREDIM_LOOP_INNER {
            if (!get_geom(*(GeometryObject **)cp1, &g)) { goto finish; }
            if (g == NULL) { continue; }
            g_copy = GEOSGeom_clone_r(context_handle, g);
            if (g_copy == NULL) { goto finish; }
            geoms[n_geoms] = g_copy;
            n_geoms++;
        }
        GEOSGeometry *ret_ptr = GEOSGeom_createCollection_r(context_handle, type, geoms, n_geoms);
        OUTPUT_Y;
    }

    finish:
        if (geoms != NULL) { free(geoms); }
}
static PyUFuncGenericFunction create_collection_funcs[1] = {&create_collection_func};

/*
TODO polygonizer functions
TODO prepared geometry predicate functions
TODO relate functions
*/


#define DEFINE_Y_b(NAME)\
    ufunc = PyUFunc_FromFuncAndData(Y_b_funcs, NAME ##_data, Y_b_dtypes, 1, 1, 1, PyUFunc_None, # NAME, NULL, 0);\
    PyDict_SetItemString(d, # NAME, ufunc)

#define DEFINE_O_b(NAME)\
    ufunc = PyUFunc_FromFuncAndData(O_b_funcs, NAME ##_data, O_b_dtypes, 1, 1, 1, PyUFunc_None, # NAME, NULL, 0);\
    PyDict_SetItemString(d, # NAME, ufunc)

#define DEFINE_YY_b(NAME)\
    ufunc = PyUFunc_FromFuncAndData(YY_b_funcs, NAME ##_data, YY_b_dtypes, 1, 2, 1, PyUFunc_None, # NAME, "", 0);\
    PyDict_SetItemString(d, # NAME, ufunc)

#define DEFINE_Y_Y(NAME)\
    ufunc = PyUFunc_FromFuncAndData(Y_Y_funcs, NAME ##_data, Y_Y_dtypes, 1, 1, 1, PyUFunc_None, # NAME, "", 0);\
    PyDict_SetItemString(d, # NAME, ufunc)

#define DEFINE_Yi_Y(NAME)\
    ufunc = PyUFunc_FromFuncAndData(Yi_Y_funcs, NAME ##_data, Yi_Y_dtypes, 1, 2, 1, PyUFunc_None, # NAME, "", 0);\
    PyDict_SetItemString(d, # NAME, ufunc)

#define DEFINE_Yd_Y(NAME)\
    ufunc = PyUFunc_FromFuncAndData(Yd_Y_funcs, NAME ##_data, Yd_Y_dtypes, 1, 2, 1, PyUFunc_None, # NAME, "", 0);\
    PyDict_SetItemString(d, # NAME, ufunc)

#define DEFINE_YY_Y(NAME)\
    ufunc = PyUFunc_FromFuncAndData(YY_Y_funcs, NAME ##_data, YY_Y_dtypes, 1, 2, 1, PyUFunc_None, # NAME, "", 0);\
    PyDict_SetItemString(d, # NAME, ufunc)

#define DEFINE_Y_d(NAME)\
    ufunc = PyUFunc_FromFuncAndData(Y_d_funcs, NAME ##_data, Y_d_dtypes, 1, 1, 1, PyUFunc_None, # NAME, "", 0);\
    PyDict_SetItemString(d, # NAME, ufunc)

#define DEFINE_Y_B(NAME)\
    ufunc = PyUFunc_FromFuncAndData(Y_B_funcs, NAME ##_data, Y_B_dtypes, 1, 1, 1, PyUFunc_None, # NAME, "", 0);\
    PyDict_SetItemString(d, # NAME, ufunc)

#define DEFINE_Y_i(NAME)\
    ufunc = PyUFunc_FromFuncAndData(Y_i_funcs, NAME ##_data, Y_i_dtypes, 1, 1, 1, PyUFunc_None, # NAME, "", 0);\
    PyDict_SetItemString(d, # NAME, ufunc)

#define DEFINE_YY_d(NAME)\
    ufunc = PyUFunc_FromFuncAndData(YY_d_funcs, NAME ##_data, YY_d_dtypes, 1, 2, 1, PyUFunc_None, # NAME, "", 0);\
    PyDict_SetItemString(d, # NAME, ufunc)

#define DEFINE_CUSTOM(NAME, N_IN)\
    ufunc = PyUFunc_FromFuncAndData(NAME ##_funcs, null_data, NAME ##_dtypes, 1, N_IN, 1, PyUFunc_None, # NAME, "", 0);\
    PyDict_SetItemString(d, # NAME, ufunc)

#define DEFINE_GENERALIZED(NAME, N_IN, SIGNATURE)\
    ufunc = PyUFunc_FromFuncAndDataAndSignature(NAME ##_funcs, null_data, NAME ##_dtypes, 1, N_IN, 1, PyUFunc_None, # NAME, "", 0, SIGNATURE);\
    PyDict_SetItemString(d, # NAME, ufunc)

int init_ufuncs(PyObject *m, PyObject *d)
{
    PyObject *ufunc;

    DEFINE_Y_b (is_empty);
    DEFINE_Y_b (is_simple);
    DEFINE_Y_b (is_geometry);
    DEFINE_Y_b (is_ring);
    DEFINE_Y_b (has_z);
    DEFINE_Y_b (is_closed);
    DEFINE_Y_b (is_valid);

    DEFINE_O_b (is_geometry);
    DEFINE_O_b (is_missing);
    DEFINE_O_b (is_valid_input);

    DEFINE_YY_b (disjoint);
    DEFINE_YY_b (touches);
    DEFINE_YY_b (intersects);
    DEFINE_YY_b (crosses);
    DEFINE_YY_b (within);
    DEFINE_YY_b (contains);
    DEFINE_YY_b (overlaps);
    DEFINE_YY_b (equals);
    DEFINE_YY_b (covers);
    DEFINE_YY_b (covered_by);

    DEFINE_Y_Y (envelope);
    DEFINE_Y_Y (convex_hull);
    DEFINE_Y_Y (boundary);
    DEFINE_Y_Y (unary_union);
    DEFINE_Y_Y (point_on_surface);
    DEFINE_Y_Y (centroid);
    DEFINE_Y_Y (line_merge);
    DEFINE_Y_Y (extract_unique_points);
    DEFINE_Y_Y (get_exterior_ring);

    DEFINE_Yi_Y (get_point);
    DEFINE_Yi_Y (get_interior_ring);
    DEFINE_Yi_Y (get_geometry);
    DEFINE_Yi_Y (set_srid);

    DEFINE_Yd_Y (interpolate);
    DEFINE_Yd_Y (interpolate_normalized);
    DEFINE_Yd_Y (simplify);
    DEFINE_Yd_Y (simplify_preserve_topology);

    DEFINE_YY_Y (intersection);
    DEFINE_YY_Y (difference);
    DEFINE_YY_Y (symmetric_difference);
    DEFINE_YY_Y (union);
    DEFINE_YY_Y (shared_paths);

    DEFINE_Y_d (get_x);
    DEFINE_Y_d (get_y);
    DEFINE_Y_d (area);
    DEFINE_Y_d (length);

    DEFINE_Y_i (get_type_id);
    DEFINE_Y_i (get_dimensions);
    DEFINE_Y_i (get_coordinate_dimensions);
    DEFINE_Y_i (get_srid);
    DEFINE_Y_i (get_num_points);
    DEFINE_Y_i (get_num_interior_rings);
    DEFINE_Y_i (get_num_geometries);
    DEFINE_Y_i (get_num_coordinates);

    DEFINE_YY_d (distance);
    DEFINE_YY_d (hausdorff_distance);
    DEFINE_YY_d (project);
    DEFINE_YY_d (project_normalized);

    DEFINE_CUSTOM (buffer, 7);
    DEFINE_CUSTOM (snap, 3);
    DEFINE_CUSTOM (equals_exact, 3);
    DEFINE_CUSTOM (haussdorf_distance_densify, 3);
    DEFINE_CUSTOM (delaunay_triangles, 3);
    DEFINE_CUSTOM (voronoi_polygons, 4);
    DEFINE_CUSTOM (is_valid_reason, 1);
    DEFINE_GENERALIZED(points, 1, "(d)->()");
    DEFINE_GENERALIZED(linestrings, 1, "(i, d)->()");
    DEFINE_GENERALIZED(linearrings, 1, "(i, d)->()");
    DEFINE_Y_Y (polygons_without_holes);
    DEFINE_GENERALIZED(polygons_with_holes, 2, "(),(i)->()");
    DEFINE_GENERALIZED(create_collection, 2, "(i),()->()");

    Py_DECREF(ufunc);
    return 0;
}

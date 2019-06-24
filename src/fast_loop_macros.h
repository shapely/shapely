/**
 * Copied from numpy/src/comp/umath/fast_loop_macros.h
 *
 * Macros to help build fast ufunc inner loops.
 *
 * These expect to have access to the arguments of a typical ufunc loop,
 *
 *     char **args
 *     npy_intp *dimensions
 *     npy_intp *steps
 */

/** (ip1) -> (op1) */
#define UNARY_LOOP\
    char *ip1 = args[0], *op1 = args[1];\
    npy_intp is1 = steps[0], os1 = steps[1];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for(i = 0; i < n; i++, ip1 += is1, op1 += os1)


/** (ip1, ip2) -> (op1) */
#define BINARY_LOOP\
    char *ip1 = args[0], *ip2 = args[1], *op1 = args[2];\
    npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for(i = 0; i < n; i++, ip1 += is1, ip2 += is2, op1 += os1)


/** (ip1, ip2, ip3) -> (op1) */
#define TERNARY_LOOP\
    char *ip1 = args[0], *ip2 = args[1], *ip3 = args[2], *op1 = args[3];\
    npy_intp is1 = steps[0], is2 = steps[1], is3 = steps[2], os1 = steps[3];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for(i = 0; i < n; i++, ip1 += is1, ip2 += is2, ip3 += is3, op1 += os1)


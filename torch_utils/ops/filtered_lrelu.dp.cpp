// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "filtered_lrelu.h"
#include <sycl/accessor.hpp>
#include <ipex.h>
#include <torch/extension.h>
#include <c10/util/Half.h>
#include <cstdint>
#include <cmath>

//------------------------------------------------------------------------
// Helpers.

// enum // Filter modes.
// {
//     MODE_SUSD = 0,  // Separable upsampling, separable downsampling.
//     MODE_FUSD = 1,  // Full upsampling, separable downsampling.
//     MODE_SUFD = 2,  // Separable upsampling, full downsampling.
//     MODE_FUFD = 3,  // Full upsampling, full downsampling.
// };

template <class T> struct InternalType;
template <> struct InternalType<double>
{
    typedef double scalar_t; typedef sycl::double2 vec2_t;
        typedef sycl::double4 vec4_t;
    __dpct_inline__ static vec2_t zero_vec2() { return sycl::double2(0, 0); }
    __dpct_inline__ static vec4_t zero_vec4() {
        return sycl::double4(0, 0, 0, 0);
    }
    __dpct_inline__ static double clamp(double x, double c) {
        return sycl::fmin(sycl::fmax(x, -c), c);
    }
};

template <> struct InternalType<float>
{
    typedef float scalar_t; typedef sycl::float2 vec2_t;
        typedef sycl::float4 vec4_t;
    __dpct_inline__ static vec2_t zero_vec2() { return sycl::float2(0, 0); }
    __dpct_inline__ static vec4_t zero_vec4() {
        return sycl::float4(0, 0, 0, 0);
    }
    __dpct_inline__ static float clamp(float x, float c) {
        return sycl::fmin(sycl::fmax(x, -c), c);
    }
};

template <> struct InternalType<c10::Half>
{
    typedef float scalar_t; typedef sycl::float2 vec2_t;
        typedef sycl::float4 vec4_t;
    __dpct_inline__ static vec2_t zero_vec2() { return sycl::float2(0, 0); }
    __dpct_inline__ static vec4_t zero_vec4() {
        return sycl::float4(0, 0, 0, 0);
    }
    __dpct_inline__ static float clamp(float x, float c) {
        return sycl::fmin(sycl::fmax(x, -c), c);
    }
};

#define MIN(A, B)       ((A) < (B) ? (A) : (B))
#define MAX(A, B)       ((A) > (B) ? (A) : (B))
#define CEIL_DIV(A, B) (((B)==1) ? (A) : \
                        ((B)==2) ? ((int)((A)+1) >> 1) : \
                        ((B)==4) ? ((int)((A)+3) >> 2) : \
                        (((A) + ((A) > 0 ? (B) - 1 : 0)) / (B)))

// This works only up to blocks of size 256 x 256 and for all N that are powers of two.
template <int N>
__dpct_inline__ void fast_div_mod(int &x, int &y, unsigned int i)
{
    if ((N & (N-1)) && N <= 256)
        y = (i * ((1<<24)/N + 1)) >> 24; // Assumes N <= 256, i < N*256.
    else
        y = i/N;

    x = i - y*N;
}

// Type cast stride before reading it.
template <class T> __dpct_inline__ T get_stride(const int64_t &x)
{
    return *reinterpret_cast<const T*>(&x);
}

//------------------------------------------------------------------------
// Filters, setup kernel, copying function.

#define MAX_FILTER_SIZE 32

// Combined up/down filter buffers so that transfer can be done with one copy.
// dpct::global_memory<float, 1> g_fbuf(
//     2 * MAX_FILTER_SIZE *
//     MAX_FILTER_SIZE); // Filters in global memory, written by setup kernel.
static dpct::global_memory<float, 1> g_fbuf(
    2 * MAX_FILTER_SIZE *
    MAX_FILTER_SIZE); // Filters in global memory, written by setup kernel.
static dpct::constant_memory<float, 1>
    c_fbuf(2 * MAX_FILTER_SIZE *
           MAX_FILTER_SIZE); // Filters in constant memory, read by main kernel.

// Accessors to combined buffers to index up/down filters individually.
#define c_fu (c_fbuf)
#define c_fd (c_fbuf + MAX_FILTER_SIZE * MAX_FILTER_SIZE)
#define g_fu (g_fbuf)
#define g_fd (g_fbuf + MAX_FILTER_SIZE * MAX_FILTER_SIZE)

// Set up filters into global memory buffer.
static void setup_filters_kernel(filtered_lrelu_kernel_params p,
                                 const sycl::nd_item<3> &item_ct1, float *g_fbuf)
{
    for (int idx = item_ct1.get_local_id(2);
         idx < MAX_FILTER_SIZE * MAX_FILTER_SIZE;
         idx += item_ct1.get_local_range(2))
    {
        int x, y;
        fast_div_mod<MAX_FILTER_SIZE>(x, y, idx);

        int fu_x = p.flip ? x : (p.fuShape.x() - 1 - x);
        int fu_y = p.flip ? y : (p.fuShape.y() - 1 - y);
        if (p.fuShape.y() > 0)
            g_fu[idx] =
                (x >= p.fuShape.x() || y >= p.fuShape.y())
                    ? 0.0f
                    : p.fu[fu_x * p.fuStride.x() + fu_y * p.fuStride.y()];
        else
            g_fu[idx] = (x >= p.fuShape.x() || y > 0)
                            ? 0.0f
                            : p.fu[fu_x * p.fuStride.x()];

        int fd_x = p.flip ? x : (p.fdShape.x() - 1 - x);
        int fd_y = p.flip ? y : (p.fdShape.y() - 1 - y);
        if (p.fdShape.y() > 0)
            g_fd[idx] =
                (x >= p.fdShape.x() || y >= p.fdShape.y())
                    ? 0.0f
                    : p.fd[fd_x * p.fdStride.x() + fd_y * p.fdStride.y()];
        else
            g_fd[idx] = (x >= p.fdShape.x() || y > 0)
                            ? 0.0f
                            : p.fd[fd_x * p.fdStride.x()];
    }
}

// Host function to copy filters written by setup kernel into constant buffer for main kernel.
template <bool, bool>
static dpct::err0 copy_filters(dpct::queue_ptr stream) try {
    void* src = 0;
    dpct::err0 err = DPCT_CHECK_ERROR(*(&src) = g_fbuf.get_ptr());
    /*
    DPCT1001:41: The statement could not be removed.
    */
    /*
    DPCT1000:42: Error handling if-stmt was detected but could not be rewritten.
    */
    if (err) return err;
    return DPCT_CHECK_ERROR(
        stream->memcpy(c_fbuf.get_ptr(*stream), src,
                       2 * MAX_FILTER_SIZE * MAX_FILTER_SIZE * sizeof(float)));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

//------------------------------------------------------------------------
// Coordinate spaces:
// - Relative to input tensor:      inX, inY, tileInX, tileInY
// - Relative to input tile:        relInX, relInY, tileInW, tileInH
// - Relative to upsampled tile:    relUpX, relUpY, tileUpW, tileUpH
// - Relative to output tile:       relOutX, relOutY, tileOutW, tileOutH
// - Relative to output tensor:     outX, outY, tileOutX, tileOutY
//
// Relationships between coordinate spaces:
// - inX = tileInX + relInX
// - inY = tileInY + relInY
// - relUpX = relInX * up + phaseInX
// - relUpY = relInY * up + phaseInY
// - relUpX = relOutX * down
// - relUpY = relOutY * down
// - outX = tileOutX + relOutX
// - outY = tileOutY + relOutY

 // When sharedKB <= 48, allocate shared memory statically inside the kernel,
 // otherwise use the externally allocated shared memory buffer.

// template <class T, class index_t, int sharedKB, bool signWrite, bool signRead,
//           int filterMode, int up, int fuSize, int down, int fdSize,
//           int tileOutW, int tileOutH, int threadsPerBlock, bool enableXrep,
//           bool enableWriteSkip>
template <class T, class index_t, bool signWrite, bool signRead,
          int filterMode, int up, int fuSize, int down, int fdSize,
          int tileOutW, int tileOutH, int threadsPerBlock, bool enableXrep,
          bool enableWriteSkip, class AccessorT
          >
/*
DPCT1110:0: The total declared local variable size in device function
filtered_lrelu_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
// static void filtered_lrelu_kernel(filtered_lrelu_kernel_params p,
//                                   const sycl::nd_item<3> &item_ct1,
//                                   float const *c_fbuf, char *s_buf_raw,
//                                   scalar_t *s_buf0_st)
static void filtered_lrelu_kernel(filtered_lrelu_kernel_params p,
                                  const sycl::nd_item<3> &item_ct1,
                                  float *c_fbuf,
                                  typename InternalType<T>::scalar_t *s_buf0_st,
                                  AccessorT yacc
                                  )
{
    // Check that we don't try to support non-existing filter modes.
    static_assert(up   == 1 || up   == 2 || up   == 4, "only up=1, up=2, up=4 scales supported");
    static_assert(down == 1 || down == 2 || down == 4, "only down=1, down=2, down=4 scales supported");
    static_assert(fuSize >= up,   "upsampling filter size must be at least upsampling factor");
    static_assert(fdSize >= down, "downsampling filter size must be at least downsampling factor");
    static_assert(fuSize % up   == 0, "upsampling filter size must be divisible with upsampling factor");
    static_assert(fdSize % down == 0, "downsampling filter size must be divisible with downsampling factor");
    static_assert(fuSize <= MAX_FILTER_SIZE && fdSize <= MAX_FILTER_SIZE, "filter size greater than MAX_FILTER_SIZE");
    static_assert(up   != 1 || (fuSize == 1 && (filterMode == MODE_FUFD || filterMode == MODE_FUSD)), "up=1 supported only for 1x1 full filters");
    static_assert(down != 1 || (fdSize == 1 && (filterMode == MODE_FUFD || filterMode == MODE_SUFD)), "down=1 supported only for 1x1 full filters");
    static_assert(!(up   == 4 && (filterMode == MODE_FUFD || filterMode == MODE_FUSD)), "full filters not supported for up=4");
    static_assert(!(down == 4 && (filterMode == MODE_FUFD || filterMode == MODE_SUFD)), "full filters not supported for down=4");

    // Static definitions.
    typedef typename InternalType<T>::scalar_t scalar_t;
    typedef typename InternalType<T>::vec2_t vec2_t;
    typedef typename InternalType<T>::vec4_t vec4_t;
    const int tileUpW    = (tileOutW * down + (fdSize - 1) - (down - 1) + 3) & ~3;  // Upsampled tile width, rounded up to multiple of 4.
    const int tileUpH    = tileOutH * down + (fdSize - 1) - (down - 1);             // Upsampled tile height.
    const int tileInW    = CEIL_DIV(tileUpW  + (fuSize - 1), up);                   // Input tile width.
    const int tileInH    = CEIL_DIV(tileUpH  + (fuSize - 1), up);                   // Input tile height.
    const int tileUpH_up = CEIL_DIV(tileUpH, up) * up;                              // Upsampled tile height rounded up to a multiple of up.
    const int tileInH_up = CEIL_DIV(tileUpH_up + (fuSize - 1), up);                 // For allocations only, to avoid shared memory read overruns with up=2 and up=4.

    // Merge 1x1 downsampling into last upsampling step for upf1 and ups2.
    const bool downInline = (down == 1) && ((up == 1 && filterMode == MODE_FUFD) || (up == 2 && filterMode == MODE_SUFD));

    // Sizes of logical buffers.
    const int szIn    = tileInH_up * tileInW;
    const int szUpX   = tileInH_up * tileUpW;
    const int szUpXY  = downInline ? 0 : (tileUpH * tileUpW);
    const int szDownX = tileUpH * tileOutW;

    // Sizes for shared memory arrays.
    const int s_buf0_size_base =
        (filterMode == MODE_SUSD) ? MAX(szIn, szUpXY) :
        (filterMode == MODE_FUSD) ? MAX(szIn, szDownX) :
        (filterMode == MODE_SUFD) ? MAX(szIn, szUpXY) :
        (filterMode == MODE_FUFD) ? szIn :
        -1;
    const int s_buf1_size_base =
        (filterMode == MODE_SUSD) ? MAX(szUpX, szDownX) :
        (filterMode == MODE_FUSD) ? szUpXY :
        (filterMode == MODE_SUFD) ? szUpX  :
        (filterMode == MODE_FUFD) ? szUpXY :
        -1;

    // Ensure U128 alignment.
    const int s_buf0_size = (s_buf0_size_base + 3) & ~3;
    const int s_buf1_size = (s_buf1_size_base + 3) & ~3;

    // Check at compile time that we don't use too much shared memory.
    // static_assert((s_buf0_size + s_buf1_size) * sizeof(scalar_t) <= (sharedKB << 10), "shared memory overflow");

    // Declare shared memory arrays.
    scalar_t* s_buf0;
    scalar_t* s_buf1;

    // if (sharedKB <= 48)
    // {
        // Allocate shared memory arrays here.
        // Prevent launching if this isn't optimized away when unused.
        s_buf0 = s_buf0_st;
        s_buf1 = s_buf0 + s_buf0_size;
    // }
    // else
    // {
    //     // Use the dynamically allocated shared memory array.
    //     s_buf0 = (scalar_t*)s_buf_raw;
    //     s_buf1 = s_buf0 + s_buf0_size;
    // }

    // Pointers to the buffers.
    scalar_t* s_tileIn;       // Input tile:                      [relInX * tileInH + relInY]
    scalar_t* s_tileUpX;      // After horizontal upsampling:     [relInY * tileUpW + relUpX]
    scalar_t* s_tileUpXY;     // After upsampling:                [relUpY * tileUpW + relUpX]
    scalar_t* s_tileDownX;    // After horizontal downsampling:   [relUpY * tileOutW + relOutX]
    if (filterMode == MODE_SUSD)
    {
        s_tileIn    = s_buf0;
        s_tileUpX   = s_buf1;
        s_tileUpXY  = s_buf0;
        s_tileDownX = s_buf1;
    }
    else if (filterMode == MODE_FUSD)
    {
        s_tileIn    = s_buf0;
        s_tileUpXY  = s_buf1;
        s_tileDownX = s_buf0;
    }
    else if (filterMode == MODE_SUFD)
    {
        s_tileIn    = s_buf0;
        s_tileUpX   = s_buf1;
        s_tileUpXY  = s_buf0;
    }
    else if (filterMode == MODE_FUFD)
    {
        s_tileIn    = s_buf0;
        s_tileUpXY  = s_buf1;
    }

    // Allow large grids in z direction via per-launch offset.
    int channelIdx = item_ct1.get_group(0) + p.blockZofs;
    int batchIdx = channelIdx / p.yShape.z();
    channelIdx -= batchIdx * p.yShape.z();

    // Offset to output feature map. In bytes.
    index_t mapOfsOut = channelIdx * get_stride<index_t>(p.yStride.z()) +
                        batchIdx * get_stride<index_t>(p.yStride.w());

    // Sign shift amount.
    uint32_t signXo = ((item_ct1.get_local_id(2) + p.sOfs.x()) << 1) & 6;

    // Inner tile loop.
    #pragma unroll 1
    for (int tileIdx = 0;
         !enableXrep ||
         (tileIdx <
          MIN(p.tilesXrep, p.tilesXdim - p.tilesXrep * item_ct1.get_group(1)));
         tileIdx++)
    {
        // Locate output tile.
        int tileX = enableXrep ? item_ct1.get_group(1) * p.tilesXrep + tileIdx
                               : item_ct1.get_group(2);
        int tileOutX = tileX * tileOutW;
        int tileOutY =
            (enableXrep ? item_ct1.get_group(2) : item_ct1.get_group(1)) *
            tileOutH;

        // Locate input tile.
        int tmpX = tileOutX * down - p.pad0.x();
        int tmpY = tileOutY * down - p.pad0.y();
        int tileInX = CEIL_DIV(tmpX, up);
        int tileInY = CEIL_DIV(tmpY, up);
        const int phaseInX = tileInX * up - tmpX;
        const int phaseInY = tileInY * up - tmpY;

        // Extra sync if input and output buffers are the same and we are not on first tile.
        if (enableXrep && tileIdx > 0 && (filterMode == MODE_FUSD || (filterMode == MODE_SUFD && !downInline) || (filterMode == MODE_FUFD && downInline)))
            /*
            DPCT1118:1: SYCL group functions and algorithms must be encountered
            in converged control flow. You may need to adjust the code.
            */
            /*
            DPCT1065:43: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier(sycl::access::fence_space::local_space);

        // Load input tile & apply bias. Unrolled.
        scalar_t b = (scalar_t)*(const T*)((const char*)p.b + (channelIdx * get_stride<index_t>(p.bStride)));
        index_t mapOfsIn = channelIdx * get_stride<index_t>(p.xStride.z()) +
                           batchIdx * get_stride<index_t>(p.xStride.w());
        int idx = item_ct1.get_local_id(2);
        const int loopCountIN = CEIL_DIV(tileInW * tileInH, threadsPerBlock);
        #pragma unroll
        for (int loop = 0; loop < loopCountIN; loop++)
        {
            int relInX, relInY;
            fast_div_mod<tileInW>(relInX, relInY, idx);
            int inX = tileInX + relInX;
            int inY = tileInY + relInY;
            scalar_t v = 0;

            if ((uint32_t)inX < p.xShape.x() && (uint32_t)inY < p.xShape.y())
                v = (scalar_t) *
                        ((const T *)((const char *)p.x +
                                     (inX * get_stride<index_t>(p.xStride.x()) +
                                      inY * get_stride<index_t>(p.xStride.y()) +
                                      mapOfsIn))) +
                    b;

            bool skip = (loop == loopCountIN-1) && (idx >= tileInW * tileInH);
            if (!skip)
                s_tileIn[idx] = v;

            idx += threadsPerBlock;
        }

        if (filterMode == MODE_SUSD || filterMode == MODE_SUFD) // Separable upsampling filter.
        {
            // Horizontal upsampling.
            /*
            DPCT1118:2: SYCL group functions and algorithms must be encountered
            in converged control flow. You may need to adjust the code.
            */
            /*
            DPCT1065:44: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier(sycl::access::fence_space::local_space);
            if (up == 4)
            {
                for (int idx = item_ct1.get_local_id(2) * up;
                     idx < tileUpW * tileInH;
                     idx += item_ct1.get_local_range(2) * up)
                {
                    int relUpX0, relInY;
                    fast_div_mod<tileUpW>(relUpX0, relInY, idx);
                    int relInX0 = relUpX0 / up;
                    int src0 = relInX0 + tileInW * relInY;
                    int dst = relInY * tileUpW + relUpX0;
                    vec4_t v = InternalType<T>::zero_vec4();
                    scalar_t a = s_tileIn[src0];
                    if (phaseInX == 0)
                    {
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                        {
                            v.x() += a * (scalar_t)c_fu[step * up + 0];
                            a = s_tileIn[src0 + step + 1];
                            v.y() += a * (scalar_t)c_fu[step * up + 3];
                            v.z() += a * (scalar_t)c_fu[step * up + 2];
                            v.w() += a * (scalar_t)c_fu[step * up + 1];
                        }
                    }
                    else if (phaseInX == 1)
                    {
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                        {
                            v.x() += a * (scalar_t)c_fu[step * up + 1];
                            v.y() += a * (scalar_t)c_fu[step * up + 0];
                            a = s_tileIn[src0 + step + 1];
                            v.z() += a * (scalar_t)c_fu[step * up + 3];
                            v.w() += a * (scalar_t)c_fu[step * up + 2];
                        }
                    }
                    else if (phaseInX == 2)
                    {
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                        {
                            v.x() += a * (scalar_t)c_fu[step * up + 2];
                            v.y() += a * (scalar_t)c_fu[step * up + 1];
                            v.z() += a * (scalar_t)c_fu[step * up + 0];
                            a = s_tileIn[src0 + step + 1];
                            v.w() += a * (scalar_t)c_fu[step * up + 3];
                        }
                    }
                    else // (phaseInX == 3)
                    {
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                        {
                            v.x() += a * (scalar_t)c_fu[step * up + 3];
                            v.y() += a * (scalar_t)c_fu[step * up + 2];
                            v.z() += a * (scalar_t)c_fu[step * up + 1];
                            v.w() += a * (scalar_t)c_fu[step * up + 0];
                            a = s_tileIn[src0 + step + 1];
                        }
                    }
                    s_tileUpX[dst + 0] = v.x();
                    s_tileUpX[dst + 1] = v.y();
                    s_tileUpX[dst + 2] = v.z();
                    s_tileUpX[dst + 3] = v.w();
                }
            }
            else if (up == 2)
            {
                bool p0 = (phaseInX == 0);
                for (int idx = item_ct1.get_local_id(2) * up;
                     idx < tileUpW * tileInH;
                     idx += item_ct1.get_local_range(2) * up)
                {
                    int relUpX0, relInY;
                    fast_div_mod<tileUpW>(relUpX0, relInY, idx);
                    int relInX0 = relUpX0 / up;
                    int src0 = relInX0 + tileInW * relInY;
                    int dst = relInY * tileUpW + relUpX0;
                    vec2_t v = InternalType<T>::zero_vec2();
                    scalar_t a = s_tileIn[src0];
                    if (p0) // (phaseInX == 0)
                    {
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                        {
                            v.x() += a * (scalar_t)c_fu[step * up + 0];
                            a = s_tileIn[src0 + step + 1];
                            v.y() += a * (scalar_t)c_fu[step * up + 1];
                        }
                    }
                    else // (phaseInX == 1)
                    {
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                        {
                            v.x() += a * (scalar_t)c_fu[step * up + 1];
                            v.y() += a * (scalar_t)c_fu[step * up + 0];
                            a = s_tileIn[src0 + step + 1];
                        }
                    }
                    s_tileUpX[dst + 0] = v.x();
                    s_tileUpX[dst + 1] = v.y();
                }
            }

            // Vertical upsampling & nonlinearity.

            /*
            DPCT1118:3: SYCL group functions and algorithms must be encountered
            in converged control flow. You may need to adjust the code.
            */
            /*
            DPCT1065:45: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier(sycl::access::fence_space::local_space);
            int groupMask = 15 << ((item_ct1.get_local_id(2) & 31) & ~3);
            int minY = tileOutY ? (tileOutY - tileOutH) * down + tileUpH : 0; // Skip already written signs.
            int sShapeMaxY = MIN(p.sShape.y(),
                                 tileOutY * down +
                                     tileUpH); // Avoid out-of-tile sign writes.
            if (up == 4)
            {
                minY -= 3; // Adjust according to block height.
                for (int idx = item_ct1.get_local_id(2);
                     idx < tileUpW * tileUpH_up / up;
                     idx += item_ct1.get_local_range(2))
                {
                    int relUpX, relInY0;
                    fast_div_mod<tileUpW>(relUpX, relInY0, idx);
                    int relUpY0 = relInY0 * up;
                    int src0 = relInY0 * tileUpW + relUpX;
                    int dst = relUpY0 * tileUpW + relUpX;
                    vec4_t v = InternalType<T>::zero_vec4();

                    scalar_t a = s_tileUpX[src0];
                    if (phaseInY == 0)
                    {
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                        {
                            v.x() += a * (scalar_t)c_fu[step * up + 0];
                            a = s_tileUpX[src0 + (step + 1) * tileUpW];
                            v.y() += a * (scalar_t)c_fu[step * up + 3];
                            v.z() += a * (scalar_t)c_fu[step * up + 2];
                            v.w() += a * (scalar_t)c_fu[step * up + 1];
                        }
                    }
                    else if (phaseInY == 1)
                    {
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                        {
                            v.x() += a * (scalar_t)c_fu[step * up + 1];
                            v.y() += a * (scalar_t)c_fu[step * up + 0];
                            a = s_tileUpX[src0 + (step + 1) * tileUpW];
                            v.z() += a * (scalar_t)c_fu[step * up + 3];
                            v.w() += a * (scalar_t)c_fu[step * up + 2];
                        }
                    }
                    else if (phaseInY == 2)
                    {
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                        {
                            v.x() += a * (scalar_t)c_fu[step * up + 2];
                            v.y() += a * (scalar_t)c_fu[step * up + 1];
                            v.z() += a * (scalar_t)c_fu[step * up + 0];
                            a = s_tileUpX[src0 + (step + 1) * tileUpW];
                            v.w() += a * (scalar_t)c_fu[step * up + 3];
                        }
                    }
                    else // (phaseInY == 3)
                    {
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                        {
                            v.x() += a * (scalar_t)c_fu[step * up + 3];
                            v.y() += a * (scalar_t)c_fu[step * up + 2];
                            v.z() += a * (scalar_t)c_fu[step * up + 1];
                            v.w() += a * (scalar_t)c_fu[step * up + 0];
                            a = s_tileUpX[src0 + (step + 1) * tileUpW];
                        }
                    }

                    int x = tileOutX * down + relUpX;
                    int y = tileOutY * down + relUpY0;
                    int signX = x + p.sOfs.x();
                    int signY = y + p.sOfs.y();
                    int signZ = item_ct1.get_group(0) + p.blockZofs;
                    int signXb = signX >> 2;
                    index_t si0 =
                        signXb +
                        p.sShape.x() * (signY + (index_t)p.sShape.y() * signZ);
                    index_t si1 = si0 + p.sShape.x();
                    index_t si2 = si0 + p.sShape.x() * 2;
                    index_t si3 = si0 + p.sShape.x() * 3;

                    v.x() *= (scalar_t)((float)up * (float)up * p.gain);
                    v.y() *= (scalar_t)((float)up * (float)up * p.gain);
                    v.z() *= (scalar_t)((float)up * (float)up * p.gain);
                    v.w() *= (scalar_t)((float)up * (float)up * p.gain);

                    if (signWrite)
                    {
                        if (!enableWriteSkip)
                        {
                            // Determine and write signs.
                            int sx =
                                sycl::bit_cast<unsigned int>(v.x()) >> 31 << 0;
                            int sy =
                                sycl::bit_cast<unsigned int>(v.y()) >> 31 << 8;
                            int sz =
                                sycl::bit_cast<unsigned int>(v.z()) >> 31 << 16;
                            int sw =
                                sycl::bit_cast<unsigned int>(v.w()) >> 31 << 24;
                            if (sx) v.x() *= p.slope;
                            if (sy) v.y() *= p.slope;
                            if (sz) v.z() *= p.slope;
                            if (sw) v.w() *= p.slope;
                            if (sycl::fabs(v.x()) > p.clamp) {
                                sx = 2 << 0;
                                v.x() = InternalType<T>::clamp(v.x(), p.clamp);
                            }
                            if (sycl::fabs(v.y()) > p.clamp) {
                                sy = 2 << 8;
                                v.y() = InternalType<T>::clamp(v.y(), p.clamp);
                            }
                            if (sycl::fabs(v.z()) > p.clamp) {
                                sz = 2 << 16;
                                v.z() = InternalType<T>::clamp(v.z(), p.clamp);
                            }
                            if (sycl::fabs(v.w()) > p.clamp) {
                                sw = 2 << 24;
                                v.w() = InternalType<T>::clamp(v.w(), p.clamp);
                            }

                            if ((uint32_t)signXb < p.swLimit && signY >= minY)
                            {
                                // Combine signs.
                                uint32_t s = sx + sy + sw + sz;
                                s <<= (signX & 3) << 1;
                                /*
                                DPCT1108:4: '__shfl_xor_sync' was migrated with
                                the experimental feature masked sub_group
                                function which may not be supported by all
                                compilers or runtimes. You may need to adjust
                                the code.
                                */
                                s |= dpct::experimental::
                                    permute_sub_group_by_xor(
                                        groupMask, item_ct1.get_sub_group(), s,
                                        1);
                                /*
                                DPCT1108:5: '__shfl_xor_sync' was migrated with
                                the experimental feature masked sub_group
                                function which may not be supported by all
                                compilers or runtimes. You may need to adjust
                                the code.
                                */
                                s |= dpct::experimental::
                                    permute_sub_group_by_xor(
                                        groupMask, item_ct1.get_sub_group(), s,
                                        2);

                                // Write signs.
                                if ((uint32_t)(signY + 0) < sShapeMaxY) { p.s[si0] = (unsigned char)(s >>  0); }
                                if ((uint32_t)(signY + 1) < sShapeMaxY) { p.s[si1] = (unsigned char)(s >>  8); }
                                if ((uint32_t)(signY + 2) < sShapeMaxY) { p.s[si2] = (unsigned char)(s >> 16); }
                                if ((uint32_t)(signY + 3) < sShapeMaxY) { p.s[si3] = (unsigned char)(s >> 24); }
                            }
                        }
                        else
                        {
                            // Determine and write signs.
                            if ((uint32_t)signXb < p.swLimit && signY >= minY)
                            {
                                int sx = sycl::bit_cast<unsigned int>(v.x()) >>
                                         31 << 0;
                                int sy = sycl::bit_cast<unsigned int>(v.y()) >>
                                         31 << 8;
                                int sz = sycl::bit_cast<unsigned int>(v.z()) >>
                                         31 << 16;
                                int sw = sycl::bit_cast<unsigned int>(v.w()) >>
                                         31 << 24;
                                if (sx) v.x() *= p.slope;
                                if (sy) v.y() *= p.slope;
                                if (sz) v.z() *= p.slope;
                                if (sw) v.w() *= p.slope;
                                if (sycl::fabs(v.x()) > p.clamp) {
                                    sx = 2 << 0; v.x() = InternalType<T>::clamp(
                                                     v.x(), p.clamp);
                                }
                                if (sycl::fabs(v.y()) > p.clamp) {
                                    sy = 2 << 8; v.y() = InternalType<T>::clamp(
                                                     v.y(), p.clamp);
                                }
                                if (sycl::fabs(v.z()) > p.clamp) {
                                    sz = 2 << 16;
                                    v.z() =
                                        InternalType<T>::clamp(v.z(), p.clamp);
                                }
                                if (sycl::fabs(v.w()) > p.clamp) {
                                    sw = 2 << 24;
                                    v.w() =
                                        InternalType<T>::clamp(v.w(), p.clamp);
                                }

                                // Combine signs.
                                uint32_t s = sx + sy + sw + sz;
                                s <<= (signX & 3) << 1;
                                /*
                                DPCT1108:6: '__shfl_xor_sync' was migrated with
                                the experimental feature masked sub_group
                                function which may not be supported by all
                                compilers or runtimes. You may need to adjust
                                the code.
                                */
                                s |= dpct::experimental::
                                    permute_sub_group_by_xor(
                                        groupMask, item_ct1.get_sub_group(), s,
                                        1);
                                /*
                                DPCT1108:7: '__shfl_xor_sync' was migrated with
                                the experimental feature masked sub_group
                                function which may not be supported by all
                                compilers or runtimes. You may need to adjust
                                the code.
                                */
                                s |= dpct::experimental::
                                    permute_sub_group_by_xor(
                                        groupMask, item_ct1.get_sub_group(), s,
                                        2);

                                // Write signs.
                                if ((uint32_t)(signY + 0) < sShapeMaxY) { p.s[si0] = (unsigned char)(s >>  0); }
                                if ((uint32_t)(signY + 1) < sShapeMaxY) { p.s[si1] = (unsigned char)(s >>  8); }
                                if ((uint32_t)(signY + 2) < sShapeMaxY) { p.s[si2] = (unsigned char)(s >> 16); }
                                if ((uint32_t)(signY + 3) < sShapeMaxY) { p.s[si3] = (unsigned char)(s >> 24); }
                            }
                            else
                            {
                                // Just compute the values.
                                if (v.x() < 0.f) v.x() *= p.slope;
                                    v.x() =
                                        InternalType<T>::clamp(v.x(), p.clamp);
                                if (v.y() < 0.f) v.y() *= p.slope;
                                    v.y() =
                                        InternalType<T>::clamp(v.y(), p.clamp);
                                if (v.z() < 0.f) v.z() *= p.slope;
                                    v.z() =
                                        InternalType<T>::clamp(v.z(), p.clamp);
                                if (v.w() < 0.f) v.w() *= p.slope;
                                    v.w() =
                                        InternalType<T>::clamp(v.w(), p.clamp);
                            }
                        }
                    }
                    else if (signRead) // Read signs and apply.
                    {
                        if ((uint32_t)signXb < p.swLimit)
                        {
                            int ss = (signX & 3) << 1;
                            if ((uint32_t)(signY + 0) < p.sShape.y()) {
                                int s = p.s[si0] >> ss;
                                if (s & 1) v.x() *= p.slope;
                                if (s & 2) v.x() = 0.f;
                            }
                            if ((uint32_t)(signY + 1) < p.sShape.y()) {
                                int s = p.s[si1] >> ss;
                                if (s & 1) v.y() *= p.slope;
                                if (s & 2) v.y() = 0.f;
                            }
                            if ((uint32_t)(signY + 2) < p.sShape.y()) {
                                int s = p.s[si2] >> ss;
                                if (s & 1) v.z() *= p.slope;
                                if (s & 2) v.z() = 0.f;
                            }
                            if ((uint32_t)(signY + 3) < p.sShape.y()) {
                                int s = p.s[si3] >> ss;
                                if (s & 1) v.w() *= p.slope;
                                if (s & 2) v.w() = 0.f;
                            }
                        }
                    }
                    else // Forward pass with no sign write.
                    {
                        if (v.x() < 0.f) v.x() *= p.slope;
                            v.x() = InternalType<T>::clamp(v.x(), p.clamp);
                        if (v.y() < 0.f) v.y() *= p.slope;
                            v.y() = InternalType<T>::clamp(v.y(), p.clamp);
                        if (v.z() < 0.f) v.z() *= p.slope;
                            v.z() = InternalType<T>::clamp(v.z(), p.clamp);
                        if (v.w() < 0.f) v.w() *= p.slope;
                            v.w() = InternalType<T>::clamp(v.w(), p.clamp);
                    }

                    s_tileUpXY[dst + 0 * tileUpW] = v.x();
                    if (relUpY0 + 1 < tileUpH) s_tileUpXY[dst + 1 * tileUpW] =
                        v.y();
                    if (relUpY0 + 2 < tileUpH) s_tileUpXY[dst + 2 * tileUpW] =
                        v.z();
                    if (relUpY0 + 3 < tileUpH) s_tileUpXY[dst + 3 * tileUpW] =
                        v.w();
                }
            }
            else if (up == 2)
            {
                minY -= 1; // Adjust according to block height.
                for (int idx = item_ct1.get_local_id(2);
                     idx < tileUpW * tileUpH_up / up;
                     idx += item_ct1.get_local_range(2))
                {
                    int relUpX, relInY0;
                    fast_div_mod<tileUpW>(relUpX, relInY0, idx);
                    int relUpY0 = relInY0 * up;
                    int src0 = relInY0 * tileUpW + relUpX;
                    int dst = relUpY0 * tileUpW + relUpX;
                    vec2_t v = InternalType<T>::zero_vec2();

                    scalar_t a = s_tileUpX[src0];
                    if (phaseInY == 0)
                    {
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                        {
                            v.x() += a * (scalar_t)c_fu[step * up + 0];
                            a = s_tileUpX[src0 + (step + 1) * tileUpW];
                            v.y() += a * (scalar_t)c_fu[step * up + 1];
                        }
                    }
                    else // (phaseInY == 1)
                    {
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                        {
                            v.x() += a * (scalar_t)c_fu[step * up + 1];
                            v.y() += a * (scalar_t)c_fu[step * up + 0];
                            a = s_tileUpX[src0 + (step + 1) * tileUpW];
                        }
                    }

                    int x = tileOutX * down + relUpX;
                    int y = tileOutY * down + relUpY0;
                    int signX = x + p.sOfs.x();
                    int signY = y + p.sOfs.y();
                    int signZ = item_ct1.get_group(0) + p.blockZofs;
                    int signXb = signX >> 2;
                    index_t si0 =
                        signXb +
                        p.sShape.x() * (signY + (index_t)p.sShape.y() * signZ);
                    index_t si1 = si0 + p.sShape.x();

                    v.x() *= (scalar_t)((float)up * (float)up * p.gain);
                    v.y() *= (scalar_t)((float)up * (float)up * p.gain);

                    if (signWrite)
                    {
                        if (!enableWriteSkip)
                        {
                            // Determine and write signs.
                            int sx =
                                sycl::bit_cast<unsigned int>(v.x()) >> 31 << 0;
                            int sy =
                                sycl::bit_cast<unsigned int>(v.y()) >> 31 << 8;
                            if (sx) v.x() *= p.slope;
                            if (sy) v.y() *= p.slope;
                            if (sycl::fabs(v.x()) > p.clamp) {
                                sx = 2 << 0;
                                v.x() = InternalType<T>::clamp(v.x(), p.clamp);
                            }
                            if (sycl::fabs(v.y()) > p.clamp) {
                                sy = 2 << 8;
                                v.y() = InternalType<T>::clamp(v.y(), p.clamp);
                            }

                            if ((uint32_t)signXb < p.swLimit && signY >= minY)
                            {
                                // Combine signs.
                                int s = sx + sy;
                                s <<= signXo;
                                /*
                                DPCT1108:8: '__shfl_xor_sync' was migrated with
                                the experimental feature masked sub_group
                                function which may not be supported by all
                                compilers or runtimes. You may need to adjust
                                the code.
                                */
                                s |= dpct::experimental::
                                    permute_sub_group_by_xor(
                                        groupMask, item_ct1.get_sub_group(), s,
                                        1);
                                /*
                                DPCT1108:9: '__shfl_xor_sync' was migrated with
                                the experimental feature masked sub_group
                                function which may not be supported by all
                                compilers or runtimes. You may need to adjust
                                the code.
                                */
                                s |= dpct::experimental::
                                    permute_sub_group_by_xor(
                                        groupMask, item_ct1.get_sub_group(), s,
                                        2);

                                // Write signs.
                                if ((uint32_t)(signY + 0) < sShapeMaxY) { p.s[si0] = (unsigned char)(s >>  0); }
                                if ((uint32_t)(signY + 1) < sShapeMaxY) { p.s[si1] = (unsigned char)(s >>  8); }
                            }
                        }
                        else
                        {
                            // Determine and write signs.
                            if ((uint32_t)signXb < p.swLimit && signY >= minY)
                            {
                                int sx = sycl::bit_cast<unsigned int>(v.x()) >>
                                         31 << 0;
                                int sy = sycl::bit_cast<unsigned int>(v.y()) >>
                                         31 << 8;
                                if (sx) v.x() *= p.slope;
                                if (sy) v.y() *= p.slope;
                                if (sycl::fabs(v.x()) > p.clamp) {
                                    sx = 2 << 0; v.x() = InternalType<T>::clamp(
                                                     v.x(), p.clamp);
                                }
                                if (sycl::fabs(v.y()) > p.clamp) {
                                    sy = 2 << 8; v.y() = InternalType<T>::clamp(
                                                     v.y(), p.clamp);
                                }

                                // Combine signs.
                                int s = sx + sy;
                                s <<= signXo;
                                /*
                                DPCT1108:10: '__shfl_xor_sync' was migrated with
                                the experimental feature masked sub_group
                                function which may not be supported by all
                                compilers or runtimes. You may need to adjust
                                the code.
                                */
                                s |= dpct::experimental::
                                    permute_sub_group_by_xor(
                                        groupMask, item_ct1.get_sub_group(), s,
                                        1);
                                /*
                                DPCT1108:11: '__shfl_xor_sync' was migrated with
                                the experimental feature masked sub_group
                                function which may not be supported by all
                                compilers or runtimes. You may need to adjust
                                the code.
                                */
                                s |= dpct::experimental::
                                    permute_sub_group_by_xor(
                                        groupMask, item_ct1.get_sub_group(), s,
                                        2);

                                // Write signs.
                                if ((uint32_t)(signY + 0) < sShapeMaxY) { p.s[si0] = (unsigned char)(s >>  0); }
                                if ((uint32_t)(signY + 1) < sShapeMaxY) { p.s[si1] = (unsigned char)(s >>  8); }
                            }
                            else
                            {
                                // Just compute the values.
                                if (v.x() < 0.f) v.x() *= p.slope;
                                    v.x() =
                                        InternalType<T>::clamp(v.x(), p.clamp);
                                if (v.y() < 0.f) v.y() *= p.slope;
                                    v.y() =
                                        InternalType<T>::clamp(v.y(), p.clamp);
                            }
                        }
                    }
                    else if (signRead) // Read signs and apply.
                    {
                        if ((uint32_t)signXb < p.swLimit)
                        {
                            if ((uint32_t)(signY + 0) < p.sShape.y()) {
                                int s = p.s[si0] >> signXo;
                                if (s & 1) v.x() *= p.slope;
                                if (s & 2) v.x() = 0.f;
                            }
                            if ((uint32_t)(signY + 1) < p.sShape.y()) {
                                int s = p.s[si1] >> signXo;
                                if (s & 1) v.y() *= p.slope;
                                if (s & 2) v.y() = 0.f;
                            }
                        }
                    }
                    else // Forward pass with no sign write.
                    {
                        if (v.x() < 0.f) v.x() *= p.slope;
                            v.x() = InternalType<T>::clamp(v.x(), p.clamp);
                        if (v.y() < 0.f) v.y() *= p.slope;
                            v.y() = InternalType<T>::clamp(v.y(), p.clamp);
                    }

                    if (!downInline)
                    {
                        // Write into temporary buffer.
                        s_tileUpXY[dst] = v.x();
                        if (relUpY0 < tileUpH - 1)
                            s_tileUpXY[dst + tileUpW] = v.y();
                    }
                    else
                    {
                        // Write directly into output buffer.
                        if ((uint32_t)x < p.yShape.x())
                        {
                            int ymax =
                                MIN(p.yShape.y(), tileUpH + tileOutY * down);
                            index_t ofs =
                                x * get_stride<index_t>(p.yStride.x()) +
                                y * get_stride<index_t>(p.yStride.y()) +
                                mapOfsOut;
                            // if ((uint32_t)y + 0 < p.yShape.y()) *
                            //     ((T *)((char *)p.y + ofs)) =
                            //     (T)(v.x() * (scalar_t)c_fd[0]);
                            // if ((uint32_t)y + 1 < ymax) *
                            //     ((T *)((char *)p.y + ofs +
                            //            get_stride<index_t>(p.yStride.y()))) =
                            //     (T)(v.y() * (scalar_t)c_fd[0]);
                            if ((uint32_t)y + 0 < p.yShape.y())
                                (T&)(yacc[ofs]) = (T)(v.x() * (scalar_t)c_fd[0]);
                            if ((uint32_t)y + 1 < ymax)
                                (T&)(yacc[ofs + get_stride<index_t>(p.yStride.y())]) = (T)(v.y() * (scalar_t)c_fd[0]);
                        }
                    }
                }
            }
        }
        else if (filterMode == MODE_FUSD || filterMode == MODE_FUFD)
        {
            // Full upsampling filter.

            if (up == 2)
            {
                // 2 x 2-wide.
                /*
                DPCT1118:12: SYCL group functions and algorithms must be
                encountered in converged control flow. You may need to adjust
                the code.
                */
                /*
                DPCT1065:46: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier(sycl::access::fence_space::local_space);
                int minY = tileOutY ? (tileOutY - tileOutH) * down + tileUpH +
                                          p.sOfs.y()
                                    : 0; // Skip already written signs.
                for (int idx = item_ct1.get_local_id(2) * 4;
                     idx < tileUpW * tileUpH;
                     idx += item_ct1.get_local_range(2) * 4)
                {
                    int relUpX0, relUpY0;
                    fast_div_mod<tileUpW>(relUpX0, relUpY0, idx);
                    int relInX0 = CEIL_DIV(relUpX0 - phaseInX, up);
                    int relInY0 = CEIL_DIV(relUpY0 - phaseInY, up);
                    int src0 = relInX0 + tileInW * relInY0;
                    int tap0y = (relInY0 * up + phaseInY - relUpY0);

#define X_LOOP(TAPY, PX)                                                       \
    for (int sx = 0; sx < fuSize / up; sx++)                                   \
    {                                                                          \
        v.x() += a * (scalar_t)c_fu[(sx * up + (((PX) - 0) & (up - 1))) +      \
                                    (sy * up + (TAPY)) * MAX_FILTER_SIZE];     \
        v.z() += b * (scalar_t)c_fu[(sx * up + (((PX) - 0) & (up - 1))) +      \
                                    (sy * up + (TAPY)) * MAX_FILTER_SIZE];     \
            if ((PX) == 0) {                                                   \
                a = b; b = s_tileIn[src0 + 2 + sx + sy * tileInW];             \
            }                                                                  \
        v.y() += a * (scalar_t)c_fu[(sx * up + (((PX) - 1) & (up - 1))) +      \
                                    (sy * up + (TAPY)) * MAX_FILTER_SIZE];     \
        v.w() += b * (scalar_t)c_fu[(sx * up + (((PX) - 1) & (up - 1))) +      \
                                    (sy * up + (TAPY)) * MAX_FILTER_SIZE];     \
            if ((PX) == 1) {                                                   \
                a = b; b = s_tileIn[src0 + 2 + sx + sy * tileInW];             \
            }                                                                  \
    }

                    vec4_t v = InternalType<T>::zero_vec4();
                    if (tap0y == 0 && phaseInX == 0)
                        #pragma unroll
                        for (int sy = 0; sy < fuSize / up; sy++) { scalar_t a = s_tileIn[src0 + sy * tileInW]; scalar_t b = s_tileIn[src0 + sy * tileInW + 1];
                            #pragma unroll
                            X_LOOP(0, 0) }
                    if (tap0y == 0 && phaseInX == 1)
                        #pragma unroll
                        for (int sy = 0; sy < fuSize / up; sy++) { scalar_t a = s_tileIn[src0 + sy * tileInW]; scalar_t b = s_tileIn[src0 + sy * tileInW + 1];
                            #pragma unroll
                            X_LOOP(0, 1) }
                    if (tap0y == 1 && phaseInX == 0)
                        #pragma unroll
                        for (int sy = 0; sy < fuSize / up; sy++) { scalar_t a = s_tileIn[src0 + sy * tileInW]; scalar_t b = s_tileIn[src0 + sy * tileInW + 1];
                            #pragma unroll
                            X_LOOP(1, 0) }
                    if (tap0y == 1 && phaseInX == 1)
                        #pragma unroll
                        for (int sy = 0; sy < fuSize / up; sy++) { scalar_t a = s_tileIn[src0 + sy * tileInW]; scalar_t b = s_tileIn[src0 + sy * tileInW + 1];
                            #pragma unroll
                            X_LOOP(1, 1) }

                    #undef X_LOOP

                    int x = tileOutX * down + relUpX0;
                    int y = tileOutY * down + relUpY0;
                    int signX = x + p.sOfs.x();
                    int signY = y + p.sOfs.y();
                    int signZ = item_ct1.get_group(0) + p.blockZofs;
                    int signXb = signX >> 2;
                    index_t si =
                        signXb +
                        p.sShape.x() * (signY + (index_t)p.sShape.y() * signZ);

                    v.x() *= (scalar_t)((float)up * (float)up * p.gain);
                    v.y() *= (scalar_t)((float)up * (float)up * p.gain);
                    v.z() *= (scalar_t)((float)up * (float)up * p.gain);
                    v.w() *= (scalar_t)((float)up * (float)up * p.gain);

                    if (signWrite)
                    {
                        if (!enableWriteSkip)
                        {
                            // Determine and write signs.
                            int sx = sycl::bit_cast<unsigned int>(v.x()) >> 31;
                            int sy = sycl::bit_cast<unsigned int>(v.y()) >> 31;
                            int sz = sycl::bit_cast<unsigned int>(v.z()) >> 31;
                            int sw = sycl::bit_cast<unsigned int>(v.w()) >> 31;
                            if (sx) v.x() *= p.slope; if (sycl::fabs(v.x()) >
                                                          p.clamp) {
                                sx = 2;
                                v.x() = InternalType<T>::clamp(v.x(), p.clamp);
                            }
                            if (sy) v.y() *= p.slope; if (sycl::fabs(v.y()) >
                                                          p.clamp) {
                                sy = 2;
                                v.y() = InternalType<T>::clamp(v.y(), p.clamp);
                            }
                            if (sz) v.z() *= p.slope; if (sycl::fabs(v.z()) >
                                                          p.clamp) {
                                sz = 2;
                                v.z() = InternalType<T>::clamp(v.z(), p.clamp);
                            }
                            if (sw) v.w() *= p.slope; if (sycl::fabs(v.w()) >
                                                          p.clamp) {
                                sw = 2;
                                v.w() = InternalType<T>::clamp(v.w(), p.clamp);
                            }

                            if ((uint32_t)signXb < p.swLimit &&
                                (uint32_t)signY < p.sShape.y() && signY >= minY)
                            {
                                p.s[si] = sx + (sy << 2) + (sz << 4) + (sw << 6);
                            }
                        }
                        else
                        {
                            // Determine and write signs.
                            if ((uint32_t)signXb < p.swLimit &&
                                (uint32_t)signY < p.sShape.y() && signY >= minY)
                            {
                                int sx =
                                    sycl::bit_cast<unsigned int>(v.x()) >> 31;
                                int sy =
                                    sycl::bit_cast<unsigned int>(v.y()) >> 31;
                                int sz =
                                    sycl::bit_cast<unsigned int>(v.z()) >> 31;
                                int sw =
                                    sycl::bit_cast<unsigned int>(v.w()) >> 31;
                                if (sx) v.x() *= p.slope;
                                    if (sycl::fabs(v.x()) > p.clamp) {
                                        sx = 2; v.x() = InternalType<T>::clamp(
                                                    v.x(), p.clamp);
                                    }
                                if (sy) v.y() *= p.slope;
                                    if (sycl::fabs(v.y()) > p.clamp) {
                                        sy = 2; v.y() = InternalType<T>::clamp(
                                                    v.y(), p.clamp);
                                    }
                                if (sz) v.z() *= p.slope;
                                    if (sycl::fabs(v.z()) > p.clamp) {
                                        sz = 2; v.z() = InternalType<T>::clamp(
                                                    v.z(), p.clamp);
                                    }
                                if (sw) v.w() *= p.slope;
                                    if (sycl::fabs(v.w()) > p.clamp) {
                                        sw = 2; v.w() = InternalType<T>::clamp(
                                                    v.w(), p.clamp);
                                    }

                                p.s[si] = sx + (sy << 2) + (sz << 4) + (sw << 6);
                            }
                            else
                            {
                                // Just compute the values.
                                if (v.x() < 0.f) v.x() *= p.slope;
                                    v.x() =
                                        InternalType<T>::clamp(v.x(), p.clamp);
                                if (v.y() < 0.f) v.y() *= p.slope;
                                    v.y() =
                                        InternalType<T>::clamp(v.y(), p.clamp);
                                if (v.z() < 0.f) v.z() *= p.slope;
                                    v.z() =
                                        InternalType<T>::clamp(v.z(), p.clamp);
                                if (v.w() < 0.f) v.w() *= p.slope;
                                    v.w() =
                                        InternalType<T>::clamp(v.w(), p.clamp);
                            }
                        }
                    }
                    else if (signRead) // Read sign and apply.
                    {
                        if ((uint32_t)signY < p.sShape.y())
                        {
                            int s = 0;
                            if ((uint32_t)signXb     < p.swLimit) s  = p.s[si];
                            if ((uint32_t)signXb + 1 < p.swLimit) s |= p.s[si + 1] << 8;
                            s >>= (signX & 3) << 1;
                            if (s & 0x01) v.x() *= p.slope;
                                if (s & 0x02) v.x() = 0.f;
                            if (s & 0x04) v.y() *= p.slope;
                                if (s & 0x08) v.y() = 0.f;
                            if (s & 0x10) v.z() *= p.slope;
                                if (s & 0x20) v.z() = 0.f;
                            if (s & 0x40) v.w() *= p.slope;
                                if (s & 0x80) v.w() = 0.f;
                        }
                    }
                    else // Forward pass with no sign write.
                    {
                        if (v.x() < 0.f) v.x() *= p.slope;
                            v.x() = InternalType<T>::clamp(v.x(), p.clamp);
                        if (v.y() < 0.f) v.y() *= p.slope;
                            v.y() = InternalType<T>::clamp(v.y(), p.clamp);
                        if (v.z() < 0.f) v.z() *= p.slope;
                            v.z() = InternalType<T>::clamp(v.z(), p.clamp);
                        if (v.w() < 0.f) v.w() *= p.slope;
                            v.w() = InternalType<T>::clamp(v.w(), p.clamp);
                    }

                    s_tileUpXY[idx + 0] = v.x();
                    s_tileUpXY[idx + 1] = v.y();
                    s_tileUpXY[idx + 2] = v.z();
                    s_tileUpXY[idx + 3] = v.w();
                }
            }
            else if (up == 1)
            {
                /*
                DPCT1118:13: SYCL group functions and algorithms must be
                encountered in converged control flow. You may need to adjust
                the code.
                */
                /*
                DPCT1065:47: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier(sycl::access::fence_space::local_space);
                uint32_t groupMask = 15
                                     << ((item_ct1.get_local_id(2) & 31) & ~3);
                int minY = tileOutY ? (tileOutY - tileOutH) * down + tileUpH : 0; // Skip already written signs.
                for (int idx = item_ct1.get_local_id(2);
                     idx < tileUpW * tileUpH;
                     idx += item_ct1.get_local_range(2))
                {
                    int relUpX0, relUpY0;
                    fast_div_mod<tileUpW>(relUpX0, relUpY0, idx);
                    scalar_t v = s_tileIn[idx] * (scalar_t)c_fu[0]; // 1x1 filter.

                    int x = tileOutX * down + relUpX0;
                    int y = tileOutY * down + relUpY0;
                    int signX = x + p.sOfs.x();
                    int signY = y + p.sOfs.y();
                    int signZ = item_ct1.get_group(0) + p.blockZofs;
                    int signXb = signX >> 2;
                    index_t si =
                        signXb +
                        p.sShape.x() * (signY + (index_t)p.sShape.y() * signZ);
                    v *= (scalar_t)((float)up * (float)up * p.gain);

                    if (signWrite)
                    {
                        if (!enableWriteSkip)
                        {
                            // Determine and write sign.
                            uint32_t s = 0;
                            uint32_t signXbit = (1u << signXo);
                            if (v < 0.f)
                            {
                                s = signXbit;
                                v *= p.slope;
                            }
                            if (sycl::fabs(v) > p.clamp)
                            {
                                s = signXbit * 2;
                                v = InternalType<T>::clamp(v, p.clamp);
                            }
                            if ((uint32_t)signXb < p.swLimit &&
                                (uint32_t)signY < p.sShape.y() && signY >= minY)
                            {
                                /*
                                DPCT1108:14: '__shfl_xor_sync' was migrated with
                                the experimental feature masked sub_group
                                function which may not be supported by all
                                compilers or runtimes. You may need to adjust
                                the code.
                                */
                                s += dpct::experimental::
                                    permute_sub_group_by_xor(
                                        groupMask, item_ct1.get_sub_group(), s,
                                        1); // Coalesce.
                                /*
                                DPCT1108:15: '__shfl_xor_sync' was migrated with
                                the experimental feature masked sub_group
                                function which may not be supported by all
                                compilers or runtimes. You may need to adjust
                                the code.
                                */
                                s += dpct::experimental::
                                    permute_sub_group_by_xor(
                                        groupMask, item_ct1.get_sub_group(), s,
                                        2); // Coalesce.
                                p.s[si] = s;                            // Write.
                            }
                        }
                        else
                        {
                            // Determine and write sign.
                            if ((uint32_t)signXb < p.swLimit &&
                                (uint32_t)signY < p.sShape.y() && signY >= minY)
                            {
                                uint32_t s = 0;
                                uint32_t signXbit = (1u << signXo);
                                if (v < 0.f)
                                {
                                    s = signXbit;
                                    v *= p.slope;
                                }
                                if (sycl::fabs(v) > p.clamp)
                                {
                                    s = signXbit * 2;
                                    v = InternalType<T>::clamp(v, p.clamp);
                                }
                                /*
                                DPCT1108:16: '__shfl_xor_sync' was migrated with
                                the experimental feature masked sub_group
                                function which may not be supported by all
                                compilers or runtimes. You may need to adjust
                                the code.
                                */
                                s += dpct::experimental::
                                    permute_sub_group_by_xor(
                                        groupMask, item_ct1.get_sub_group(), s,
                                        1); // Coalesce.
                                /*
                                DPCT1108:17: '__shfl_xor_sync' was migrated with
                                the experimental feature masked sub_group
                                function which may not be supported by all
                                compilers or runtimes. You may need to adjust
                                the code.
                                */
                                s += dpct::experimental::
                                    permute_sub_group_by_xor(
                                        groupMask, item_ct1.get_sub_group(), s,
                                        2); // Coalesce.
                                p.s[si] = s;                            // Write.
                            }
                            else
                            {
                                // Just compute the value.
                                if (v < 0.f) v *= p.slope;
                                v = InternalType<T>::clamp(v, p.clamp);
                            }
                        }
                    }
                    else if (signRead)
                    {
                        // Read sign and apply if within sign tensor bounds.
                        if ((uint32_t)signXb < p.swLimit &&
                            (uint32_t)signY < p.sShape.y())
                        {
                            int s = p.s[si];
                            s >>= signXo;
                            if (s & 1) v *= p.slope;
                            if (s & 2) v = 0.f;
                        }
                    }
                    else // Forward pass with no sign write.
                    {
                        if (v < 0.f) v *= p.slope;
                        v = InternalType<T>::clamp(v, p.clamp);
                    }

                    if (!downInline) // Write into temporary buffer.
                        s_tileUpXY[idx] = v;
                    else if ((uint32_t)x < p.yShape.x() &&
                             (uint32_t)y <
                                 p.yShape
                                     .y()) // Write directly into output buffer
                        // *((T *)((char *)p.y +
                        //         (x * get_stride<index_t>(p.yStride.x()) +
                        //          y * get_stride<index_t>(p.yStride.y()) +
                        //          mapOfsOut))) = (T)(v * (scalar_t)c_fd[0]);
                        (T&)(yacc[x * get_stride<index_t>(p.yStride.x()) +
                             y * get_stride<index_t>(p.yStride.y()) +
                             mapOfsOut]) = (T)(v * (scalar_t)c_fd[0]);
                }
            }
        }

        // Downsampling.
        if (filterMode == MODE_SUSD || filterMode == MODE_FUSD)
        {
            // Horizontal downsampling.
            /*
            DPCT1118:18: SYCL group functions and algorithms must be encountered
            in converged control flow. You may need to adjust the code.
            */
            /*
            DPCT1065:48: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier(sycl::access::fence_space::local_space);
            if (down == 4 && tileOutW % 4 == 0)
            {
                // Calculate 4 pixels at a time.
                for (int idx = item_ct1.get_local_id(2) * 4;
                     idx < tileOutW * tileUpH;
                     idx += item_ct1.get_local_range(2) * 4)
                {
                    int relOutX0, relUpY;
                    fast_div_mod<tileOutW>(relOutX0, relUpY, idx);
                    int relUpX0 = relOutX0 * down;
                    int src0 = relUpY * tileUpW + relUpX0;
                    vec4_t v = InternalType<T>::zero_vec4();
                    #pragma unroll
                    for (int step = 0; step < fdSize; step++)
                    {
                        v.x() +=
                            s_tileUpXY[src0 + 0 + step] * (scalar_t)c_fd[step];
                        v.y() +=
                            s_tileUpXY[src0 + 4 + step] * (scalar_t)c_fd[step];
                        v.z() +=
                            s_tileUpXY[src0 + 8 + step] * (scalar_t)c_fd[step];
                        v.w() +=
                            s_tileUpXY[src0 + 12 + step] * (scalar_t)c_fd[step];
                    }
                    s_tileDownX[idx + 0] = v.x();
                    s_tileDownX[idx + 1] = v.y();
                    s_tileDownX[idx + 2] = v.z();
                    s_tileDownX[idx + 3] = v.w();
                }
            }
            else if ((down == 2 || down == 4) && (tileOutW % 2 == 0))
            {
                // Calculate 2 pixels at a time.
                for (int idx = item_ct1.get_local_id(2) * 2;
                     idx < tileOutW * tileUpH;
                     idx += item_ct1.get_local_range(2) * 2)
                {
                    int relOutX0, relUpY;
                    fast_div_mod<tileOutW>(relOutX0, relUpY, idx);
                    int relUpX0 = relOutX0 * down;
                    int src0 = relUpY * tileUpW + relUpX0;
                    vec2_t v = InternalType<T>::zero_vec2();
                    #pragma unroll
                    for (int step = 0; step < fdSize; step++)
                    {
                        v.x() +=
                            s_tileUpXY[src0 + 0 + step] * (scalar_t)c_fd[step];
                        v.y() += s_tileUpXY[src0 + down + step] *
                                 (scalar_t)c_fd[step];
                    }
                    s_tileDownX[idx + 0] = v.x();
                    s_tileDownX[idx + 1] = v.y();
                }
            }
            else
            {
                // Calculate 1 pixel at a time.
                for (int idx = item_ct1.get_local_id(2);
                     idx < tileOutW * tileUpH;
                     idx += item_ct1.get_local_range(2))
                {
                    int relOutX0, relUpY;
                    fast_div_mod<tileOutW>(relOutX0, relUpY, idx);
                    int relUpX0 = relOutX0 * down;
                    int src = relUpY * tileUpW + relUpX0;
                    scalar_t v = 0.f;
                    #pragma unroll
                    for (int step = 0; step < fdSize; step++)
                        v += s_tileUpXY[src + step] * (scalar_t)c_fd[step];
                    s_tileDownX[idx] = v;
                }
            }

            // Vertical downsampling & store output tile.
            /*
            DPCT1118:19: SYCL group functions and algorithms must be encountered
            in converged control flow. You may need to adjust the code.
            */
            /*
            DPCT1065:49: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier(sycl::access::fence_space::local_space);
            for (int idx = item_ct1.get_local_id(2); idx < tileOutW * tileOutH;
                 idx += item_ct1.get_local_range(2))
            {
                int relOutX, relOutY0;
                fast_div_mod<tileOutW>(relOutX, relOutY0, idx);
                int relUpY0 = relOutY0 * down;
                int src0 = relUpY0 * tileOutW + relOutX;
                scalar_t v = 0;
                #pragma unroll
                for (int step = 0; step < fdSize; step++)
                    v += s_tileDownX[src0 + step * tileOutW] * (scalar_t)c_fd[step];

                int outX = tileOutX + relOutX;
                int outY = tileOutY + relOutY0;

                if (outX < p.yShape.x() & outY < p.yShape.y())
                    // *((T *)((char *)p.y +
                    //         (outX * get_stride<index_t>(p.yStride.x()) +
                    //          outY * get_stride<index_t>(p.yStride.y()) +
                    //          mapOfsOut))) = (T)v;
                    (T&)(yacc[outX * get_stride<index_t>(p.yStride.x()) +
                         outY * get_stride<index_t>(p.yStride.y()) +
                         mapOfsOut]) = (T)v;
            }
        }
        else if (filterMode == MODE_SUFD || filterMode == MODE_FUFD)
        {
            // Full downsampling filter.
            if (down == 2)
            {
                // 2-wide.
                /*
                DPCT1118:20: SYCL group functions and algorithms must be
                encountered in converged control flow. You may need to adjust
                the code.
                */
                /*
                DPCT1065:50: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier(sycl::access::fence_space::local_space);
                for (int idx = item_ct1.get_local_id(2) * 2;
                     idx < tileOutW * tileOutH;
                     idx += item_ct1.get_local_range(2) * 2)
                {
                    int relOutX0, relOutY0;
                    fast_div_mod<tileOutW>(relOutX0, relOutY0, idx);
                    int relUpX0 = relOutX0 * down;
                    int relUpY0 = relOutY0 * down;
                    int src0 = relUpY0 * tileUpW + relUpX0;
                    vec2_t v = InternalType<T>::zero_vec2();
                    #pragma unroll
                    for (int sy = 0; sy < fdSize; sy++)
                    #pragma unroll
                    for (int sx = 0; sx < fdSize; sx++)
                    {
                        v.x() += s_tileUpXY[src0 + 0 + sx + sy * tileUpW] *
                                 (scalar_t)c_fd[sx + sy * MAX_FILTER_SIZE];
                        v.y() += s_tileUpXY[src0 + 2 + sx + sy * tileUpW] *
                                 (scalar_t)c_fd[sx + sy * MAX_FILTER_SIZE];
                    }

                    int outX = tileOutX + relOutX0;
                    int outY = tileOutY + relOutY0;
                    if ((uint32_t)outY < p.yShape.y())
                    {
                        index_t ofs =
                            outX * get_stride<index_t>(p.yStride.x()) +
                            outY * get_stride<index_t>(p.yStride.y()) +
                            mapOfsOut;
                        // if (outX + 0 < p.yShape.x()) *
                        //     ((T *)((char *)p.y + ofs)) = (T)v.x();
                        // if (outX + 1 < p.yShape.x()) *
                        //     ((T *)((char *)p.y + ofs +
                        //            get_stride<index_t>(p.yStride.x()))) =
                        //     (T)v.y();
                        if (outX + 0 < p.yShape.x())
                            (T&)(yacc[ofs]) = (T)v.x();
                        if (outX + 1 < p.yShape.x())
                            (T&)(yacc[ofs + get_stride<index_t>(p.yStride.x())]) = (T)v.y();
                    }
                }
            }
            else if (down == 1 && !downInline)
            {
                // Thread per pixel.
                /*
                DPCT1118:21: SYCL group functions and algorithms must be
                encountered in converged control flow. You may need to adjust
                the code.
                */
                /*
                DPCT1065:51: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier(sycl::access::fence_space::local_space);
                for (int idx = item_ct1.get_local_id(2);
                     idx < tileOutW * tileOutH;
                     idx += item_ct1.get_local_range(2))
                {
                    int relOutX0, relOutY0;
                    fast_div_mod<tileOutW>(relOutX0, relOutY0, idx);
                    scalar_t v = s_tileUpXY[idx] * (scalar_t)c_fd[0]; // 1x1 filter.

                    int outX = tileOutX + relOutX0;
                    int outY = tileOutY + relOutY0;
                    if ((uint32_t)outX < p.yShape.x() &&
                        (uint32_t)outY < p.yShape.y())
                        // *((T *)((char *)p.y +
                        //         (outX * get_stride<index_t>(p.yStride.x()) +
                        //          outY * get_stride<index_t>(p.yStride.y()) +
                        //          mapOfsOut))) = (T)v;
                        (T&)(yacc[outX * get_stride<index_t>(p.yStride.x()) +
                             outY * get_stride<index_t>(p.yStride.y()) +
                             mapOfsOut]) = (T)v;
                }
            }
        }

        if (!enableXrep)
            break;
    }
}

//------------------------------------------------------------------------
// Compute activation function and signs for upsampled data tensor, modifying data tensor in-place. Used for accelerating the generic variant.
// Sign tensor is known to be contiguous, and p.x and p.s have the same z, w dimensions. 64-bit indexing is always used.

template <class T, bool signWrite, bool signRead>
/*
DPCT1110:26: The total declared local variable size in device function
filtered_lrelu_act_kernel exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
static void filtered_lrelu_act_kernel(filtered_lrelu_act_kernel_params p,
                                      const sycl::nd_item<3> &item_ct1)
{
    typedef typename InternalType<T>::scalar_t scalar_t;

    // Indexing.
    int32_t x = item_ct1.get_local_id(2) +
                item_ct1.get_group(2) * item_ct1.get_local_range(2);
    int32_t ymax = signWrite ? p.sShape.y() : p.xShape.y();
    int32_t qmax = p.xShape.z() *
                   p.xShape.w(); // Combined minibatch*channel maximum index.

    // Loop to accommodate oversized tensors.
    for (int32_t q = item_ct1.get_group(0); q < qmax;
         q += item_ct1.get_group_range(0))
    for (int32_t y = item_ct1.get_group(1); y < ymax;
         y += item_ct1.get_group_range(1))
    {
        // Extract z and w (channel, minibatch index).
        int32_t w = q / p.xShape.z();
        int32_t z = q - w * p.xShape.z();

        // Choose behavior based on sign read/write mode.
        if (signWrite)
        {
            // Process value if in p.x.
            uint32_t s = 0;
            if (x < p.xShape.x() && y < p.xShape.y())
            {
                int64_t ix = x * p.xStride.x() + y * p.xStride.y() +
                             z * p.xStride.z() + w * p.xStride.w();
                T* pv = ((T*)p.x) + ix;
                scalar_t v = (scalar_t)(*pv);

                // Gain, LReLU, clamp.
                v *= p.gain;
                if (v < 0.f)
                {
                    v *= p.slope;
                    s = 1; // Sign.
                }
                /*
                DPCT1064:52: Migrated fabsf call is used in a macro/template
                definition and may not be valid for all macro/template uses.
                Adjust the code.
                */
                if (sycl::fabs((float)v) > p.clamp)
                {
                    v = InternalType<T>::clamp(v, p.clamp);
                    s = 2; // Clamp.
                }

                *pv = (T)v; // Write value.
            }

            // Coalesce into threads 0 and 16 of warp.
            uint32_t m =
                (item_ct1.get_local_id(2) & 16) ? 0xffff0000u : 0x0000ffffu;
            s <<= ((item_ct1.get_local_id(2) & 15) << 1); // Shift into place.
            /*
            DPCT1108:22: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            s |= dpct::experimental::permute_sub_group_by_xor(
                m, item_ct1.get_sub_group(), s, 1); // Distribute.
            /*
            DPCT1108:23: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            s |= dpct::experimental::permute_sub_group_by_xor(
                m, item_ct1.get_sub_group(), s, 2);
            /*
            DPCT1108:24: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            s |= dpct::experimental::permute_sub_group_by_xor(
                m, item_ct1.get_sub_group(), s, 4);
            /*
            DPCT1108:25: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            s |= dpct::experimental::permute_sub_group_by_xor(
                m, item_ct1.get_sub_group(), s, 8);

            // Write signs if leader and in p.s.
            if (!(item_ct1.get_local_id(2) & 15) &&
                x < p.sShape.x()) // y is always in.
            {
                uint64_t is = x + p.sShape.x() * (y + (int64_t)p.sShape.y() *
                                                          q); // Contiguous.
                ((uint32_t*)p.s)[is >> 4] = s;
            }
        }
        else if (signRead)
        {
            // Process value if in p.x.
            if (x < p.xShape.x()) // y is always in.
            {
                int64_t ix = x * p.xStride.x() + y * p.xStride.y() +
                             z * p.xStride.z() + w * p.xStride.w();
                T* pv = ((T*)p.x) + ix;
                scalar_t v = (scalar_t)(*pv);
                v *= p.gain;

                // Apply sign buffer offset.
                uint32_t sx = x + p.sOfs.x();
                uint32_t sy = y + p.sOfs.y();

                // Read and apply signs if we land inside valid region of sign buffer.
                if (sx < p.sShape.x() && sy < p.sShape.y())
                {
                    uint64_t is = (sx >> 2) + (p.sShape.x() >> 2) *
                                                  (sy + (uint64_t)p.sShape.y() *
                                                            q); // Contiguous.
                    unsigned char s = p.s[is];
                    s >>= (sx & 3) << 1; // Shift into place.
                    if (s & 1) // Sign?
                        v *= p.slope;
                    if (s & 2) // Clamp?
                        v = 0.f;
                }

                *pv = (T)v; // Write value.
            }
        }
        else
        {
            // Forward pass with no sign write. Process value if in p.x.
            if (x < p.xShape.x()) // y is always in.
            {
                int64_t ix = x * p.xStride.x() + y * p.xStride.y() +
                             z * p.xStride.z() + w * p.xStride.w();
                T* pv = ((T*)p.x) + ix;
                scalar_t v = (scalar_t)(*pv);
                v *= p.gain;
                if (v < 0.f)
                    v *= p.slope;
                /*
                DPCT1064:53: Migrated fabsf call is used in a macro/template
                definition and may not be valid for all macro/template uses.
                Adjust the code.
                */
                if (sycl::fabs((float)v) > p.clamp)
                    v = InternalType<T>::clamp(v, p.clamp);
                *pv = (T)v; // Write value.
            }
        }
    }
}

// template <class T, bool signWrite, bool signRead> void* choose_filtered_lrelu_act_kernel(void)
// {
//     return (void*)filtered_lrelu_act_kernel<T, signWrite, signRead>;
// }

//------------------------------------------------------------------------
// CUDA kernel selection.

// template <class T, class index_t, bool signWrite, bool signRead> filtered_lrelu_kernel_spec choose_filtered_lrelu_kernel(const filtered_lrelu_kernel_params& p, int sharedKB)
// {
//     filtered_lrelu_kernel_spec s = { 0 };

//     // Return the first matching kernel.
// #define CASE(SH, U, FU, D, FD, MODE, TW, TH, W, XR, WS)                        \
//     if (sharedKB >= SH)                                                        \
//         if ((p.fuShape.y() == 0 &&                                             \
//              (MODE == MODE_SUSD || MODE == MODE_SUFD)) ||                      \
//             (p.fuShape.y() > 0 && (MODE == MODE_FUSD || MODE == MODE_FUFD)))   \
//             if ((p.fdShape.y() == 0 &&                                         \
//                  (MODE == MODE_SUSD || MODE == MODE_FUSD)) ||                  \
//                 (p.fdShape.y() > 0 &&                                          \
//                  (MODE == MODE_SUFD || MODE == MODE_FUFD)))                    \
//                 if (p.up == U && p.fuShape.x() <= FU && p.fuShape.y() <= FU && \
//                     p.down == D && p.fdShape.x() <= FD && p.fdShape.y() <= FD) \
//                 {                                                              \
//                     static_assert((D * TW % 4) == 0,                           \
//                                   "down * tileWidth must be divisible by 4");  \
//                     static_assert(FU % U == 0,                                 \
//                                   "upscaling filter size must be multiple of " \
//                                   "upscaling factor");                         \
//                     static_assert(FD % D == 0,                                 \
//                                   "downscaling filter size must be multiple "  \
//                                   "of downscaling factor");                    \
//                     s.setup = (void *)setup_filters_kernel;                    \
//                     s.exec = (void *)filtered_lrelu_kernel<                    \
//                         T, index_t, SH, signWrite, signRead, MODE, U, FU, D,   \
//                         FD, TW, TH, W * 32, !!XR, !!WS>;                       \
//                     s.tileOut = sycl::int2(TW, TH);                            \
//                     s.numWarps = W;                                            \
//                     s.xrep = XR;                                               \
//                     s.dynamicSharedKB = (SH == 48) ? 0 : SH;                   \
//                     return s;                                                  \
//                 }

//     // Launch parameters for various kernel specializations.
//     // Small filters must be listed before large filters, otherwise the kernel for larger filter will always match first.
//     // Kernels that use more shared memory must be listed before those that use less, for the same reason.

//     CASE(/*sharedKB*/48, /*up,fu*/1,1,  /*down,fd*/1,1,  /*mode*/MODE_FUFD, /*tw,th,warps,xrep,wskip*/64,  178, 32,  0,  0) // 1t-upf1-downf1
//     CASE(/*sharedKB*/48, /*up,fu*/2,8,  /*down,fd*/1,1,  /*mode*/MODE_SUFD, /*tw,th,warps,xrep,wskip*/152, 95,  16,  0,  0) // 4t-ups2-downf1
//     CASE(/*sharedKB*/48, /*up,fu*/1,1,  /*down,fd*/2,8,  /*mode*/MODE_FUSD, /*tw,th,warps,xrep,wskip*/56,  22,  16,  0,  0) // 4t-upf1-downs2
//     CASE(/*sharedKB*/48, /*up,fu*/2,8,  /*down,fd*/2,8,  /*mode*/MODE_SUSD, /*tw,th,warps,xrep,wskip*/56,  29,  16,  11, 0) // 4t-ups2-downs2
//     CASE(/*sharedKB*/48, /*up,fu*/2,8,  /*down,fd*/2,8,  /*mode*/MODE_FUSD, /*tw,th,warps,xrep,wskip*/60,  28,  16,  0,  0) // 4t-upf2-downs2
//     CASE(/*sharedKB*/48, /*up,fu*/2,8,  /*down,fd*/2,8,  /*mode*/MODE_SUFD, /*tw,th,warps,xrep,wskip*/56,  28,  16,  0,  0) // 4t-ups2-downf2
//     CASE(/*sharedKB*/48, /*up,fu*/4,16, /*down,fd*/2,8,  /*mode*/MODE_SUSD, /*tw,th,warps,xrep,wskip*/56,  31,  16,  11, 0) // 4t-ups4-downs2
//     CASE(/*sharedKB*/48, /*up,fu*/4,16, /*down,fd*/2,8,  /*mode*/MODE_SUFD, /*tw,th,warps,xrep,wskip*/56,  36,  16,  0,  0) // 4t-ups4-downf2
//     CASE(/*sharedKB*/48, /*up,fu*/2,8,  /*down,fd*/4,16, /*mode*/MODE_SUSD, /*tw,th,warps,xrep,wskip*/16,  22,  16,  12, 0) // 4t-ups2-downs4
//     CASE(/*sharedKB*/48, /*up,fu*/2,8,  /*down,fd*/4,16, /*mode*/MODE_FUSD, /*tw,th,warps,xrep,wskip*/29,  15,  16,  0,  0) // 4t-upf2-downs4
//     CASE(/*sharedKB*/48, /*up,fu*/2,12, /*down,fd*/1,1,  /*mode*/MODE_SUFD, /*tw,th,warps,xrep,wskip*/96,  150, 28,  0,  0) // 6t-ups2-downf1
//     CASE(/*sharedKB*/48, /*up,fu*/1,1,  /*down,fd*/2,12, /*mode*/MODE_FUSD, /*tw,th,warps,xrep,wskip*/32,  35,  24,  0,  0) // 6t-upf1-downs2
//     CASE(/*sharedKB*/48, /*up,fu*/2,12, /*down,fd*/2,12, /*mode*/MODE_SUSD, /*tw,th,warps,xrep,wskip*/32,  46,  16,  10, 0) // 6t-ups2-downs2
//     CASE(/*sharedKB*/48, /*up,fu*/2,12, /*down,fd*/2,12, /*mode*/MODE_FUSD, /*tw,th,warps,xrep,wskip*/58,  28,  24,  8,  0) // 6t-upf2-downs2
//     CASE(/*sharedKB*/48, /*up,fu*/2,12, /*down,fd*/2,12, /*mode*/MODE_SUFD, /*tw,th,warps,xrep,wskip*/52,  28,  16,  0,  0) // 6t-ups2-downf2
//     CASE(/*sharedKB*/48, /*up,fu*/4,24, /*down,fd*/2,12, /*mode*/MODE_SUSD, /*tw,th,warps,xrep,wskip*/32,  51,  16,  5,  0) // 6t-ups4-downs2
//     CASE(/*sharedKB*/48, /*up,fu*/4,24, /*down,fd*/2,12, /*mode*/MODE_SUFD, /*tw,th,warps,xrep,wskip*/32,  56,  16,  6,  0) // 6t-ups4-downf2
//     CASE(/*sharedKB*/48, /*up,fu*/2,12, /*down,fd*/4,24, /*mode*/MODE_SUSD, /*tw,th,warps,xrep,wskip*/16,  18,  16,  12, 0) // 6t-ups2-downs4
//     CASE(/*sharedKB*/96, /*up,fu*/2,12, /*down,fd*/4,24, /*mode*/MODE_FUSD, /*tw,th,warps,xrep,wskip*/27,  31,  32,  6,  0) // 6t-upf2-downs4 96kB
//     CASE(/*sharedKB*/48, /*up,fu*/2,12, /*down,fd*/4,24, /*mode*/MODE_FUSD, /*tw,th,warps,xrep,wskip*/27,  13,  24,  0,  0) // 6t-upf2-downs4
//     CASE(/*sharedKB*/48, /*up,fu*/2,16, /*down,fd*/1,1,  /*mode*/MODE_SUFD, /*tw,th,warps,xrep,wskip*/148, 89,  24,  0,  0) // 8t-ups2-downf1
//     CASE(/*sharedKB*/48, /*up,fu*/1,1,  /*down,fd*/2,16, /*mode*/MODE_FUSD, /*tw,th,warps,xrep,wskip*/32,  31,  16,  5,  0) // 8t-upf1-downs2
//     CASE(/*sharedKB*/48, /*up,fu*/2,16, /*down,fd*/2,16, /*mode*/MODE_SUSD, /*tw,th,warps,xrep,wskip*/32,  41,  16,  9,  0) // 8t-ups2-downs2
//     CASE(/*sharedKB*/48, /*up,fu*/2,16, /*down,fd*/2,16, /*mode*/MODE_FUSD, /*tw,th,warps,xrep,wskip*/56,  26,  24,  0,  0) // 8t-upf2-downs2
//     CASE(/*sharedKB*/48, /*up,fu*/2,16, /*down,fd*/2,16, /*mode*/MODE_SUFD, /*tw,th,warps,xrep,wskip*/32,  40,  16,  0,  0) // 8t-ups2-downf2
//     CASE(/*sharedKB*/48, /*up,fu*/4,32, /*down,fd*/2,16, /*mode*/MODE_SUSD, /*tw,th,warps,xrep,wskip*/32,  46,  24,  5,  0) // 8t-ups4-downs2
//     CASE(/*sharedKB*/48, /*up,fu*/4,32, /*down,fd*/2,16, /*mode*/MODE_SUFD, /*tw,th,warps,xrep,wskip*/32,  50,  16,  0,  0) // 8t-ups4-downf2
//     CASE(/*sharedKB*/96, /*up,fu*/2,16, /*down,fd*/4,32, /*mode*/MODE_SUSD, /*tw,th,warps,xrep,wskip*/24,  24,  32,  12, 1) // 8t-ups2-downs4 96kB
//     CASE(/*sharedKB*/48, /*up,fu*/2,16, /*down,fd*/4,32, /*mode*/MODE_SUSD, /*tw,th,warps,xrep,wskip*/16,  13,  16,  10, 1) // 8t-ups2-downs4
//     CASE(/*sharedKB*/96, /*up,fu*/2,16, /*down,fd*/4,32, /*mode*/MODE_FUSD, /*tw,th,warps,xrep,wskip*/25,  28,  28,  4,  0) // 8t-upf2-downs4 96kB
//     CASE(/*sharedKB*/48, /*up,fu*/2,16, /*down,fd*/4,32, /*mode*/MODE_FUSD, /*tw,th,warps,xrep,wskip*/25,  10,  24,  0,  0) // 8t-upf2-downs4

//     #undef CASE
//     return s; // No kernel found.
// }

//------------------------------------------------------------------------
// XPU kernel launchers.

template <class T, class index_t, bool signWrite, bool signRead,
          int MODE, int U, int FU, int D, int FD, int TW, int TH, int W, int XR,
          int WS> // TODO: XR and WS could be just booleans and the actual value
                  // passed as a parameter - to reduce the number of template
                  // instantiations/specializations (but then duplicate
                  // specializations would need to be avoided)
void run_filtered_lrelu_kernel(filtered_lrelu_kernel_params &p) try {
    //std::cout << "run_filtered_lrelu_kernel" << std::endl;
    
    auto device_type = c10::DeviceType::XPU;
    c10::impl::VirtualGuardImpl impl(device_type);
    c10::Stream c10_stream = impl.getStream(c10::Device(device_type));
    auto& queue = xpu::get_queue_from_stream(c10_stream);

    c_fbuf.init(queue);
    auto c_fbuf_ptr_ct1 = c_fbuf.get_ptr();

    // Launch XPU kernel.
    int bx = W * 32;
    int gx = (p.yShape.x() - 1) / TW + 1;
    int gy = (p.yShape.y() - 1) / TH + 1;
    int gz = p.yShape.z() * p.yShape.w();

    // Repeat multiple horizontal tiles in a CTA?
    if (XR)
    {
        p.tilesXrep = XR;
        p.tilesXdim = gx;
       
        gx = (gx + p.tilesXrep - 1) / p.tilesXrep;
        std::swap(gx, gy);
    }
    else
    {
        p.tilesXrep = 0;
        p.tilesXdim = 0;
    }

    // Launch filter setup kernel.
    /*
    DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
   auto task = queue.submit([&](sycl::handler &cgh) {
        g_fbuf.init(queue);

        auto g_fbuf_ptr_ct1 = g_fbuf.get_ptr();

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1024),
                                           sycl::range<3>(1, 1, 1024)),
                         [=](sycl::nd_item<3> item_ct1) {
                           setup_filters_kernel(p, item_ct1,
                                                g_fbuf_ptr_ct1);
                         });
      });

    // Static definitions. - partially copied from inside the .cu (needed to compute e.g. the buffer sizes)
    typedef typename InternalType<T>::scalar_t scalar_t;
    const int tileUpW    = (TW * D + (FD - 1) - (D - 1) + 3) & ~3;  // Upsampled tile width, rounded up to multiple of 4.
    const int tileUpH    = TH * D + (FD - 1) - (D - 1);             // Upsampled tile height.
    const int tileInW    = CEIL_DIV(tileUpW  + (FU - 1), U);                   // Input tile width.
    const int tileInH    = CEIL_DIV(tileUpH  + (FU - 1), U);                   // Input tile height.
    const int tileUpH_up = CEIL_DIV(tileUpH, U) * U;                              // Upsampled tile height rounded up to a multiple of up.
    const int tileInH_up = CEIL_DIV(tileUpH_up + (FU - 1), U);                 // For allocations only, to avoid shared memory read overruns with up=2 and up=4.

    // Merge 1x1 downsampling into last upsampling step for upf1 and ups2.
    const bool downInline = (D == 1) && ((U == 1 && MODE == MODE_FUFD) || (U == 2 && MODE == MODE_SUFD));

    // Sizes of logical buffers.
    const int szIn    = tileInH_up * tileInW;
    const int szUpX   = tileInH_up * tileUpW;
    const int szUpXY  = downInline ? 0 : (tileUpH * tileUpW);
    const int szDownX = tileUpH * TW;

    // Sizes for shared memory arrays.
    const int s_buf0_size_base =
        (MODE == MODE_SUSD) ? MAX(szIn, szUpXY) :
        (MODE == MODE_FUSD) ? MAX(szIn, szDownX) :
        (MODE == MODE_SUFD) ? MAX(szIn, szUpXY) :
        (MODE == MODE_FUFD) ? szIn :
        -1;
    const int s_buf1_size_base =
        (MODE == MODE_SUSD) ? MAX(szUpX, szDownX) :
        (MODE == MODE_FUSD) ? szUpXY :
        (MODE == MODE_SUFD) ? szUpX  :
        (MODE == MODE_FUFD) ? szUpXY :
        -1;

    // Ensure U128 alignment.
    const int s_buf0_size = (s_buf0_size_base + 3) & ~3;
    const int s_buf1_size = (s_buf1_size_base + 3) & ~3;

    task.wait();
    
    // Copy kernels to constant memory.
    if      ( signWrite && !signRead) (copy_filters<true,  false>(&queue));
    else if (!signWrite &&  signRead) (copy_filters<false, true >(&queue));
    else if (!signWrite && !signRead) (copy_filters<false, false>(&queue));

    // Launch main kernel.
    const int maxSubGz = 65535; // CUDA maximum for block z dimension.
    for (int zofs=0; zofs < gz; zofs += maxSubGz) // Do multiple launches if gz is too big.
    {
        p.blockZofs = zofs;
        int subGz = std::min(maxSubGz, gz - zofs);

        sycl::buffer<char> yBuf((char*)p.y, p.y_nbytes);
        /*
        DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
    queue.submit([&](sycl::handler &cgh) {

          sycl::accessor yAccessor(yBuf, cgh, sycl::write_only, sycl::no_init);
          //sycl::accessor yAccessor(yBuf, cgh, sycl::write_only);
          //sycl::accessor yAccessor(yBuf, cgh);
//          T * ptr = yAccessor.get_pointer();

          sycl::local_accessor<scalar_t, 1> s_buf0_st_acc_ct1(
              s_buf0_size + s_buf1_size,
              cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(sycl::range<3>(subGz, gy, gx) *
                                    sycl::range<3>(1, 1, bx),
                                sycl::range<3>(1, 1, bx)),
              [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                  32)]] {
                filtered_lrelu_kernel<T, index_t, signWrite, signRead, MODE,
                                      U, FU, D, FD, TW, TH, W * 32, !!XR, !!WS>(
                    p, item_ct1, c_fbuf_ptr_ct1,
                    s_buf0_st_acc_ct1.get_pointer(), yAccessor
                    );
              });
        }).wait();
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

template <class T, bool signWrite, bool signRead>
void run_filtered_lrelu_act_kernel(filtered_lrelu_act_kernel_params &p) try {
    //std::cout << "run_filtered_lrelu_act_kernel" << std::endl;
    // Launch kernel.
    void* args[] = {&p};
    int bx = 128; // 4 warps per block.

    // Logical size of launch = writeSigns ? p.s : p.x
    uint32_t gx = signWrite ? p.sShape.x() : p.xShape.x();
    uint32_t gy = signWrite ? p.sShape.y() : p.xShape.y();
    uint32_t gz =
        p.xShape.z() * p.xShape.w(); // Same as in p.sShape if signs are in use.
    gx = (gx - 1) / bx + 1;

    // Make sure grid y and z dimensions are within CUDA launch limits. Kernel loops internally to do the rest.
    const uint32_t gmax = 65535;
    gy = std::min(gy, gmax);
    gz = std::min(gz, gmax);

    // Launch.
    /*
    DPCT1049:2: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
  {
    auto device_type = c10::DeviceType::XPU;
    c10::impl::VirtualGuardImpl impl(device_type);
    c10::Stream c10_stream = impl.getStream(c10::Device(device_type));
    auto& queue = xpu::get_queue_from_stream(c10_stream);

    dpct::has_capability_or_fail(
        queue.get_device(),
        {sycl::aspect::fp64});
    queue.submit([&](sycl::handler &cgh) {
          auto p_ct0 = *(filtered_lrelu_act_kernel_params *)args[0];

          cgh.parallel_for(
              sycl::nd_range<3>(sycl::range<3>(gz, gy, gx) *
                                    sycl::range<3>(1, 1, bx),
                                sycl::range<3>(1, 1, bx)),
              [=](sycl::nd_item<3> item_ct1)
                  [[intel::reqd_sub_group_size(32)]] {
                    filtered_lrelu_act_kernel<T, signWrite, signRead>(p_ct0,
                                                                      item_ct1);
                  });
        }).wait();
  }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

//------------------------------------------------------------------------

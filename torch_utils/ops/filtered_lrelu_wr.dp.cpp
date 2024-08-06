// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

// #include <sycl/sycl.hpp>
// #include <dpct/dpct.hpp>
// #include "filtered_lrelu.dp.cpp"

// // Template/kernel specializations for sign write mode.

// // Full op, 32-bit indexing.
// template filtered_lrelu_kernel_spec choose_filtered_lrelu_kernel<c10::Half, int32_t, true, false>(const filtered_lrelu_kernel_params& p, int sharedKB);
// template filtered_lrelu_kernel_spec choose_filtered_lrelu_kernel<float,     int32_t, true, false>(const filtered_lrelu_kernel_params& p, int sharedKB);

// // Full op, 64-bit indexing.
// template filtered_lrelu_kernel_spec choose_filtered_lrelu_kernel<c10::Half, int64_t, true, false>(const filtered_lrelu_kernel_params& p, int sharedKB);
// template filtered_lrelu_kernel_spec choose_filtered_lrelu_kernel<float,     int64_t, true, false>(const filtered_lrelu_kernel_params& p, int sharedKB);

// // Activation/signs only for generic variant. 64-bit indexing.
// template void* choose_filtered_lrelu_act_kernel<c10::Half, true, false>(void);
// template void* choose_filtered_lrelu_act_kernel<float,     true, false>(void);
// template void* choose_filtered_lrelu_act_kernel<double,    true, false>(void);

// // Copy filters to constant memory.
// template dpct::err0 copy_filters<true, false>(dpct::queue_ptr stream);


#include "filtered_lrelu.dp.cpp"

// Template/kernel specializations for sign write mode.

// Full op, 32-bit indexing.
#define CASE(...) \
    SPECIALIZE_CASE(c10::Half, int32_t, true, false, __VA_ARGS__) \
    SPECIALIZE_CASE(float,     int32_t, true, false, __VA_ARGS__)
#include "filtered_lrelu_cases.h"
#undef CASE

// Full op, 64-bit indexing.
#define CASE(...) \
    SPECIALIZE_CASE(c10::Half, int64_t, true, false, __VA_ARGS__) \
    SPECIALIZE_CASE(float,     int64_t, true, false, __VA_ARGS__)
#include "filtered_lrelu_cases.h"
#undef CASE

// Activation/signs only for generic variant. 64-bit indexing.
template void run_filtered_lrelu_act_kernel<c10::Half, true, false>(filtered_lrelu_act_kernel_params& p);
template void run_filtered_lrelu_act_kernel<float,     true, false>(filtered_lrelu_act_kernel_params& p);
template void run_filtered_lrelu_act_kernel<double,    true, false>(filtered_lrelu_act_kernel_params& p);

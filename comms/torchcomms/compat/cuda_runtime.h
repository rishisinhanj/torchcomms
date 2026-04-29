// Copyright (c) Meta Platforms, Inc. and affiliates.
// CUDA-to-HIP compatibility wrapper — maps CUDA runtime API to HIP.
// Included via -I priority when building with ROCm.
#pragma once

#include <hip/hip_runtime.h>

// Types
#define cudaError_t hipError_t
#define cudaStream_t hipStream_t
#define cudaEvent_t hipEvent_t
#define cudaDeviceProp hipDeviceProp_t
#define cudaPointerAttributes hipPointerAttribute_t
#define cudaIpcMemHandle_t hipIpcMemHandle_t
#define cudaMemcpyKind hipMemcpyKind
#define cudaStreamCaptureStatus hipStreamCaptureStatus
#define cudaUserObject_t hipUserObject_t
#define cudaGraph_t hipGraph_t
#define cudaGraphNode_t hipGraphNode_t
#define cudaHostFn_t hipHostFn_t

// Error codes
#define cudaSuccess hipSuccess
#define cudaErrorNotReady hipErrorNotReady
#define cudaErrorInvalidValue hipErrorInvalidValue
#define cudaErrorInvalidResourceHandle hipErrorInvalidResourceHandle
#define cudaErrorIllegalState hipErrorNotReady
#define cudaErrorLaunchFailure hipErrorLaunchFailure
#define cudaErrorContextIsDestroyed hipErrorDeinitialized
#define cudaErrorCudartUnloading hipErrorDeinitialized

// Memory types
#define cudaMemoryTypeUnregistered hipMemoryTypeUnregistered
#define cudaMemoryTypeDevice hipMemoryTypeDevice
#define cudaMemoryTypeHost hipMemoryTypeHost
#define cudaMemoryTypeManaged hipMemoryTypeManaged

// Device management
#define cudaSetDevice hipSetDevice
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDeviceGetPCIBusId hipDeviceGetPCIBusId
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDeviceReset hipDeviceReset
#define cudaDriverGetVersion hipDriverGetVersion
#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString
#define cudaPointerGetAttributes hipPointerGetAttributes

// Memory management
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMallocHost hipHostMalloc
#define cudaHostAlloc hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaHostGetDevicePointer hipHostGetDevicePointer
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDefault hipMemcpyDefault
#define cudaHostAllocDefault hipHostMallocDefault
#define cudaHostAllocMapped hipHostMallocMapped

// IPC
#define cudaIpcGetMemHandle hipIpcGetMemHandle
#define cudaIpcOpenMemHandle hipIpcOpenMemHandle

// Stream management
#define cudaStreamDefault hipStreamDefault
#define cudaStreamCaptureMode hipStreamCaptureMode
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamCreateWithPriority hipStreamCreateWithPriority
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStreamNonBlocking hipStreamNonBlocking

// Event management
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventQuery hipEventQuery
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventDefault hipEventDefault
#define cudaEventDisableTiming hipEventDisableTiming

// Stream capture status
#define cudaStreamCaptureStatusActive hipStreamCaptureStatusActive
#define cudaStreamCaptureStatusNone hipStreamCaptureStatusNone

// Occupancy
#define cudaOccupancyMaxPotentialBlockSize hipOccupancyMaxPotentialBlockSize

// Stream capture / CUDA graphs
#define cudaStreamGetCaptureInfo hipStreamGetCaptureInfo
#define cudaStreamCaptureModeRelaxed hipStreamCaptureModeRelaxed
#define cudaStreamUpdateCaptureDependencies hipStreamUpdateCaptureDependencies
#define cudaThreadExchangeStreamCaptureMode hipThreadExchangeStreamCaptureMode
#define cudaLaunchHostFunc hipLaunchHostFunc
#define cudaUserObjectCreate hipUserObjectCreate
#define cudaGraphRetainUserObject hipGraphRetainUserObject

// Kernel launch
#define cudaLaunchKernel hipLaunchKernel
#define cudaStreamQuery hipStreamQuery

// IPC flags
#define cudaIpcMemLazyEnablePeerAccess hipIpcMemLazyEnablePeerAccess
#define cudaIpcCloseMemHandle hipIpcCloseMemHandle

// User objects
#define cudaUserObjectNoDestructorSync hipUserObjectNoDestructorSync
#define cudaUserObjectRelease hipUserObjectRelease
#define cudaGraphUserObjectMove hipGraphUserObjectMove

// Stream capture v2
#define cudaStreamGetCaptureInfo_v2 hipStreamGetCaptureInfo_v2
#define cudaStreamSetCaptureDependencies hipStreamSetCaptureDependencies

// Graph nodes
#define cudaGraphAddEventRecordNode hipGraphAddEventRecordNode
#define cudaGraphAddDependencies hipGraphAddDependencies
#define cudaGraphEdgeData hipGraphEdgeData

// Misc
#define cudaMemGetInfo hipMemGetInfo
#define cudaStreamIsCapturing hipStreamIsCapturing

// Event flags (used by GPE)
#ifndef cudaEventWaitDefault
#define cudaEventWaitDefault 0x0
#endif
#ifndef cudaEventWaitExternal
#define cudaEventWaitExternal 0x1
#endif
#ifndef cudaEventRecordDefault
#define cudaEventRecordDefault 0x0
#endif
#ifndef cudaEventRecordExternal
#define cudaEventRecordExternal 0x1
#endif

// Kernel attributes
#define cudaFuncSetAttribute hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize

// Device attributes
#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor
#define cudaDevAttrMaxSharedMemoryPerBlockOptin hipDeviceAttributeSharedMemPerBlockOptin

//! Minimal TensorFlow Lite C bindings
//!
//! This module provides targeted bindings for only the TensorFlow Lite functions
//! we actually use, avoiding the complexity of full bindgen generation.

use std::os::raw::{c_char, c_int, c_void};

// =============================================================================
// BASIC TYPES
// =============================================================================

#[repr(C)]
pub struct TfLiteModel {
    _private: [u8; 0],
}

#[repr(C)]
pub struct TfLiteInterpreter {
    _private: [u8; 0],
}

#[repr(C)]
pub struct TfLiteInterpreterOptions {
    _private: [u8; 0],
}

#[repr(C)]
pub struct TfLiteTensor {
    _private: [u8; 0],
}

#[repr(C)]
pub struct TfLiteDelegate {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TfLiteType {
    kTfLiteNoType = 0,
    kTfLiteFloat32 = 1,
    kTfLiteInt32 = 2,
    kTfLiteUInt8 = 3,
    kTfLiteInt64 = 4,
    kTfLiteString = 5,
    kTfLiteBool = 6,
    kTfLiteInt16 = 7,
    kTfLiteComplex64 = 8,
    kTfLiteInt8 = 9,
    kTfLiteFloat16 = 10,
    kTfLiteFloat64 = 11,
    kTfLiteComplex128 = 12,
    kTfLiteUInt64 = 13,
    kTfLiteResource = 14,
    kTfLiteVariant = 15,
    kTfLiteUInt32 = 16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TfLiteStatus {
    kTfLiteOk = 0,
    kTfLiteError = 1,
    kTfLiteDelegateError = 2,
    kTfLiteApplicationError = 3,
    kTfLiteDelegateDataNotFound = 4,
    kTfLiteDelegateDataWriteError = 5,
    kTfLiteDelegateDataReadError = 6,
    kTfLiteUnresolvedOps = 7,
}

// =============================================================================
// XNNPACK FLAGS (from tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h)
// =============================================================================

// Enable XNNPACK acceleration for signed quantized 8-bit inference.
pub const TFLITE_XNNPACK_DELEGATE_FLAG_QS8: u32 = 0x00000001;
// Enable XNNPACK acceleration for unsigned quantized 8-bit inference.
pub const TFLITE_XNNPACK_DELEGATE_FLAG_QU8: u32 = 0x00000002;
// Force FP16 inference for FP32 operators.
pub const TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16: u32 = 0x00000004;
// Enable XNNPACK acceleration for FULLY_CONNECTED operator with dynamic weights.
pub const TFLITE_XNNPACK_DELEGATE_FLAG_DYNAMIC_FULLY_CONNECTED: u32 = 0x00000008;
// Enable XNNPACK acceleration for VAR_HANDLE, READ_VARIABLE, and ASSIGN_VARIABLE operators.
pub const TFLITE_XNNPACK_DELEGATE_FLAG_VARIABLE_OPERATORS: u32 = 0x00000010;
// Enable transient indirection buffer to reduce memory usage in selected operators.
pub const TFLITE_XNNPACK_DELEGATE_FLAG_TRANSIENT_INDIRECTION_BUFFER: u32 = 0x00000020;
// Enable the latest XNNPACK operators and features in the delegate.
pub const TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS: u32 = 0x00000040;
// Enable XNNPack subgraph reshaping for dynamic tensors.
pub const TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_SUBGRAPH_RESHAPING: u32 = 0x00000080;

// =============================================================================
// XNNPACK TYPES (with correct signatures for latest TF Lite)
// =============================================================================

#[repr(C)]
#[derive(Debug, Clone)]
pub struct TfLiteXNNPackDelegateOptions {
    pub num_threads: c_int,
    pub flags: u32,
    pub weights_cache: *mut c_void, // struct TfLiteXNNPackDelegateWeightsCache*
    pub handle_variable_ops: bool,  // bool
    pub weight_cache_file_path: *const c_char, // const char*
}

// =============================================================================
// CORE TENSORFLOW LITE C API
// =============================================================================

extern "C" {
    // Model management
    pub fn TfLiteModelCreateFromFile(model_path: *const c_char) -> *mut TfLiteModel;
    pub fn TfLiteModelCreate(model_data: *const c_void, model_size: usize) -> *mut TfLiteModel;
    pub fn TfLiteModelDelete(model: *mut TfLiteModel);

    // Interpreter options
    pub fn TfLiteInterpreterOptionsCreate() -> *mut TfLiteInterpreterOptions;
    pub fn TfLiteInterpreterOptionsDelete(options: *mut TfLiteInterpreterOptions);
    pub fn TfLiteInterpreterOptionsSetNumThreads(
        options: *mut TfLiteInterpreterOptions,
        num_threads: c_int,
    );

    // Interpreter management
    pub fn TfLiteInterpreterCreate(
        model: *const TfLiteModel,
        optional_options: *const TfLiteInterpreterOptions,
    ) -> *mut TfLiteInterpreter;
    pub fn TfLiteInterpreterDelete(interpreter: *mut TfLiteInterpreter);

    // Tensor operations
    pub fn TfLiteInterpreterAllocateTensors(interpreter: *mut TfLiteInterpreter) -> TfLiteStatus;
    pub fn TfLiteInterpreterInvoke(interpreter: *mut TfLiteInterpreter) -> TfLiteStatus;

    pub fn TfLiteInterpreterGetInputTensor(
        interpreter: *const TfLiteInterpreter,
        input_index: c_int,
    ) -> *mut TfLiteTensor;
    pub fn TfLiteInterpreterGetOutputTensor(
        interpreter: *const TfLiteInterpreter,
        output_index: c_int,
    ) -> *const TfLiteTensor;

    // Tensor count functions
    pub fn TfLiteInterpreterGetInputTensorCount(interpreter: *const TfLiteInterpreter) -> c_int;
    pub fn TfLiteInterpreterGetOutputTensorCount(interpreter: *const TfLiteInterpreter) -> c_int;

    pub fn TfLiteInterpreterResizeInputTensor(
        interpreter: *mut TfLiteInterpreter,
        input_index: c_int,
        input_dims: *const c_int,
        input_dims_size: c_int,
    ) -> TfLiteStatus;

    // Tensor data access
    pub fn TfLiteTensorCopyFromBuffer(
        tensor: *mut TfLiteTensor,
        input_data: *const c_void,
        input_data_size: usize,
    ) -> TfLiteStatus;
    pub fn TfLiteTensorData(tensor: *const TfLiteTensor) -> *mut c_void;
    pub fn TfLiteTensorByteSize(tensor: *const TfLiteTensor) -> usize;

    // Tensor shape info
    pub fn TfLiteTensorNumDims(tensor: *const TfLiteTensor) -> c_int;
    pub fn TfLiteTensorDim(tensor: *const TfLiteTensor, dim_index: c_int) -> c_int;

    // Tensor metadata functions
    pub fn TfLiteTensorName(tensor: *const TfLiteTensor) -> *const c_char;
    pub fn TfLiteTensorType(tensor: *const TfLiteTensor) -> TfLiteType;
    pub fn TfLiteTensorQuantizationParams(tensor: *const TfLiteTensor) -> *const c_void;
}

// =============================================================================
// XNNPACK DELEGATE API (with correct modern signatures)
// =============================================================================

extern "C" {
    // Try the old signature first - return struct by value
    pub fn TfLiteXNNPackDelegateOptionsDefault() -> TfLiteXNNPackDelegateOptions;

    pub fn TfLiteXNNPackDelegateCreate(
        options: *const TfLiteXNNPackDelegateOptions,
    ) -> *mut TfLiteDelegate;
    pub fn TfLiteXNNPackDelegateDelete(delegate: *mut TfLiteDelegate);

    pub fn TfLiteInterpreterOptionsAddDelegate(
        options: *mut TfLiteInterpreterOptions,
        delegate: *mut TfLiteDelegate,
    );
}

// =============================================================================
// SAFE RUST WRAPPERS
// =============================================================================

impl Default for TfLiteXNNPackDelegateOptions {
    fn default() -> Self {
        TfLiteXNNPackDelegateOptions {
            num_threads: 1, // Force exactly 1 thread to avoid futex/threading issues
            // Use default flags that match Python TensorFlow Lite:
            // Enable quantized inference (QS8 + QU8) like Python does
            flags: TFLITE_XNNPACK_DELEGATE_FLAG_QS8 | TFLITE_XNNPACK_DELEGATE_FLAG_QU8,
            weights_cache: std::ptr::null_mut(),
            handle_variable_ops: false,
            weight_cache_file_path: std::ptr::null(),
        }
    }
}

impl TfLiteStatus {
    pub fn is_ok(&self) -> bool {
        *self == TfLiteStatus::kTfLiteOk
    }

    pub fn is_error(&self) -> bool {
        *self != TfLiteStatus::kTfLiteOk
    }
}

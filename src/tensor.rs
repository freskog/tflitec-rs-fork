//! TensorFlow Lite input or output [`Tensor`] associated with an interpreter.
use std::ffi::{c_void, CStr};

use crate::minimal_bindings::*;
use crate::{Error, ErrorKind, Result};
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;

/// Parameters that determine the mapping of quantized values to real values.
///
/// Quantized values can be mapped to float values using the following conversion:
/// `realValue = scale * (quantizedValue - zeroPoint)`.
#[derive(Copy, Clone, PartialEq, Debug, PartialOrd)]
pub struct QuantizationParameters {
    /// The difference between real values corresponding to consecutive quantized
    /// values differing by 1. For example, the range of quantized values for `u8`
    /// data type is [0, 255].
    pub scale: f32,

    /// The quantized value that corresponds to the real 0 value.
    pub zero_point: i32,
}

/// The supported [`Tensor`] data types.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub enum DataType {
    /// A boolean.
    Bool,
    /// An 8-bit unsigned integer.
    Uint8,
    /// An 8-bit signed integer.
    Int8,
    /// A 16-bit signed integer.
    Int16,
    /// A 32-bit signed integer.
    Int32,
    /// A 64-bit signed integer.
    Int64,
    /// A 16-bit half precision floating point.
    Float16,
    /// A 32-bit single precision floating point.
    Float32,
    /// A 64-bit double precision floating point.
    Float64,
}

impl DataType {
    /// Creates a new instance from the given [`Option<TfLiteType>`].
    ///
    /// # Arguments
    ///
    /// * `tflite_type`: A data type for a tensor.
    ///
    /// returns: [`None`] if the data type is unsupported or could not
    /// be determined because there was an error, otherwise returns
    /// [`Some`] corresponding enum variant.
    pub(crate) fn new(tflite_type: TfLiteType) -> Option<DataType> {
        match tflite_type {
            TfLiteType::kTfLiteBool => Some(DataType::Bool),
            TfLiteType::kTfLiteUInt8 => Some(DataType::Uint8),
            TfLiteType::kTfLiteInt8 => Some(DataType::Int8),
            TfLiteType::kTfLiteInt16 => Some(DataType::Int16),
            TfLiteType::kTfLiteInt32 => Some(DataType::Int32),
            TfLiteType::kTfLiteInt64 => Some(DataType::Int64),
            TfLiteType::kTfLiteFloat16 => Some(DataType::Float16),
            TfLiteType::kTfLiteFloat32 => Some(DataType::Float32),
            TfLiteType::kTfLiteFloat64 => Some(DataType::Float64),
            _ => None,
        }
    }
}

#[derive(Clone, Eq, PartialEq, Debug, Hash)]
/// The shape of a [`Tensor`].
pub struct Shape {
    /// The number of dimensions of the [`Tensor`]
    rank: usize,

    /// An array of dimensions for the [`Tensor`]
    dimensions: Vec<usize>,
}

impl Shape {
    /// Creates a new instance with the given `dimensions`.
    ///
    /// # Arguments
    ///
    /// * `dimensions`: Dimensions for the [`Tensor`].
    ///
    /// returns: Shape
    ///
    /// # Examples
    ///
    /// ```
    /// use tflitec::tensor;
    /// let shape = tensor::Shape::new(vec![8, 16, 16]);
    /// assert_eq!(shape.rank(), 3);
    /// assert_eq!(shape.dimensions(), &vec![8, 16, 16]);
    /// ```
    pub fn new(dimensions: Vec<usize>) -> Shape {
        Shape {
            rank: dimensions.len(),
            dimensions,
        }
    }

    /// Returns dimensions of the [`Tensor`].
    pub fn dimensions(&self) -> &Vec<usize> {
        &self.dimensions
    }

    /// Returns rank(number of dimensions) of the [`Tensor`].
    pub fn rank(&self) -> usize {
        self.rank
    }
}

pub(crate) struct TensorData {
    data_ptr: *mut u8,
    data_length: usize,
}

/// An input or output tensor in a TensorFlow Lite graph.
pub struct Tensor<'a> {
    /// The name of the `Tensor`.
    name: String,

    /// The data type of the `Tensor`.
    data_type: DataType,

    /// The shape of the `Tensor`.
    shape: Shape,

    /// The data in the input or output `Tensor`.
    data: TensorData,

    /// The quantization parameters for the `Tensor` if using a quantized model.
    quantization_parameters: Option<QuantizationParameters>,

    /// The underlying [`TfLiteTensor`] C pointer.
    tensor_ptr: *mut TfLiteTensor,

    // To set lifetime of the Tensor
    phantom: PhantomData<&'a TfLiteTensor>,
}

impl Debug for Tensor<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("name", &self.name)
            .field("shape", &self.shape)
            .field("data_type", &self.data_type)
            .field("quantization_parameters", &self.quantization_parameters)
            .finish()
    }
}

impl<'a> Tensor<'a> {
    pub(crate) fn from_raw(tensor_ptr: *mut TfLiteTensor) -> Result<Tensor<'a>> {
        unsafe {
            if tensor_ptr.is_null() {
                return Err(Error::new(ErrorKind::ReadTensorError));
            }

            let name_ptr = TfLiteTensorName(tensor_ptr);
            if name_ptr.is_null() {
                return Err(Error::new(ErrorKind::ReadTensorError));
            }
            let data_ptr = TfLiteTensorData(tensor_ptr) as *mut u8;
            if data_ptr.is_null() {
                return Err(Error::new(ErrorKind::ReadTensorError));
            }
            let name = CStr::from_ptr(name_ptr).to_str().unwrap().to_owned();

            let data_length = TfLiteTensorByteSize(tensor_ptr);
            let data_type = DataType::new(TfLiteTensorType(tensor_ptr))
                .ok_or_else(|| Error::new(ErrorKind::InvalidTensorDataType))?;

            let rank = TfLiteTensorNumDims(tensor_ptr);
            let dimensions = (0..rank)
                .map(|i| TfLiteTensorDim(tensor_ptr, i) as usize)
                .collect();
            let shape = Shape::new(dimensions);
            let data = TensorData {
                data_ptr,
                data_length,
            };
            // Quantization not implemented in minimal bindings - return default
            let scale = 1.0f32;
            let quantization_parameters =
                if scale == 0.0 || (data_type != DataType::Uint8 && data_type != DataType::Int8) {
                    None
                } else {
                    Some(QuantizationParameters {
                        scale: 1.0,
                        zero_point: 0,
                    })
                };
            Ok(Tensor {
                name,
                data_type,
                shape,
                data,
                quantization_parameters,
                tensor_ptr,
                phantom: PhantomData,
            })
        }
    }

    /// Returns [`Shape`] of the tensor
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns data of the tensor as a slice of given type `T`.
    ///
    /// # Panics
    ///
    /// * If number of bytes in buffer of the [`Tensor`] is not integer
    /// multiple of byte count of a single `T` (see [`std::mem::size_of`])
    pub fn data<T>(&self) -> &[T] {
        let element_size = std::mem::size_of::<T>();
        if self.data.data_length % element_size != 0 {
            panic!(
                "data length {} should be divisible by size of type {}",
                self.data.data_length, element_size
            )
        }
        unsafe {
            std::slice::from_raw_parts(
                self.data.data_ptr as *const T,
                self.data.data_length / element_size,
            )
        }
    }

    /// Sets data of the tensor by copying given data slice
    ///
    /// # Arguments
    ///
    /// * `data`: Data to be copied
    ///
    /// # Errors
    ///
    /// Returns error if byte count of the data does not match the buffer size of the
    /// input tensor or TensorFlow Lite C fails internally.
    pub fn set_data<T>(&self, data: &[T]) -> Result<()> {
        let element_size = std::mem::size_of::<T>();
        let input_byte_count = element_size * data.len();
        if self.data.data_length != input_byte_count {
            return Err(Error::new(ErrorKind::InvalidTensorDataCount(
                data.len(),
                input_byte_count,
            )));
        }
        let status = unsafe {
            TfLiteTensorCopyFromBuffer(
                self.tensor_ptr,
                data.as_ptr() as *const c_void,
                input_byte_count,
            )
        };
        if status != TfLiteStatus::kTfLiteOk {
            Err(Error::new(ErrorKind::FailedToCopyDataToInputTensor))
        } else {
            Ok(())
        }
    }

    /// Returns [data type][`DataType`] of the [`Tensor`].
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    /// Returns optional [`QuantizationParameters`] of the [`Tensor`].
    pub fn quantization_parameters(&self) -> Option<QuantizationParameters> {
        self.quantization_parameters
    }

    /// Returns name of the [`Tensor`].
    pub fn name(&self) -> &str {
        self.name.as_str()
    }
}

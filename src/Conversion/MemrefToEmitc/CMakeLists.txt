add_onnx_mlir_library(OMMemrefToEmitc
  ConvertMemrefToEmitc.cpp
  MemrefAlloc.cpp

  LINK_LIBS PUBLIC
  OMSpecializedKernelOpInterface
  OMSupport
  MLIRTransforms
  MLIRAffineUtils
  )
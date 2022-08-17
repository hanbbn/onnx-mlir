#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

namespace onnx_mlir {
namespace memref{
class MemrefReinterpretCastOpLowering : public ConversionPattern{
    MemrefReinterpretCastOpLowering(MLIRContext *ctx)
        : ConversionPattern(memref::ReinterpretCastOp::getOperationName(), 1, ctx){}
    
    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const override {
        rewriter.eraseOp(op);
        return success();
    }
}


void populateLoweringMemrefReinterpretCastOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
    patterns.insert<MemrefReinterpretCastOpLowering>(typeConverter, ctx);
}


}
}
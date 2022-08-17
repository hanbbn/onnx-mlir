#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

namespace onnx_mlir {
namespace Memref{
class MemrefLoadOpLowering : public ConversionPattern{
    MemrefLoadOpLowering(MLIRContext *ctx)
        : ConversionPattern(memref::LoadOp::getOperationName(), 1, ctx){}
    
    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const override {
        auto loc = op->getLoc();
        auto memrefAllocOp = cast<memref::LoadOp>(op);
        Value loadResult = rewriter.create<emitc::callOp>(loc, emitc::PointerType(builder.getF32Type()), "memref_load");
        rewriter.replaceOp(op, {loadResult});
        return success();
    }
}


void populateLoweringMemrefLoadOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
    patterns.insert<MemrefLoadOpLowering>(typeConverter, ctx);
}


}
}
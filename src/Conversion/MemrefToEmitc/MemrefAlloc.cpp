#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

namespace onnx_mlir {
namespace Memref{
class MemrefAllocOpLowering : public ConversionPattern{
    MemrefAllocOpLowering(MLIRContext *ctx)
        : ConversionPattern(MemrefAllocOp::getOperationName(), 1, ctx){}
    
    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const override {
        auto loc = op->getLoc();
        auto memrefAllocOp = cast<MemrefAllocOp>(op);
        Value allocResult = rewriter.create<emitc::callOp>(loc, emitc::PointerType(builder.getF32Type()), "alloc");
        rewriter.replaceOp(op, {allocResult});
        return success();
    }
}


void populateLoweringMemrefAllocOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
    patterns.insert<MemrefAllocOpLowering>(typeConverter, ctx);
}


}
}
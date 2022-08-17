#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

namespace onnx_mlir {
namespace memref{
class MemrefStoreOpLowering : public ConversionPattern{
    MemrefStoreOpLowering(MLIRContext *ctx)
        : ConversionPattern(memref::StoreOp::getOperationName(), 1, ctx){}
    
    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const override {
        auto loc = op->getLoc();
        auto memrefStoreOp = cast<memref::StoreOp>(op);
        Value storeResult = rewriter.create<emitc::callOp>(loc, emitc::PointerType(builder.getF32Type()), "memref_store");
        rewriter.replaceOp(op, {storeResult});
        return success();
    }
}


void populateLoweringMemrefStoreOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
    patterns.insert<MemrefStoreOpLowering>(typeConverter, ctx);
}


}
}
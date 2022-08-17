#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

#include "src/Conversion/MemrefToEmitc/ConvertMemrefToEmitc.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/Common.hpp"

using namespace mlir;

namespace onnx-mlir{

struct ConvertMemrefToEmitcPass
    : public PassWrapper<ConvertMemrefToEmitcPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertMemrefToEmitcPass);

    StringRef getArgument() const override { return "convert-Memref-to-Emitc"; }

    StringRef getDescription() const override { return "Lower Memref dialect."; }

    void runOnOperation() final;
};

void ConvertMemrefToEmitcPass::runOnOperation(){
    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);

    TypeConverter typeConverter(ctx);
    typeConverter.addConversion([&](MemRefType memref_type) -> Type {
        return emitc::PointerType(builder.getF32Type());
    });

    ConversionTarget target(*ctx);
    target.addLegalDialect<emitc::EmitCDialect>();
    target.addLegalDialect<arith::ArithmeticDialect>();
    target.addIlegalDialect<AffineDialect, memref::MemRefDialect>();
    target.addIllegalOp<arith::IndexCastOp>();

    RewritePatternSet patterns(&getContext());

    if(failed(applyPartialConversion(getOperation(), target, patterns)))
        signalPassFailure();
}

std::unique_ptr<Pass> createConvertMemrefToEmitcPass() {
  return std::make_unique<ConvertMemrefToEmitcPass>();
}

void populateMemrefToEmitcConversion(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  krnl::populateLoweringKrnlCopyFromBufferOpPattern(
      typeConverter, patterns, ctx);
  Memref::populateLoweringMemrefAllocOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlLoadOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlStoreOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlMatmultOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlMemsetOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlTerminatorOpPattern(typeConverter, patterns, ctx);
}

}



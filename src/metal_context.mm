// ─────────────────────────────────────────────────────────────────────
// metal_context.mm
// Metal GPU context implementation for PhysiCell
// ─────────────────────────────────────────────────────────────────────

#import "metal_context.h"
#include <cstdio>
#include <cstdlib>

// ─────────────────────────────────────────────────────────────────────
// Constructor: Initialize Metal device and command queue
// ─────────────────────────────────────────────────────────────────────
MetalContext::MetalContext()
    : device(nil), commandQueue(nil), library(nil),
      diffusionXPipeline(nil), diffusionYPipeline(nil),
      clearHashPipeline(nil), buildHashPipeline(nil),
      forcesPipeline(nil), integratePipeline(nil),
      lastCommandBuffer(nil)
{
    // Get the default Metal device (the system GPU)
    device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "[MetalContext] ERROR: No Metal device found.\n");
        abort();
    }

    fprintf(stderr, "[MetalContext] Using device: %s\n",
            [[device name] UTF8String]);

    // Create command queue
    commandQueue = [device newCommandQueue];
    if (!commandQueue) {
        fprintf(stderr, "[MetalContext] ERROR: Failed to create command queue.\n");
        abort();
    }
}

// ─────────────────────────────────────────────────────────────────────
// Destructor
// ─────────────────────────────────────────────────────────────────────
MetalContext::~MetalContext() {
    // ARC handles release of Objective-C objects
    // Just wait for any in-flight work
    if (lastCommandBuffer) {
        [lastCommandBuffer waitUntilCompleted];
    }
}

// ─────────────────────────────────────────────────────────────────────
// Buffer allocation — MTLStorageModeShared for Apple Silicon UMA
// ─────────────────────────────────────────────────────────────────────
id<MTLBuffer> MetalContext::allocateBuffer(size_t bytes) {
    id<MTLBuffer> buffer = [device newBufferWithLength:bytes
                                              options:MTLResourceStorageModeShared];
    if (!buffer) {
        fprintf(stderr, "[MetalContext] ERROR: Failed to allocate buffer of %zu bytes.\n", bytes);
        abort();
    }
    return buffer;
}

// ─────────────────────────────────────────────────────────────────────
// Load shaders — try precompiled .metallib first, fall back to runtime
// compilation from .metal source files
// ─────────────────────────────────────────────────────────────────────
void MetalContext::loadShaders(const char* metallib_path) {
    NSError* error = nil;

    // Try precompiled metallib first
    NSString* path = [NSString stringWithUTF8String:metallib_path];
    if ([[NSFileManager defaultManager] fileExistsAtPath:path]) {
        NSURL* url = [NSURL fileURLWithPath:path];
        library = [device newLibraryWithURL:url error:&error];
        if (library) {
            fprintf(stderr, "[MetalContext] Loaded precompiled shader library: %s\n", metallib_path);
            goto create_pipelines;
        }
        fprintf(stderr, "[MetalContext] WARNING: Failed to load metallib, falling back to source compilation\n");
    }

    // Fall back: compile from .metal source files at runtime
    {
        fprintf(stderr, "[MetalContext] Compiling shaders from source...\n");

        // Read all shader source files
        NSMutableString* allSource = [NSMutableString string];

        // First include the types header
        NSString* typesPath = @"shaders/types.h";
        NSString* typesSource = [NSString stringWithContentsOfFile:typesPath
                                                          encoding:NSUTF8StringEncoding
                                                             error:&error];
        if (!typesSource) {
            fprintf(stderr, "[MetalContext] ERROR: Cannot read %s: %s\n",
                    [typesPath UTF8String], [[error localizedDescription] UTF8String]);
            abort();
        }

        // Read each .metal file
        NSArray* shaderFiles = @[@"shaders/diffusion_2d.metal",
                                  @"shaders/mechanics.metal",
                                  @"shaders/integrate.metal"];

        for (NSString* file in shaderFiles) {
            NSString* source = [NSString stringWithContentsOfFile:file
                                                         encoding:NSUTF8StringEncoding
                                                            error:&error];
            if (!source) {
                fprintf(stderr, "[MetalContext] ERROR: Cannot read %s: %s\n",
                        [file UTF8String], [[error localizedDescription] UTF8String]);
                abort();
            }
            // Strip #include "types.h" since we'll prepend the types inline
            source = [source stringByReplacingOccurrencesOfString:@"#include \"types.h\""
                                                       withString:@"// types.h included inline"];
            [allSource appendString:source];
            [allSource appendString:@"\n"];
            fprintf(stderr, "[MetalContext]   Read %s\n", [file UTF8String]);
        }

        // Prepend types (strip the C++ guards and non-Metal stuff)
        NSMutableString* metalTypes = [NSMutableString string];
        [metalTypes appendString:@"#include <metal_stdlib>\nusing namespace metal;\n\n"];
        // Extract Metal-compatible parts from types.h
        // The types.h already handles this with __METAL_VERSION__ guards,
        // so we set the define manually
        NSString* patchedTypes = [typesSource stringByReplacingOccurrencesOfString:@"#ifndef PHYSICELL_METAL_TYPES_H"
                                                                       withString:@"// types included inline"];
        patchedTypes = [patchedTypes stringByReplacingOccurrencesOfString:@"#define PHYSICELL_METAL_TYPES_H"
                                                              withString:@""];
        patchedTypes = [patchedTypes stringByReplacingOccurrencesOfString:@"#endif // PHYSICELL_METAL_TYPES_H"
                                                              withString:@""];
        patchedTypes = [patchedTypes stringByReplacingOccurrencesOfString:@"#include <cstdint>"
                                                              withString:@""];

        NSString* fullSource = [NSString stringWithFormat:@"%@\n%@", patchedTypes, allSource];

        // Compile
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        opts.languageVersion = MTLLanguageVersion3_0;
        opts.mathMode = MTLMathModeFast;
        // Define __METAL_VERSION__ so types.h selects Metal path
        opts.preprocessorMacros = @{@"__METAL_VERSION__": @300};

        library = [device newLibraryWithSource:fullSource options:opts error:&error];
        if (!library) {
            fprintf(stderr, "[MetalContext] ERROR: Shader compilation failed:\n%s\n",
                    [[error localizedDescription] UTF8String]);
            // Print first part of source for debugging
            NSString* preview = [fullSource substringToIndex:MIN(2000u, [fullSource length])];
            fprintf(stderr, "--- Source preview ---\n%s\n--- End preview ---\n",
                    [preview UTF8String]);
            abort();
        }
        fprintf(stderr, "[MetalContext] Shaders compiled successfully at runtime\n");
    }

create_pipelines:
    // Create all compute pipelines
    diffusionXPipeline  = createPipeline("diffusion_sweep_x");
    diffusionYPipeline  = createPipeline("diffusion_sweep_y");
    clearHashPipeline   = createPipeline("clear_spatial_hash");
    buildHashPipeline   = createPipeline("build_spatial_hash");
    forcesPipeline      = createPipeline("compute_forces");
    integratePipeline   = createPipeline("integrate_positions");

    fprintf(stderr, "[MetalContext] All compute pipelines created.\n");
}

// ─────────────────────────────────────────────────────────────────────
// Create a compute pipeline from a named function
// ─────────────────────────────────────────────────────────────────────
id<MTLComputePipelineState> MetalContext::createPipeline(const char* functionName) {
    NSString* name = [NSString stringWithUTF8String:functionName];
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (!function) {
        fprintf(stderr, "[MetalContext] ERROR: Shader function '%s' not found in library.\n",
                functionName);
        abort();
    }

    NSError* error = nil;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
        fprintf(stderr, "[MetalContext] ERROR: Failed to create pipeline for '%s': %s\n",
                functionName, [[error localizedDescription] UTF8String]);
        abort();
    }

    fprintf(stderr, "[MetalContext]   Pipeline '%s': maxTotalThreadsPerThreadgroup = %lu\n",
            functionName, (unsigned long)[pipeline maxTotalThreadsPerThreadgroup]);

    return pipeline;
}

// ─────────────────────────────────────────────────────────────────────
// Calculate optimal 1D threadgroup size
// ─────────────────────────────────────────────────────────────────────
MTLSize MetalContext::optimalThreadgroupSize(id<MTLComputePipelineState> pipeline,
                                              uint32_t totalThreads) {
    NSUInteger maxThreads = [pipeline maxTotalThreadsPerThreadgroup];
    // Use thread execution width (SIMD width) as the base unit (typically 32 on Apple Silicon)
    NSUInteger threadWidth = [pipeline threadExecutionWidth];

    // Choose threadgroup size as a multiple of SIMD width, up to maxThreads
    NSUInteger tgSize = threadWidth;
    while (tgSize * 2 <= maxThreads && tgSize * 2 <= 1024) {
        tgSize *= 2;
    }

    // Don't exceed total threads needed
    if (tgSize > totalThreads) {
        tgSize = totalThreads;
    }

    return MTLSizeMake(tgSize, 1, 1);
}

// ─────────────────────────────────────────────────────────────────────
// Dispatch: Diffusion X-sweep (1 thread per row)
// ─────────────────────────────────────────────────────────────────────
void MetalContext::dispatchDiffusionX(id<MTLBuffer> density,
                                       id<MTLBuffer> gridParams,
                                       id<MTLBuffer> substrateParams,
                                       id<MTLBuffer> thomasCoeffs,
                                       uint32_t ny) {
    id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

    [encoder setComputePipelineState:diffusionXPipeline];
    [encoder setBuffer:density         offset:0 atIndex:0];
    [encoder setBuffer:gridParams      offset:0 atIndex:1];
    [encoder setBuffer:substrateParams offset:0 atIndex:2];
    [encoder setBuffer:thomasCoeffs    offset:0 atIndex:3];

    MTLSize gridSize = MTLSizeMake(ny, 1, 1);
    MTLSize tgSize = optimalThreadgroupSize(diffusionXPipeline, ny);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
    [encoder endEncoding];
    [cmdBuffer commit];

    lastCommandBuffer = cmdBuffer;
}

// ─────────────────────────────────────────────────────────────────────
// Dispatch: Diffusion Y-sweep (1 thread per column)
// ─────────────────────────────────────────────────────────────────────
void MetalContext::dispatchDiffusionY(id<MTLBuffer> density,
                                       id<MTLBuffer> gridParams,
                                       id<MTLBuffer> substrateParams,
                                       id<MTLBuffer> thomasCoeffs,
                                       uint32_t nx) {
    id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

    [encoder setComputePipelineState:diffusionYPipeline];
    [encoder setBuffer:density         offset:0 atIndex:0];
    [encoder setBuffer:gridParams      offset:0 atIndex:1];
    [encoder setBuffer:substrateParams offset:0 atIndex:2];
    [encoder setBuffer:thomasCoeffs    offset:0 atIndex:3];

    MTLSize gridSize = MTLSizeMake(nx, 1, 1);
    MTLSize tgSize = optimalThreadgroupSize(diffusionYPipeline, nx);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
    [encoder endEncoding];
    [cmdBuffer commit];

    lastCommandBuffer = cmdBuffer;
}

// ─────────────────────────────────────────────────────────────────────
// Dispatch: Clear spatial hash (1 thread per voxel)
// ─────────────────────────────────────────────────────────────────────
void MetalContext::dispatchClearHash(id<MTLBuffer> hashCounts, uint32_t numVoxels) {
    id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

    [encoder setComputePipelineState:clearHashPipeline];
    [encoder setBuffer:hashCounts offset:0 atIndex:0];
    [encoder setBytes:&numVoxels  length:sizeof(uint32_t) atIndex:1];

    MTLSize gridSize = MTLSizeMake(numVoxels, 1, 1);
    MTLSize tgSize = optimalThreadgroupSize(clearHashPipeline, numVoxels);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
    [encoder endEncoding];
    [cmdBuffer commit];

    lastCommandBuffer = cmdBuffer;
}

// ─────────────────────────────────────────────────────────────────────
// Dispatch: Build spatial hash (1 thread per cell)
// ─────────────────────────────────────────────────────────────────────
void MetalContext::dispatchBuildHash(id<MTLBuffer> cells,
                                      id<MTLBuffer> hashCounts,
                                      id<MTLBuffer> hashCells,
                                      id<MTLBuffer> mechParams,
                                      id<MTLBuffer> gridParams,
                                      uint32_t numCells,
                                      uint32_t maxCells) {
    id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

    [encoder setComputePipelineState:buildHashPipeline];
    [encoder setBuffer:cells      offset:0 atIndex:0];
    [encoder setBuffer:hashCounts offset:0 atIndex:1];
    [encoder setBuffer:hashCells  offset:0 atIndex:2];
    [encoder setBuffer:mechParams offset:0 atIndex:3];
    [encoder setBuffer:gridParams offset:0 atIndex:4];
    [encoder setBytes:&numCells   length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&maxCells   length:sizeof(uint32_t) atIndex:6];

    MTLSize gridSize = MTLSizeMake(numCells, 1, 1);
    MTLSize tgSize = optimalThreadgroupSize(buildHashPipeline, numCells);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
    [encoder endEncoding];
    [cmdBuffer commit];

    lastCommandBuffer = cmdBuffer;
}

// ─────────────────────────────────────────────────────────────────────
// Dispatch: Compute forces (1 thread per cell)
// ─────────────────────────────────────────────────────────────────────
void MetalContext::dispatchForces(id<MTLBuffer> cells,
                                   id<MTLBuffer> hashCounts,
                                   id<MTLBuffer> hashCells,
                                   id<MTLBuffer> mechParams,
                                   uint32_t numCells,
                                   uint32_t maxCells) {
    id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

    [encoder setComputePipelineState:forcesPipeline];
    [encoder setBuffer:cells      offset:0 atIndex:0];
    [encoder setBuffer:hashCounts offset:0 atIndex:1];
    [encoder setBuffer:hashCells  offset:0 atIndex:2];
    [encoder setBuffer:mechParams offset:0 atIndex:3];
    [encoder setBytes:&numCells   length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&maxCells   length:sizeof(uint32_t) atIndex:5];

    MTLSize gridSize = MTLSizeMake(numCells, 1, 1);
    MTLSize tgSize = optimalThreadgroupSize(forcesPipeline, numCells);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
    [encoder endEncoding];
    [cmdBuffer commit];

    lastCommandBuffer = cmdBuffer;
}

// ─────────────────────────────────────────────────────────────────────
// Dispatch: Integrate positions (1 thread per cell)
// ─────────────────────────────────────────────────────────────────────
void MetalContext::dispatchIntegrate(id<MTLBuffer> cells,
                                      id<MTLBuffer> gridParams,
                                      float dt,
                                      uint32_t numCells,
                                      uint32_t maxCells) {
    id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

    [encoder setComputePipelineState:integratePipeline];
    [encoder setBuffer:cells      offset:0 atIndex:0];
    [encoder setBuffer:gridParams offset:0 atIndex:1];
    [encoder setBytes:&dt         length:sizeof(float)    atIndex:2];
    [encoder setBytes:&numCells   length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&maxCells   length:sizeof(uint32_t) atIndex:4];

    MTLSize gridSize = MTLSizeMake(numCells, 1, 1);
    MTLSize tgSize = optimalThreadgroupSize(integratePipeline, numCells);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
    [encoder endEncoding];
    [cmdBuffer commit];

    lastCommandBuffer = cmdBuffer;
}

// ─────────────────────────────────────────────────────────────────────
// Forces-only pipeline: clear_hash → build_hash → compute_forces
// Does NOT run integrate_positions, allowing CPU motility to be added
// to velocity before the caller dispatches dispatchIntegrate.
// ─────────────────────────────────────────────────────────────────────
void MetalContext::dispatchForcesOnlyPipeline(id<MTLBuffer> cells,
                                               id<MTLBuffer> hashCounts,
                                               id<MTLBuffer> hashCells,
                                               id<MTLBuffer> mechParams,
                                               id<MTLBuffer> gridParams,
                                               uint32_t numCells,
                                               uint32_t maxCells,
                                               uint32_t numMechVoxels) {
    id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

    // ─── 1. Clear spatial hash ───
    [encoder setComputePipelineState:clearHashPipeline];
    [encoder setBuffer:hashCounts offset:0 atIndex:0];
    [encoder setBytes:&numMechVoxels length:sizeof(uint32_t) atIndex:1];

    MTLSize gridSize = MTLSizeMake(numMechVoxels, 1, 1);
    MTLSize tgSize = optimalThreadgroupSize(clearHashPipeline, numMechVoxels);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // ─── 2. Build spatial hash ───
    [encoder setComputePipelineState:buildHashPipeline];
    [encoder setBuffer:cells      offset:0 atIndex:0];
    [encoder setBuffer:hashCounts offset:0 atIndex:1];
    [encoder setBuffer:hashCells  offset:0 atIndex:2];
    [encoder setBuffer:mechParams offset:0 atIndex:3];
    [encoder setBuffer:gridParams offset:0 atIndex:4];
    [encoder setBytes:&numCells   length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&maxCells   length:sizeof(uint32_t) atIndex:6];

    gridSize = MTLSizeMake(numCells, 1, 1);
    tgSize = optimalThreadgroupSize(buildHashPipeline, numCells);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // ─── 3. Compute forces ───
    [encoder setComputePipelineState:forcesPipeline];
    [encoder setBuffer:cells      offset:0 atIndex:0];
    [encoder setBuffer:hashCounts offset:0 atIndex:1];
    [encoder setBuffer:hashCells  offset:0 atIndex:2];
    [encoder setBuffer:mechParams offset:0 atIndex:3];
    [encoder setBytes:&numCells   length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&maxCells   length:sizeof(uint32_t) atIndex:5];

    gridSize = MTLSizeMake(numCells, 1, 1);
    tgSize = optimalThreadgroupSize(forcesPipeline, numCells);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];

    [encoder endEncoding];
    [cmdBuffer commit];

    lastCommandBuffer = cmdBuffer;
}

// ─────────────────────────────────────────────────────────────────────
// Wait for all GPU work to complete
// ─────────────────────────────────────────────────────────────────────
void MetalContext::waitForCompletion() {
    if (lastCommandBuffer) {
        [lastCommandBuffer waitUntilCompleted];

        // Check for errors
        if ([lastCommandBuffer status] == MTLCommandBufferStatusError) {
            NSError* error = [lastCommandBuffer error];
            fprintf(stderr, "[MetalContext] ERROR: Command buffer failed: %s\n",
                    [[error localizedDescription] UTF8String]);
        }

        lastCommandBuffer = nil;
    }
}

// ─────────────────────────────────────────────────────────────────────
// Unified mechanics pipeline — single command buffer with barriers
// Encodes: clear_hash → barrier → build_hash → barrier → forces → barrier → integrate
// This guarantees correct execution order on the GPU.
// ─────────────────────────────────────────────────────────────────────
void MetalContext::dispatchMechanicsPipeline(id<MTLBuffer> cells,
                                              id<MTLBuffer> hashCounts,
                                              id<MTLBuffer> hashCells,
                                              id<MTLBuffer> mechParams,
                                              id<MTLBuffer> gridParams,
                                              float dt,
                                              uint32_t numCells,
                                              uint32_t maxCells,
                                              uint32_t numMechVoxels) {
    id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

    // ─── 1. Clear spatial hash ───
    [encoder setComputePipelineState:clearHashPipeline];
    [encoder setBuffer:hashCounts offset:0 atIndex:0];
    [encoder setBytes:&numMechVoxels length:sizeof(uint32_t) atIndex:1];

    MTLSize gridSize = MTLSizeMake(numMechVoxels, 1, 1);
    MTLSize tgSize = optimalThreadgroupSize(clearHashPipeline, numMechVoxels);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];

    // Memory barrier: ensure clear completes before build reads
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // ─── 2. Build spatial hash ───
    [encoder setComputePipelineState:buildHashPipeline];
    [encoder setBuffer:cells      offset:0 atIndex:0];
    [encoder setBuffer:hashCounts offset:0 atIndex:1];
    [encoder setBuffer:hashCells  offset:0 atIndex:2];
    [encoder setBuffer:mechParams offset:0 atIndex:3];
    [encoder setBuffer:gridParams offset:0 atIndex:4];
    [encoder setBytes:&numCells   length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&maxCells   length:sizeof(uint32_t) atIndex:6];

    gridSize = MTLSizeMake(numCells, 1, 1);
    tgSize = optimalThreadgroupSize(buildHashPipeline, numCells);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];

    // Memory barrier: ensure hash is fully built before force computation reads it
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // ─── 3. Compute forces ───
    [encoder setComputePipelineState:forcesPipeline];
    [encoder setBuffer:cells      offset:0 atIndex:0];
    [encoder setBuffer:hashCounts offset:0 atIndex:1];
    [encoder setBuffer:hashCells  offset:0 atIndex:2];
    [encoder setBuffer:mechParams offset:0 atIndex:3];
    [encoder setBytes:&numCells   length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&maxCells   length:sizeof(uint32_t) atIndex:5];

    gridSize = MTLSizeMake(numCells, 1, 1);
    tgSize = optimalThreadgroupSize(forcesPipeline, numCells);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];

    // Memory barrier: ensure forces are written before integration reads them
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // ─── 4. Integrate positions ───
    [encoder setComputePipelineState:integratePipeline];
    [encoder setBuffer:cells      offset:0 atIndex:0];
    [encoder setBuffer:gridParams offset:0 atIndex:1];
    [encoder setBytes:&dt         length:sizeof(float)    atIndex:2];
    [encoder setBytes:&numCells   length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&maxCells   length:sizeof(uint32_t) atIndex:4];

    gridSize = MTLSizeMake(numCells, 1, 1);
    tgSize = optimalThreadgroupSize(integratePipeline, numCells);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];

    [encoder endEncoding];
    [cmdBuffer commit];

    lastCommandBuffer = cmdBuffer;
}

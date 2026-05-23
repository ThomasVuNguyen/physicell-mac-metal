// ─────────────────────────────────────────────────────────────────────
// metal_context.h
// Metal GPU context manager for PhysiCell
// ─────────────────────────────────────────────────────────────────────

#ifndef PHYSICELL_METAL_CONTEXT_H
#define PHYSICELL_METAL_CONTEXT_H

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "../shaders/types.h"

class MetalContext {
public:
    MetalContext();
    ~MetalContext();

    // ─── Buffer allocation (MTLStorageModeShared for UMA) ───
    id<MTLBuffer> allocateBuffer(size_t bytes);

    // ─── Shader loading ───
    void loadShaders(const char* metallib_path);

    // ─── Diffusion dispatch ───
    void dispatchDiffusionX(id<MTLBuffer> density, id<MTLBuffer> gridParams,
                            id<MTLBuffer> substrateParams, id<MTLBuffer> thomasCoeffs,
                            uint32_t ny);

    void dispatchDiffusionY(id<MTLBuffer> density, id<MTLBuffer> gridParams,
                            id<MTLBuffer> substrateParams, id<MTLBuffer> thomasCoeffs,
                            uint32_t nx);

    // ─── Mechanics dispatch ───
    void dispatchClearHash(id<MTLBuffer> hashCounts, uint32_t numVoxels);

    void dispatchBuildHash(id<MTLBuffer> cells, id<MTLBuffer> hashCounts,
                           id<MTLBuffer> hashCells, id<MTLBuffer> mechParams,
                           id<MTLBuffer> gridParams,
                           uint32_t numCells, uint32_t maxCells);

    void dispatchForces(id<MTLBuffer> cells, id<MTLBuffer> hashCounts,
                        id<MTLBuffer> hashCells, id<MTLBuffer> mechParams,
                        uint32_t numCells, uint32_t maxCells);

    // ─── Integration dispatch ───
    void dispatchIntegrate(id<MTLBuffer> cells, id<MTLBuffer> gridParams,
                           float dt, uint32_t numCells, uint32_t maxCells);

    // ─── Unified mechanics pipeline (single command buffer with barriers) ───
    void dispatchMechanicsPipeline(id<MTLBuffer> cells,
                                   id<MTLBuffer> hashCounts,
                                   id<MTLBuffer> hashCells,
                                   id<MTLBuffer> mechParams,
                                   id<MTLBuffer> gridParams,
                                   float dt,
                                   uint32_t numCells,
                                   uint32_t maxCells,
                                   uint32_t numMechVoxels);

    // ─── Split mechanics pipeline: forces-only (no integrate) ───
    // Use this when CPU motility must run between forces and integrate_positions.
    void dispatchForcesOnlyPipeline(id<MTLBuffer> cells,
                                     id<MTLBuffer> hashCounts,
                                     id<MTLBuffer> hashCells,
                                     id<MTLBuffer> mechParams,
                                     id<MTLBuffer> gridParams,
                                     uint32_t numCells,
                                     uint32_t maxCells,
                                     uint32_t numMechVoxels);

    // ─── Synchronization ───
    void waitForCompletion();

    // ─── Accessors ───
    id<MTLDevice> getDevice() const { return device; }
    id<MTLCommandQueue> getCommandQueue() const { return commandQueue; }

private:
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLLibrary> library;

    // Compute pipelines
    id<MTLComputePipelineState> diffusionXPipeline;
    id<MTLComputePipelineState> diffusionYPipeline;
    id<MTLComputePipelineState> clearHashPipeline;
    id<MTLComputePipelineState> buildHashPipeline;
    id<MTLComputePipelineState> forcesPipeline;
    id<MTLComputePipelineState> integratePipeline;

    // Last command buffer for synchronization
    id<MTLCommandBuffer> lastCommandBuffer;

    // Helper to create a pipeline from a function name
    id<MTLComputePipelineState> createPipeline(const char* functionName);

    // Helper to calculate optimal threadgroup size
    MTLSize optimalThreadgroupSize(id<MTLComputePipelineState> pipeline, uint32_t totalThreads);
};

#endif // PHYSICELL_METAL_CONTEXT_H

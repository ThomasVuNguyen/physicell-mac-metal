#ifndef PHYSICELL_OUTPUT_WRITER_H
#define PHYSICELL_OUTPUT_WRITER_H

// ─────────────────────────────────────────────────────────────────────
// OutputWriter — writes simulation frames as JSON for the 3D viewer
// ─────────────────────────────────────────────────────────────────────

#include <string>
#include <vector>
#include "cell_data.h"

class OutputWriter {
public:
    /// Create a writer that outputs to `folder`.
    /// The folder will be created if it doesn't exist.
    explicit OutputWriter(const std::string& folder);

    /// Write a single frame JSON file:
    /// {"time": T, "num_cells": N, "cells": [{x,y,z,radius,type,oncoprotein}, ...]}
    void writeFrame(const CellData& cells, double time, int frame_index);

    /// Write a combined multi-frame JSON:
    /// {"frames": [{frame0}, {frame1}, ...]}
    /// Call at the end of the simulation.
    void writeAllFrames();

    /// Get the output folder path.
    const std::string& getFolder() const { return folder_; }

    /// Get the number of frames written so far.
    int getFrameCount() const { return static_cast<int>(frame_files_.size()); }

private:
    std::string folder_;
    std::vector<std::string> frame_files_;  // paths to individual frame JSONs
};

#endif // PHYSICELL_OUTPUT_WRITER_H

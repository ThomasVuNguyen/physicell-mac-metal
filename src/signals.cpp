// ─────────────────────────────────────────────────────────────────────
// Signal System — Implementation
//
// Reads cell state and microenvironment to produce signal values.
// Port of PhysiCell's get_single_signal / setup_signal_behavior_dictionaries
// to the SoA architecture.
//
// Reference: PhysiCell core/PhysiCell_signal_behavior.cpp
// ─────────────────────────────────────────────────────────────────────

#include "signals.h"
#include "cell_data.h"
#include "../shaders/types.h"

#include <iostream>
#include <cmath>

namespace PhysiCellMetal {

// ─── Module-level state ──────────────────────────────────────────────

static std::map<std::string, int> signal_to_int;
static std::map<int, std::string> int_to_signal;
static int n_substrates_cached = 0;
static bool dictionary_initialized = false;

// ─── Dictionary construction ─────────────────────────────────────────

void initSignalDictionary(int n_substrates,
                          const std::vector<std::string>& substrate_names)
{
    signal_to_int.clear();
    int_to_signal.clear();
    n_substrates_cached = n_substrates;

    int idx = 0;

    // [0 .. m-1] substrate concentrations
    for (int i = 0; i < n_substrates; i++) {
        std::string name = substrate_names[i];
        signal_to_int[name] = idx;
        int_to_signal[idx] = name;
        idx++;
    }

    // [m .. 2m-1] substrate gradient magnitudes
    for (int i = 0; i < n_substrates; i++) {
        std::string name = substrate_names[i] + " gradient";
        signal_to_int[name] = idx;
        int_to_signal[idx] = name;

        // synonym
        signal_to_int["grad(" + substrate_names[i] + ")"] = idx;
        idx++;
    }

    // pressure
    signal_to_int["pressure"] = idx;
    int_to_signal[idx] = "pressure";
    idx++;

    // volume
    signal_to_int["volume"] = idx;
    int_to_signal[idx] = "volume";
    idx++;

    // contact with live cells
    signal_to_int["contact with live cell"] = idx;
    int_to_signal[idx] = "contact with live cell";
    signal_to_int["contact with live cells"] = idx;
    idx++;

    // contact with dead cells
    signal_to_int["contact with dead cell"] = idx;
    int_to_signal[idx] = "contact with dead cell";
    signal_to_int["contact with dead cells"] = idx;
    idx++;

    // damage (placeholder)
    signal_to_int["damage"] = idx;
    int_to_signal[idx] = "damage";
    idx++;

    // dead state
    signal_to_int["dead"] = idx;
    int_to_signal[idx] = "dead";
    signal_to_int["is dead"] = idx;
    idx++;

    // time
    signal_to_int["time"] = idx;
    int_to_signal[idx] = "time";
    signal_to_int["current time"] = idx;
    signal_to_int["global time"] = idx;
    idx++;

    // custom: oncoprotein
    signal_to_int["custom:oncoprotein"] = idx;
    int_to_signal[idx] = "custom:oncoprotein";
    signal_to_int["custom: oncoprotein"] = idx;
    signal_to_int["oncoprotein"] = idx;
    idx++;

    dictionary_initialized = true;
}

int findSignalIndex(const std::string& name) {
    auto it = signal_to_int.find(name);
    if (it != signal_to_int.end()) {
        return it->second;
    }
    std::cerr << "Warning: signal '" << name << "' not found in dictionary."
              << std::endl;
    return -1;
}

std::string signalName(int index) {
    auto it = int_to_signal.find(index);
    if (it != int_to_signal.end()) {
        return it->second;
    }
    return "unknown";
}

int numSignals() {
    return static_cast<int>(int_to_signal.size());
}

void displaySignalDictionary() {
    std::cout << "Signals:" << std::endl;
    std::cout << "========" << std::endl;
    for (auto& [idx, name] : int_to_signal) {
        std::cout << "  " << idx << " : " << name << std::endl;
    }
    std::cout << std::endl;
}

// ─── Signal evaluation ──────────────────────────────────────────────

double getSignal(const CellData& cells, uint32_t cell_idx,
                 int signal_id, const float* density,
                 const GridParams& grid, double current_time)
{
    int m = n_substrates_cached;

    if (signal_id < 0) {
        return 0.0;
    }

    // ── Substrate concentration [0 .. m-1] ──
    if (signal_id < m) {
        if (!density) return 0.0;
        uint32_t vi = cells.voxel_index[cell_idx];
        uint32_t total_voxels = grid.nx * grid.ny * grid.nz;
        if (vi >= total_voxels) return 0.0;
        size_t offset = static_cast<size_t>(vi) +
                        static_cast<size_t>(signal_id) * total_voxels;
        return static_cast<double>(density[offset]);
    }

    // ── Substrate gradient magnitude [m .. 2m-1] ──
    if (signal_id < 2 * m) {
        // Gradient signals require gradient computation which we don't
        // have on CPU in this architecture. Return 0 as placeholder.
        // In PhysiCell, gradients are precomputed by BioFVM.
        return 0.0;
    }

    int base_idx = 2 * m;

    // ── Pressure ──
    if (signal_id == base_idx) {
        // Simple pressure: not stored per-cell in our SoA yet.
        // PhysiCell uses simple_pressure = number_of_attached_cells / reference.
        // Return 0 as placeholder; can be wired up when contact data is available.
        return 0.0;
    }

    // ── Volume ──
    if (signal_id == base_idx + 1) {
        return static_cast<double>(cells.total_volume[cell_idx]);
    }

    // ── Contact with live cells ──
    if (signal_id == base_idx + 2) {
        // Contact counting requires neighbor lists from spatial hash.
        // Return 0 as placeholder; wired up when mechanics provides neighbor data.
        return 0.0;
    }

    // ── Contact with dead cells ──
    if (signal_id == base_idx + 3) {
        return 0.0;  // placeholder
    }

    // ── Damage ──
    if (signal_id == base_idx + 4) {
        return 0.0;  // placeholder — no damage field in our SoA yet
    }

    // ── Dead state ──
    if (signal_id == base_idx + 5) {
        return (cells.is_alive[cell_idx] == 0) ? 1.0 : 0.0;
    }

    // ── Time ──
    if (signal_id == base_idx + 6) {
        return current_time;
    }

    // ── Custom: oncoprotein ──
    if (signal_id == base_idx + 7) {
        return static_cast<double>(cells.oncoprotein[cell_idx]);
    }

    // Unknown signal
    std::cerr << "Warning: getSignal called with unknown signal_id "
              << signal_id << std::endl;
    return 0.0;
}

double getSignalByName(const CellData& cells, uint32_t cell_idx,
                       const std::string& name, const float* density,
                       const GridParams& grid, double current_time)
{
    int idx = findSignalIndex(name);
    return getSignal(cells, cell_idx, idx, density, grid, current_time);
}

} // namespace PhysiCellMetal

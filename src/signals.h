#ifndef PHYSICELL_METAL_SIGNALS_H
#define PHYSICELL_METAL_SIGNALS_H

// ─────────────────────────────────────────────────────────────────────
// Signal System — Tier 4 (Signal/Behavior/Rules)
//
// Reads cell state and microenvironment to produce signal values.
// Port of PhysiCell's get_single_signal / signal dictionary to SoA.
//
// Signal index layout (dynamic, built at init):
//   [0 .. m-1]         substrate concentrations
//   [m .. 2m-1]        substrate gradient magnitudes
//   [2m]               pressure
//   [2m+1]             volume
//   [2m+2]             contact with live cells
//   [2m+3]             contact with dead cells
//   [2m+4]             damage (placeholder, always 0 for now)
//   [2m+5]             dead (0 or 1)
//   [2m+6]             time
//   [2m+7]             custom:oncoprotein
//
// Where m = number of substrates.
// ─────────────────────────────────────────────────────────────────────

#include <cstdint>
#include <string>
#include <vector>
#include <map>

class CellData;
struct GridParams;

namespace PhysiCellMetal {

// ─── Signal dictionary ───

/// Initialize the signal name → index mapping.
/// Must be called after config parse, before any getSignal calls.
void initSignalDictionary(int n_substrates,
                          const std::vector<std::string>& substrate_names);

/// Find signal index by name. Returns -1 if not found.
int findSignalIndex(const std::string& name);

/// Get signal name from index. Returns "unknown" if out of range.
std::string signalName(int index);

/// Total number of registered signals.
int numSignals();

/// Display the signal dictionary to stdout.
void displaySignalDictionary();

// ─── Signal evaluation ───

/// Get a single signal value for a specific cell.
/// @param cells     SoA cell data
/// @param cell_idx  index of the cell to query
/// @param signal_id signal index (from findSignalIndex or dictionary)
/// @param density   microenvironment density grid (float*, voxel-major)
/// @param grid      grid parameters
/// @param current_time simulation time in minutes
double getSignal(const CellData& cells, uint32_t cell_idx,
                 int signal_id, const float* density,
                 const GridParams& grid, double current_time);

/// Convenience: get signal by name.
double getSignalByName(const CellData& cells, uint32_t cell_idx,
                       const std::string& name, const float* density,
                       const GridParams& grid, double current_time);

} // namespace PhysiCellMetal

#endif // PHYSICELL_METAL_SIGNALS_H

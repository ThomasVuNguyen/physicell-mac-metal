#ifndef MATLAB_WRITER_H
#define MATLAB_WRITER_H

#include <string>
#include <vector>

/// Write a MATLAB version 4 .mat file from a 2D std::vector array.
/// The input is a list of rows, where each row is a vector of doubles.
bool write_matlab4(const std::vector<std::vector<double>>& input, const std::string& filename, const std::string& variable_name = "none");

#endif // MATLAB_WRITER_H

#include "matlab_writer.h"
#include <cstdio>
#include <iostream>

bool write_matlab4(const std::vector<std::vector<double>>& input, const std::string& filename, const std::string& variable_name) {
    if (input.empty() || input[0].empty()) return false;

    FILE* fp = fopen(filename.c_str(), "wb");
    if (fp == NULL) {
        std::cout << "Error: could not open file " << filename << "!" << std::endl;
        return false;
    }

    typedef unsigned int UINT;
    UINT UINTs = sizeof(UINT);

    UINT type_numeric_format = 0; // little-endian assumed
    UINT type_reserved = 0;
    UINT type_data_format = 0; // doubles
    UINT type_matrix_type = 0; // full matrix

    UINT temp = 1000 * type_numeric_format + 100 * type_reserved + 10 * type_data_format + type_matrix_type;
    fwrite((char*)&temp, UINTs, 1, fp);

    UINT rows = (UINT)input[0].size();
    fwrite((char*)&rows, UINTs, 1, fp);

    UINT cols = (UINT)input.size();
    fwrite((char*)&cols, UINTs, 1, fp);

    UINT imag = 0;
    fwrite((char*)&imag, UINTs, 1, fp);

    UINT name_length = (UINT)variable_name.size() + 1; // including null terminator
    fwrite((char*)&name_length, UINTs, 1, fp);

    fwrite(variable_name.c_str(), 1, name_length, fp);

    // Write column-major (since we iterate cols then rows)
    for (size_t c = 0; c < input.size(); c++) {
        for (size_t r = 0; r < input[c].size(); r++) {
            double val = input[c][r];
            fwrite((char*)&val, sizeof(double), 1, fp);
        }
    }

    fclose(fp);
    return true;
}

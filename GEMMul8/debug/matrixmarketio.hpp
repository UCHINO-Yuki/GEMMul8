#pragma once
#include "matrix.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

enum MatrixMarketFormat {
    COORDINATE
};

inline std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;

    while (getline(ss, item, delim)) {
        result.push_back(item);
    }

    return result;
}

template <typename T>
inline T GetAs(const std::string &s) {
    std::stringstream ss{s};
    T t;
    ss >> t;
    return t;
}

template <typename T>
inline void load_matrix(const std::string &matrix_file, const MatrixMarketFormat fmt, Matrix<T> &matrix) {
    if (fmt == COORDINATE) {
        std::ifstream stream(matrix_file);
        std::string str;
        bool header_passed = false;
        unsigned int sz_rows;
        unsigned int sz_cols;
        unsigned int nz_elem;
        while (std::getline(stream, str)) {
            if (str[0] == '%') continue;
            if (!header_passed) {
                std::vector strvec = split(str, ' ');
                if (strvec.size() != 3) {
                    std::cerr << "  [ERR] Invalid matrix market declaration: \"" << str << "\"" << std::endl;
                }
                sz_rows = std::stoi(strvec[0]);
                sz_cols = std::stoi(strvec[1]);
                nz_elem = std::stoi(strvec[2]);
                delete[] matrix.data;
                matrix.sz_rows = sz_rows;
                matrix.sz_cols = sz_cols;
                matrix.data    = new T[sz_rows * sz_cols];
                matrix.clean();
                header_passed = true;
                continue;
            }
            std::vector strvec = split(str, ' ');
            if (strvec.size() == 2) std::cout << "  [WAR] Interpretting empty assignment as 1" << std::endl;
            else if (strvec.size() != 3)
                std::cerr << "  [ERR] Invalid matrix market declaration: \"" << str << "\"" << std::endl;
            unsigned int i = std::stoi(strvec[0]);
            unsigned int j = std::stoi(strvec[1]);
            T value;
            if (strvec.size() == 3) value = GetAs<T>(strvec[2]);
            else value = 1;
            matrix.set(i, j, value);
            nz_elem--;
            // std::cout << nz_elem << "\n";
        }
        if (nz_elem != 0) {
            std::cerr << "  [WAR] Potentially missing/excessive non-zero elements";
        }
    } else {
        std::cerr << "  [ERR] Unsupported Matrix Market Format used";
    }
}

template <typename T>
inline void save_matrix(const std::string &matrix_file, const MatrixMarketFormat fmt, Matrix<T> &matrix) {
    if (fmt == COORDINATE) {
        std::ofstream file(matrix_file);
        file << "%%MatrixMarket matrix coordinate real general" << std::endl;
        file << matrix.sz_rows << " " << matrix.sz_cols << " " << matrix.countNonZero() << std::endl;
        for (unsigned int i = 0; i < matrix.sz_rows; i++) {
            for (unsigned int j = 0; j < matrix.sz_cols; j++) {
                if (matrix.get(i, j) != 0) {
                    file << i << " " << j << " " << matrix.get(i, j) << std::endl;
                }
            }
        }
        file.close();
    } else {
        std::cerr << "  [ERR] Unsupported Matrix Market Format used";
    }
}

#pragma once
#include "../../common/common.hpp"

namespace gemmul8::scaling::general {

inline constexpr unsigned threads_x_general                            = common::TILE_DIM;
template <Backend BACKEND> inline constexpr unsigned threads_y_rowwise = 4U;
template <> inline constexpr unsigned threads_y_rowwise<Backend::FP8>  = 16U;
template <typename T> inline constexpr unsigned threads_y_symm_hemm    = (common::isComplex<T>) ? 16U : 8U;

inline constexpr unsigned threads_x_colwise_full_tiled = 32U;
inline constexpr unsigned threads_y_colwise_full_tiled = 4U;

} // namespace gemmul8::scaling::general

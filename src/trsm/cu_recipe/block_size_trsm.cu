#include "../../../include/trsm.hpp"

namespace gemmul8 {

namespace {

int &trsm_block_size_override_storage() noexcept {
    static int nB = 0;
    return nB;
}

} // namespace

void set_block_size_trsm(const int nB) noexcept {
    trsm_block_size_override_storage() = nB;
}

int get_block_size_trsm() noexcept {
    return trsm_block_size_override_storage();
}

} // namespace gemmul8

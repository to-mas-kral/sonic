#ifndef MAKE_ARRAY_H
#define MAKE_ARRAY_H

#include <array>

namespace sonic {
/// https://stackoverflow.com/a/17923795
template <std::size_t N, class T>
std::array<T, N>
make_array(const T &v) {
    std::array<T, N> ret;
    ret.fill(v);
    return ret;
}
} // namespace sonic

#endif // MAKE_ARRAY_H

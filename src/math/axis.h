#ifndef AXIS_H
#define AXIS_H

enum class Axis : u8 {
    X = 0U,
    Y = 1U,
    Z = 2U,
};

inline Axis
next_axis(const Axis axis) {
    return static_cast<Axis>((static_cast<u8>(axis) + 1) % 3U);
}

#endif // AXIS_H

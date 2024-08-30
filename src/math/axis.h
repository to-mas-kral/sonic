#ifndef AXIS_H
#define AXIS_H

enum class Axis : u8 {
    X = 0,
    Y = 1,
    Z = 2,
};

inline Axis
next_axis(const Axis axis) {
    return static_cast<Axis>((static_cast<u8>(axis) + 1) % 3u);
}

#endif // AXIS_H

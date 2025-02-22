#ifndef COLORMAPS_H
#define COLORMAPS_H

#include <array>

#include "../math/vecmath.h"

// Tableau 20 colormap taken from:
// https://github.com/jiffyclub/palettable/blob/master/palettable/tableau/tableau.py
constexpr auto COLORMAP_TABLEAU20 =
    std::array{tuple3(31.F / 255.F, 119.F / 255.F, 180.F / 255.F),
               tuple3(174.F / 255.F, 199.F / 255.F, 232.F / 255.F),
               tuple3(255.F / 255.F, 127.F / 255.F, 14.F / 255.F),
               tuple3(255.F / 255.F, 187.F / 255.F, 120.F / 255.F),
               tuple3(44.F / 255.F, 160.F / 255.F, 44.F / 255.F),
               tuple3(152.F / 255.F, 223.F / 255.F, 138.F / 255.F),
               tuple3(214.F / 255.F, 39.F / 255.F, 40.F / 255.F),
               tuple3(255.F / 255.F, 152.F / 255.F, 150.F / 255.F),
               tuple3(148.F / 255.F, 103.F / 255.F, 189.F / 255.F),
               tuple3(197.F / 255.F, 176.F / 255.F, 213.F / 255.F),
               tuple3(140.F / 255.F, 86.F / 255.F, 75.F / 255.F),
               tuple3(196.F / 255.F, 156.F / 255.F, 148.F / 255.F),
               tuple3(227.F / 255.F, 119.F / 255.F, 194.F / 255.F),
               tuple3(247.F / 255.F, 182.F / 255.F, 210.F / 255.F),
               tuple3(127.F / 255.F, 127.F / 255.F, 127.F / 255.F),
               tuple3(199.F / 255.F, 199.F / 255.F, 199.F / 255.F),
               tuple3(188.F / 255.F, 189.F / 255.F, 34.F / 255.F),
               tuple3(219.F / 255.F, 219.F / 255.F, 141.F / 255.F),
               tuple3(23.F / 255.F, 190.F / 255.F, 207.F / 255.F),
               tuple3(158.F / 255.F, 218.F / 255.F, 229.F / 255.F)};

namespace sonic {
inline tuple3
colormap(const u32 id) {
    return COLORMAP_TABLEAU20[id % COLORMAP_TABLEAU20.size()];
}
} // namespace sonic

#endif // COLORMAPS_H

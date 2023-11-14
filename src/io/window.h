#ifndef PT_WINDOW_H
#define PT_WINDOW_H

#include <MiniFB_cpp.h>

#include "../framebuffer.h"
#include "../utils/numtypes.h"

class Window {
public:
    Window(int resx, int resy);

    void close();

    void update(Framebuffer &fb, int samples);

private:
    std::vector<u32> framebuffer;
    mfb_window *window;
};

#endif // PT_WINDOW_H

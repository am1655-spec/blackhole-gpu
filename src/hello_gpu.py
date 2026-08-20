"""
Phase 0 — does the pipeline work at all?

Run this first, on whichever machine you're on. It doesn't know or care
about black holes; it just proves venv -> taichi -> GPU backend -> a
window on screen all actually connect. Ctrl+C or close the window to quit.

ti.gpu auto-picks the best backend it can find: CUDA on the desktop
(RTX 2070), Vulkan on the laptop (Arc iGPU). Whatever it picks gets
printed on startup -- that line is the thing to check.
"""
import taichi as ti

ti.init(arch=ti.gpu)
print(f"Taichi backend in use: {ti.lang.impl.current_cfg().arch}")

W, H = 512, 512
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(W, H))


@ti.kernel
def paint(t: ti.f32):
    for x, y in pixels:
        u = x / W
        v = y / H
        pixels[x, y] = ti.Vector([
            0.5 + 0.5 * ti.sin(t + u * 6.28),
            0.5 + 0.5 * ti.sin(t + v * 6.28 + 2.0),
            0.5 + 0.5 * ti.sin(t + (u + v) * 6.28 + 4.0),
        ])


gui = ti.GUI("hello_gpu — Phase 0 check", res=(W, H))
t = 0.0
while gui.running:
    paint(t)
    gui.set_image(pixels)
    gui.show()
    t += 0.02

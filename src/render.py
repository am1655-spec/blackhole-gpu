"""
Phase 1 — static lensed image.

The rendering trick: Schwarzschild spacetime is spherically symmetric,
so a photon's path only depends on its impact parameter b (how far its
incoming line would miss the black hole by, if gravity did nothing).
That means each pixel is an independent 2D polar-coordinate trace --
no need for full 3D ray tracing. Camera sits far along phi=0 looking at
the black hole; each pixel's distance from the image center sets its b;
the pixel's *angle* around the image center is just carried along as a
"which direction around the black hole" label (azimuth) for sampling
the background pattern once the ray's done bending.

This file is the plumbing: camera, per-pixel loop, background pattern,
window display. The physics happens in geodesic.py -- this only calls
geodesic.step() in a loop and draws whatever comes back. Run it now and
you'll get a valid image (garbage bending, since geodesic.step() is
still a placeholder); implement geodesic.step() and rerun to see it
turn into an actual lensed field.
"""
import taichi as ti
import geodesic

ti.init(arch=ti.gpu)
print(f"Taichi backend in use: {ti.lang.impl.current_cfg().arch}")

# --- scene constants (natural units: rs = 1) ---
RS = 1.0
CAM_DIST = 20.0 * RS
ESCAPE_RADIUS = 60.0 * RS
B_MAX = 7.0 * RS          # impact parameter at the edge of the frame
H = 0.05                  # integration step size
MAX_STEPS = 4000

W, H_RES = 640, 640
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(W, H_RES))


@ti.func
def background(phi: ti.f32, azimuth: ti.f32) -> ti.math.vec3:
    # Simple checkerboard over (bend angle, azimuth) so lensing distortion
    # is visible as warped tiles instead of a flat gradient. Swap this out
    # for anything you like later (a real starfield texture, a skybox, etc).
    stripe = ti.floor(phi * 4.0) + ti.floor(azimuth * 6.0 / (2.0 * 3.14159265))
    lit = (ti.cast(stripe, ti.i32) % 2) == 0
    return ti.Vector([0.9, 0.9, 0.85]) if lit else ti.Vector([0.08, 0.08, 0.12])


@ti.kernel
def render():
    cx, cy = W / 2.0, H_RES / 2.0
    max_screen_r = ti.min(cx, cy)
    for x, y in pixels:
        sx = x - cx
        sy = y - cy
        r_screen = ti.sqrt(sx * sx + sy * sy)
        azimuth = ti.atan2(sy, sx)
        b = (r_screen / max_screen_r) * B_MAX

        dist = CAM_DIST
        phi = 0.0
        ddist = -ti.sqrt(ti.max(0.0, 1.0 - (b * b) / (dist * dist)))
        dphi = b / (dist * dist)

        captured = 0
        step_count = 0
        while step_count < MAX_STEPS and captured == 0:
            dist, phi, ddist, dphi = geodesic.step(dist, phi, ddist, dphi, RS, H)
            if geodesic.event_horizon_hit(dist, RS):
                captured = 1
            if dist > ESCAPE_RADIUS:
                step_count = MAX_STEPS
            step_count += 1

        if captured == 1:
            pixels[x, y] = ti.Vector([0.0, 0.0, 0.0])
        else:
            pixels[x, y] = background(phi, azimuth)


if __name__ == "__main__":
    render()
    gui = ti.GUI("blackhole-gpu — Phase 1", res=(W, H_RES))
    gui.set_image(pixels)
    gui.show("../first_render.png")
    while gui.running:
        gui.set_image(pixels)
        gui.show()

import taichi as ti
import geodesic

ti.init(arch=ti.gpu)
print(f"Taichi backend in use: {ti.lang.impl.current_cfg().arch}")

# --- scene constants (natural units: rs = 1) ---
RS = 1.0
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
def render(cam_dist: ti.f32, orbit_angle: ti.f32):
    escape_radius = cam_dist * 3.0
    cx, cy = W / 2.0, H_RES / 2.0
    max_screen_r = ti.min(cx, cy)
    for x, y in pixels:
        sx = x - cx
        sy = y - cy
        r_screen = ti.sqrt(sx * sx + sy * sy)
        azimuth = ti.atan2(sy, sx) + orbit_angle
        b = (r_screen / max_screen_r) * B_MAX

        dist = cam_dist
        phi = 0.0
        ddist = -ti.sqrt(ti.max(0.0, 1.0 - (b * b) / (dist * dist)))
        dphi = b / (dist * dist)

        captured = 0
        step_count = 0
        while step_count < MAX_STEPS and captured == 0:
            dist, phi, ddist, dphi = geodesic.step(dist, phi, ddist, dphi, RS, H)
            if geodesic.event_horizon_hit(dist, RS):
                captured = 1
            if dist > escape_radius:
                step_count = MAX_STEPS
            step_count += 1

        if captured == 1:
            pixels[x, y] = ti.Vector([0.0, 0.0, 0.0])
        else:
            pixels[x, y] = background(phi, azimuth)


if __name__ == "__main__":
    gui = ti.GUI("blackhole-gpu — Phase 2", res=(W, H_RES))
    orbit_angle = 0.0
    CAM_DIST = 50.0 * RS
    prev_mouse = None

    while gui.running:
        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            if prev_mouse is not None:
                orbit_angle += (mx - prev_mouse[0]) * 2.0 * 3.14159265
            prev_mouse = (mx, my)
        else: prev_mouse = None
        for e in gui.get_events(ti.GUI.WHEEL):
            CAM_DIST *= 1.0 - e.delta[1] * 0.001
        CAM_DIST = min(max(CAM_DIST, 3.0 * RS), 100.0 * RS)

        render(CAM_DIST, orbit_angle)
        gui.set_image(pixels)
        gui.show()

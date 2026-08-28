import taichi as ti
import numpy as np
import geodesic
import sky

ti.init(arch=ti.gpu)
print(f"Taichi backend in use: {ti.lang.impl.current_cfg().arch}")

# --- scene constants (natural units: rs = 1) ---
RS = 1.0
H = 0.05                  # integration step size
MAX_STEPS = 4000
fov = 0.5                  # camera field of view (radians)

W, H_RES = 640, 640
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(W, H_RES))


@ti.kernel
def render(cam_pos: ti.math.vec3, pitch: ti.f32, yaw: ti.f32):
    escape_radius = ti.math.length(cam_pos) * 3.0
    world_up = ti.math.vec3(0.0, 1.0, 0.0)
    cam_dir = ti.math.vec3(ti.cos(pitch) * ti.sin(yaw), ti.sin(pitch), -ti.cos(pitch) * ti.cos(yaw))
    forward = ti.math.normalize(cam_dir)
    right = ti.math.normalize(ti.math.cross(forward, world_up))
    up = ti.math.cross(right, forward)

    for x, y in pixels:
        u = (x + 0.5) / W * 2.0 - 1.0
        v = (y + 0.5) / H_RES * 2.0 - 1.0
        aspect = W / H_RES
        ray_dir = ti.math.normalize(forward + u * aspect * ti.tan(fov/2) * right + v * ti.tan(fov/2) * up)
        plane_norm = ti.math.cross(ray_dir, ti.math.normalize(cam_pos))
        basis1 = ti.math.normalize(cam_pos)
        basis2 = ti.math.normalize(ti.math.cross(plane_norm, basis1))

        dist = ti.math.length(cam_pos)
        phi = 0.0
        ddist = ti.math.dot(ray_dir, basis1)
        dphi = ti.math.dot(ray_dir, basis2) / dist

        captured = 0
        step_count = 0
        while step_count < MAX_STEPS and captured == 0:
            dist, phi, ddist, dphi = geodesic.step(dist, phi, ddist, dphi, RS, H)
            if geodesic.event_horizon_hit(dist, RS):
                captured = 1
            if dist > escape_radius:
                step_count = MAX_STEPS
            step_count += 1

        sky_dir = ti.math.cos(phi) * basis1 + ti.math.sin(phi) * basis2

        if captured == 1:
            pixels[x, y] = ti.Vector([0.0, 0.0, 0.0])
        else:
            pixels[x, y] = sky.background(sky_dir)


if __name__ == "__main__":
    gui = ti.GUI("blackhole-gpu — Phase 2", res=(W, H_RES))
    cam_pos = ti.math.vec3(0.0, 0.0, 99.0 * RS)
    prev_mouse = None
    pitch, yaw = 0.0, 0.0

    while gui.running:
        forward = np.array([np.cos(pitch) * np.sin(yaw), np.sin(pitch), -np.cos(pitch) * np.cos(yaw)])
        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            if prev_mouse is not None:
                yaw += (mx - prev_mouse[0])  * 3.14159265
                pitch += (my - prev_mouse[1]) * 3.14159265
            prev_mouse = (mx, my)
        else: prev_mouse = None
        for e in gui.get_events(ti.GUI.WHEEL):
            cam_pos += forward * (e.delta[1] * 0.1)

        dist_from_origin = np.sqrt(cam_pos[0]**2 + cam_pos[1]**2 + cam_pos[2]**2)
        clamped_dist = min(max(dist_from_origin, 3.0 * RS), 100.0 * RS)
        cam_pos = cam_pos * (clamped_dist / dist_from_origin)

        render(cam_pos, pitch, yaw)
        gui.set_image(pixels)
        gui.show()

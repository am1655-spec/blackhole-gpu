import taichi as ti
import numpy as np
import geodesic

ti.init(arch=ti.gpu)
print(f"Taichi backend in use: {ti.lang.impl.current_cfg().arch}")

# --- scene constants (natural units: rs = 1) ---
RS = 1.0
H = 0.05                  # integration step size
MAX_STEPS = 4000
fov = 0.5                  # camera field of view (radians)

STAR_DENSITY = 100       # star lattice cells per unit direction vector
STAR_THRESHOLD = 0.986      # higher = sparser stars

# SUN1 sits directly behind the black hole along the default camera's view
# axis (cam_pos starts at (0,0,10) facing (0,0,-1) -- see render()/__main__),
# i.e. source-lens-observer are aligned by default. That alignment is what
# makes an Einstein ring visible at all -- rays near the critical impact
# parameter bend ~180 deg and land back on this same axis, so this source
# shows up as a ring wrapped around the black hole's silhouette instead of
# a normal point. Sized a bit larger than a "real" point source so the ring
# is robust to small camera drift rather than needing pixel-perfect aim.
SUN1_DIR = ti.math.vec3(0.0, 0.0, -1.0)
SUN1_COLOR = ti.math.vec3(1.0, 0.85, 0.55)
SUN1_CORE_COS = float(np.cos(np.radians(5.0)))    # hard disk edge
SUN1_GLOW_COS = float(np.cos(np.radians(18.0)))   # soft halo edge

# SUN2 is off-axis -- an ordinary, unlensed star for comparison. It should
# look like a plain glowing dot with no ring, since nothing bends light
# toward it from this camera position.
SUN2_DIR = ti.math.vec3(-0.9, -0.15, 0.5)
SUN2_COLOR = ti.math.vec3(0.6, 0.78, 1.0)
SUN2_CORE_COS = float(np.cos(np.radians(2.2)))
SUN2_GLOW_COS = float(np.cos(np.radians(11.0)))

W, H_RES = 800, 800
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(W, H_RES))


@ti.func
def hash13(p: ti.math.vec3) -> ti.f32:
    # cheap GPU-friendly 3D->1D hash (Dave Hoskins-style), no lookup table needed
    p = ti.math.fract(p * 0.3183099 + 0.1)
    p = p * 17.0
    return ti.math.fract(p[0] * p[1] * p[2] * (p[0] + p[1] + p[2]))


@ti.func
def sun_glow(dir: ti.math.vec3, sun_dir: ti.math.vec3, color: ti.math.vec3,
             core_cos: ti.f32, glow_cos: ti.f32) -> ti.math.vec3:
    d = ti.math.dot(dir, ti.math.normalize(sun_dir))
    core = ti.math.smoothstep(core_cos - 0.001, core_cos, d)
    glow = ti.math.smoothstep(glow_cos, core_cos, d) * 0.35
    return color * (core + glow)


@ti.func
def background(dir: ti.math.vec3) -> ti.math.vec3:
    # Deep-space gradient (subtly brighter toward the top of the scene)
    # plus a procedural starfield: bucket the unit direction into a coarse
    # 3D lattice and hash each cell to decide if/how bright a star sits there.
    # Sampling the *direction* rather than a lat/long grid avoids pinching
    # at the poles the way a naive equirectangular texture would.
    sky = ti.math.vec3(0.01, 0.012, 0.02) + 0.01 * ti.max(0.0, dir[1])

    cell = ti.floor(dir * STAR_DENSITY)
    star_roll = hash13(cell)
    star = ti.math.smoothstep(STAR_THRESHOLD, 1.0, star_roll)
    twinkle = 0.6 + 0.4 * hash13(cell + 7.0)

    color = sky + star * twinkle * ti.math.vec3(1.0, 1.0, 0.96)
    color += sun_glow(dir, SUN1_DIR, SUN1_COLOR, SUN1_CORE_COS, SUN1_GLOW_COS)
    color += sun_glow(dir, SUN2_DIR, SUN2_COLOR, SUN2_CORE_COS, SUN2_GLOW_COS)
    return color


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
            pixels[x, y] = background(sky_dir)


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

"""
Procedural skybox background — prebuilt scene dressing, kept in its own
file specifically so it's easy to tell apart from the physics/camera code
ported from the old Java project (geodesic.py, render.py, hello_gpu.py).
render.py only ever calls sky.background(dir); nothing else here is part
of the interface between the two files.

Design (chosen after asking, not guessed): soft procedural nebula clouds
in a warm neutral palette (no hard grid lines) plus a starfield, so the
lensing shows up as clouds/stars visibly smearing and doubling near the
horizon rather than straight lines bending. Two point-like "suns" sit on
top of the nebula; one is aligned with the default camera axis so it
lenses into a full Einstein ring, the other sits off-axis as an unlensed
comparison point (see render.py's default cam_pos/cam_dir for why the
alignment matters).
"""
import taichi as ti

STAR_DENSITY = 100          # star lattice cells per unit direction vector
STAR_THRESHOLD = 0.986      # higher = sparser stars
STAR_COLOR = ti.math.vec3(1.0, 0.93, 0.82)   # warm-tinted white

NEBULA_SCALE = 2.2          # base lattice frequency (few big cloud blobs)
NEBULA_OCTAVES = 4
NEBULA_GAIN = 0.85          # overall cloud brightness
NEBULA_COLOR_A = ti.math.vec3(0.45, 0.22, 0.12)   # rust/umber
NEBULA_COLOR_B = ti.math.vec3(0.85, 0.62, 0.32)   # warm gold/cream
SKY_BASE = ti.math.vec3(0.05, 0.035, 0.025)        # warm near-black void

# SUN1 sits directly behind the black hole along the default camera's view
# axis (cam_pos starts at (0,0,10) facing (0,0,-1) -- see render.py), i.e.
# source-lens-observer are aligned by default. That alignment is what
# makes an Einstein ring visible at all -- rays near the critical impact
# parameter bend ~180 deg and land back on this same axis, so this source
# shows up as a ring wrapped around the black hole's silhouette instead of
# a normal point. Sized a bit larger than a "real" point source so the ring
# is robust to small camera drift rather than needing pixel-perfect aim.
SUN1_DIR = ti.math.vec3(0.0, 0.0, -1.0)
SUN1_COLOR = ti.math.vec3(1.0, 0.72, 0.35)   # warm gold
SUN1_CORE_COS = 0.9945219    # cos(6 deg) -- hard disk edge
SUN1_GLOW_COS = 0.9396926    # cos(20 deg) -- soft halo edge

# SUN2 is off-axis -- an ordinary, unlensed star for comparison. It should
# look like a plain glowing dot with no ring, since nothing bends light
# toward it from this camera position.
SUN2_DIR = ti.math.vec3(-0.9, -0.15, 0.5)
SUN2_COLOR = ti.math.vec3(0.92, 0.85, 0.75)   # pale warm cream
SUN2_CORE_COS = 0.9993908    # cos(2.2 deg)
SUN2_GLOW_COS = 0.9848078    # cos(10 deg)


@ti.func
def hash13(p: ti.math.vec3) -> ti.f32:
    # cheap GPU-friendly 3D->1D hash (Dave Hoskins-style), no lookup table needed
    p = ti.math.fract(p * 0.3183099 + 0.1)
    p = p * 17.0
    return ti.math.fract(p[0] * p[1] * p[2] * (p[0] + p[1] + p[2]))


@ti.func
def value_noise(p: ti.math.vec3) -> ti.f32:
    # Trilinearly-interpolated lattice noise (smooth, unlike the hard-cell
    # hash the starfield uses) -- gives the nebula soft blobby edges
    # instead of a static-y speckle.
    i = ti.floor(p)
    f = ti.math.fract(p)
    f = f * f * (3.0 - 2.0 * f)   # smoothstep easing between lattice points

    c000 = hash13(i + ti.math.vec3(0.0, 0.0, 0.0))
    c100 = hash13(i + ti.math.vec3(1.0, 0.0, 0.0))
    c010 = hash13(i + ti.math.vec3(0.0, 1.0, 0.0))
    c110 = hash13(i + ti.math.vec3(1.0, 1.0, 0.0))
    c001 = hash13(i + ti.math.vec3(0.0, 0.0, 1.0))
    c101 = hash13(i + ti.math.vec3(1.0, 0.0, 1.0))
    c011 = hash13(i + ti.math.vec3(0.0, 1.0, 1.0))
    c111 = hash13(i + ti.math.vec3(1.0, 1.0, 1.0))

    x00 = ti.math.mix(c000, c100, f[0])
    x10 = ti.math.mix(c010, c110, f[0])
    x01 = ti.math.mix(c001, c101, f[0])
    x11 = ti.math.mix(c011, c111, f[0])
    y0 = ti.math.mix(x00, x10, f[1])
    y1 = ti.math.mix(x01, x11, f[1])
    return ti.math.mix(y0, y1, f[2])


@ti.func
def fbm(p: ti.math.vec3) -> ti.f32:
    # Sum of progressively finer/dimmer noise octaves -- turns single
    # blobby noise into cloud-like detail at multiple scales.
    total = 0.0
    amp = 0.5
    freq = 1.0
    for _ in range(NEBULA_OCTAVES):
        total += amp * value_noise(p * freq)
        freq *= 2.0
        amp *= 0.5
    return total


@ti.func
def sun_glow(dir: ti.math.vec3, sun_dir: ti.math.vec3, color: ti.math.vec3,
             core_cos: ti.f32, glow_cos: ti.f32) -> ti.math.vec3:
    d = ti.math.dot(dir, ti.math.normalize(sun_dir))
    core = ti.math.smoothstep(core_cos - 0.001, core_cos, d)
    glow = ti.math.smoothstep(glow_cos, core_cos, d) * 0.35
    return color * (core + glow)


@ti.func
def nebula(dir: ti.math.vec3) -> ti.math.vec3:
    density = fbm(dir * NEBULA_SCALE)
    # Soft wisps rather than uniform fog, but leave a low floor so no
    # direction reads as flat black -- "not all black" was the point.
    density = ti.math.clamp(density * 1.15 - 0.15, 0.0, 1.0)
    density = ti.math.pow(density, 1.4)

    # A second, decorrelated noise sample (offset lattice) picks the
    # local blend between the two cloud colors, so the nebula isn't a
    # flat single tint.
    hue_t = fbm(dir * NEBULA_SCALE * 1.7 + ti.math.vec3(31.7, 9.2, 4.4))
    cloud_color = ti.math.mix(NEBULA_COLOR_A, NEBULA_COLOR_B, hue_t)

    return cloud_color * density * NEBULA_GAIN


@ti.func
def starfield(dir: ti.math.vec3) -> ti.math.vec3:
    # One lattice cell = at most one star, placed at a jittered point
    # *within* the cell and drawn with an angular falloff around that
    # point -- a round dot, not the whole cell lit up as a flat quad.
    # Checking a pixel's own cell plus its neighbors avoids clipping
    # stars whose jittered center lands near a cell boundary.
    color = ti.math.vec3(0.0)
    base_cell = ti.floor(dir * STAR_DENSITY)
    for ox in range(-1, 2):
        for oy in range(-1, 2):
            for oz in range(-1, 2):
                cell = base_cell + ti.math.vec3(ox, oy, oz)
                star_roll = hash13(cell)
                if star_roll > STAR_THRESHOLD:
                    jitter = ti.math.vec3(hash13(cell + 1.7), hash13(cell + 5.3),
                                           hash13(cell + 9.1)) - 0.5
                    star_dir = ti.math.normalize((cell + 0.5 + jitter * 0.7) / STAR_DENSITY)
                    d = ti.math.dot(ti.math.normalize(dir), star_dir)
                    # Point-sized: ~1-3px radius at this fov/resolution, not
                    # a big soft blob -- a proper starfield reads as pinpricks.
                    radius = ti.math.mix(0.0009, 0.0026, hash13(cell + 3.3))
                    edge_cos = ti.cos(radius)
                    inner_cos = ti.cos(radius * 0.5)
                    dot = ti.math.smoothstep(edge_cos, inner_cos, d)
                    twinkle = 0.6 + 0.4 * hash13(cell + 7.0)
                    color = ti.math.max(color, dot * twinkle * STAR_COLOR)
    return color


@ti.func
def background(dir: ti.math.vec3) -> ti.math.vec3:
    color = SKY_BASE + nebula(dir) + starfield(dir)
    color += sun_glow(dir, SUN1_DIR, SUN1_COLOR, SUN1_CORE_COS, SUN1_GLOW_COS)
    color += sun_glow(dir, SUN2_DIR, SUN2_COLOR, SUN2_CORE_COS, SUN2_GLOW_COS)
    # Overlapping bright elements (stars, nebula, sun glow) can sum past 1.0;
    # clamp here so exporters that don't clip float->uint8 themselves (seen
    # wrapping into jagged, wrong-hued blotches instead of clean highlights)
    # can't produce that artifact.
    return ti.math.clamp(color, 0.0, 1.0)

# Session log — 2026-08-21

Recap of one working session, kept specifically to separate what was your
own work from what came from Claude, and how much of each. Written by
Claude at your request; corrections welcome if any attribution below is
off.

## Your own work

- **`geodesic.py`'s RK4 `step()` skeleton and the angular derivative
  term.** Both arrived already correct the first time this session looked
  at the file — the four-stage RK4 combination (`k1..k4`, `h/6` weighting)
  and the angular equation `d(dphi)/dlambda = -2*ddist*dphi/dist` (which
  falls straight out of conserving `r^2 * dphi/dlambda`) were already
  written and right, ported from your old `Ray.java`.
- **The original 2D on-axis camera pan/zoom** (mouse-drag orbit, scroll
  zoom) — written from scratch. It went through several rounds of bugs
  (see below) but the implementation itself, each time, was yours.
- **The entire free 3D camera rewrite** — camera position/orientation
  state, per-pixel ray generation, the orbital-plane reduction (plane
  normal, in-plane basis vectors, projecting the ray onto them for
  `ddist`/`dphi`), and the exit-direction reconstruction — was coded by
  you, in stages, from conceptual explanations (below). This is the bulk
  of the session's work and the part most worth calling your own: the
  concepts were explained, but translating "here's the math" into working
  Taichi across ~6 iterations, finding and fixing your own follow-on bugs
  as you went, was you.
- Assorted tuning after things worked: star density/threshold, window
  resolution, starting camera distance, mouse/wheel sensitivity.

## Explained by Claude, implemented by you

- **The correct radial gravity term.** Your first attempt at
  `derivatives()`'s radial component was dimensionally inconsistent and
  produced negligible, wrong-signed bending. Claude derived the correct
  form (`dphi^2 * (dist - 1.5*rs)`) via chain rule from the docstring's
  reduced `d^2u/dphi^2` equation and explained it in words; you wrote the
  fix.
- **The 2D-to-3D camera bridge, conceptually**: what `forward` vs.
  `cam_target` mean, the pinhole-camera `ray_dir` formula (screen coords
  -> NDC -> lean by FOV), why a single photon's path stays confined to
  one plane through the black hole (letting the existing 2D integrator
  survive unchanged), how to build that plane's basis vectors from
  `cam_pos`/`ray_dir`, and how to reconstruct a 3D exit direction from the
  final `phi`. You wrote all of the actual code for this from these
  explanations.
- **Decoupling `pitch`/`yaw`.** Claude diagnosed why the mouse-look felt
  wrong (a single shared-pole spherical parametrization coupling the two
  axes instead of independent rotations) and, at your request, wrote the
  corrected formula directly (`cam_dir = (cos(pitch)*sin(yaw), sin(pitch),
  -cos(pitch)*cos(yaw))`) plus the matching swap in the mouse-drag mapping.

## Bugs found by Claude, fixed by you

A long iterative list, mostly surfaced by you asking "check the state" /
"is this right" after each pass:

- 2D phase: `orbit_angle` computed but never read by the kernel; `CAM_DIST`
  silently frozen after first frame because a `@ti.kernel` with no
  matching argument bakes in Python globals at first trace instead of
  rereading them; `gui.is_pressed(ti.GUI.WHEEL)` / a nonexistent
  `get_wheel_delta()` used instead of `get_events()`; a fixed
  `ESCAPE_RADIUS` not scaling with a now-dynamic camera distance
  (the "three triangles" bug); a `+=` wheel handler that only ever grew
  the distance regardless of scroll direction.
- 3D transition: undefined `fov`; dangling references to deleted
  `sx`/`sy`/`azimuth` variables; `basis2` built from the wrong vectors
  (anchored to `ray_dir` instead of the plane normal, twice — once
  producing the plane's normal itself rather than an in-plane vector,
  giving `dphi == 0` for every pixel); `basis2` left unnormalized;
  `cam_pos` recomputed from a scalar every frame instead of being real
  persisted state; `pitch`/`yaw` undefined in `__main__` and never passed
  into the kernel at all (mouse-look a no-op); `forward` built as a plain
  Python tuple (`tuple * float` doesn't scale — either repeats the
  sequence or raises `TypeError`); `forward` computed once before the
  loop instead of every frame (zoom always followed the original view
  direction); `np.array[...]` (subscripting the function) instead of
  `np.array([...])`.

## Written directly by Claude

- **The `cam_pos` distance clamp** (`min(max(...))` rescaling, keeping the
  camera within the range the fixed `MAX_STEPS`/`H` integration budget can
  resolve) — written twice, since the first pass was lost to an unsaved-
  edit overwrite.
- **The procedural background** in full: the `hash13` 3D hash function,
  the starfield (direction-bucketed lattice + hash, chosen over a lat/long
  texture specifically to avoid pole-pinching), the `sun_glow` disk+halo
  helper, and the two light sources — one deliberately aligned with the
  default camera's view axis so it visibly lenses into an Einstein ring
  around the shadow, the other left off-axis as an unlensed comparison
  star. Verified headlessly (this environment has no display) by sampling
  the shader-equivalent Taichi kernel directly and confirming the shadow
  boundary lands at the theoretical `b_crit = 3*sqrt(3)/2 ≈ 2.598` and
  that the aligned sun's glow shows up right at that boundary.
- The two commits (`9e7174b`, `0377470`) and the push to `origin/master`.

## Rough shape of the split

Physics core (RK4 + the angular equation): yours. The one physics
correction needed (radial gravity term): derived by Claude, coded by you.
2D camera: yours, debugged with Claude's help. 3D camera: the concepts
came from Claude, essentially all of the code came from you, across many
rounds of Claude-found bugs that you fixed. Background/lensing demo:
Claude's, end to end.

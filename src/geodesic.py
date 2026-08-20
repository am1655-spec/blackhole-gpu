"""
Phase 1 — the actual physics. This is the part left for you to fill in;
everything else in this repo just gets a ray to this function and draws
whatever it returns.

You already derived and implemented this once, in the Java project
(blackhole-sim, package blackhole_sim.physics -- Ray.java / SchwartzSim.java):
RK4 integration of the Schwarzschild null-geodesic equations in polar
coordinates. Same math here, just as a Taichi @ti.func instead of a Java
method, so it can run per-pixel in parallel on the GPU instead of
per-particle on the CPU.

State (matches your old Ray.java fields):
    dist   -- radial distance r
    phi    -- angular position
    ddist  -- dr/dlambda  (radial "velocity")
    dphi   -- dphi/dlambda (angular "velocity")

Standard form of the equations (affine parameter lambda, r_s =
Schwarzschild radius, using the conserved impact parameter b = L/E so you
don't need a separate energy/angular-momentum bookkeeping step):

    d^2u/dphi^2 + u = 1.5 * r_s * u^2      where u = 1/r

...which is the form most lensing write-ups use. Your Java version instead
carried ddist/dphi directly as first-order state and stepped all four with
RK4 -- reuse that approach; it's already proven to work and translates
directly into the state tuple below.

Until you fill this in, step() below applies no gravity at all -- it's
not even a proper straight line in these polar coordinates (that would
need a centrifugal term too), it's just the minimum stub that lets
render.py run end-to-end and put *something* on screen. Expect a
swirly, physically-meaningless pattern until you implement the real
integration -- that's the "wiring works, physics doesn't yet" checkpoint.
"""
import taichi as ti


@ti.func
def step(dist: ti.f32, phi: ti.f32, ddist: ti.f32, dphi: ti.f32,
         rs: ti.f32, h: ti.f32):
    # TODO: replace with RK4 integration of the geodesic equations.
    # Placeholder: straight-line motion (no gravity), so the rest of the
    # pipeline has something valid to render before you touch this.
    new_dist = dist + ddist * h
    new_phi = phi + dphi * h
    return new_dist, new_phi, ddist, dphi


@ti.func
def event_horizon_hit(dist: ti.f32, rs: ti.f32) -> ti.i32:
    return dist <= rs

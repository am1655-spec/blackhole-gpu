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
    k1 = derivatives(dist, phi, ddist, dphi, rs)
    k2 = derivatives(dist + 0.5 * h * k1[0], phi + 0.5 * h * k1[1],
                     ddist + 0.5 * h * k1[2], dphi + 0.5 * h * k1[3], rs)
    k3 = derivatives(dist + 0.5 * h * k2[0], phi + 0.5 * h * k2[1],
                     ddist + 0.5 * h * k2[2], dphi + 0.5 * h * k2[3], rs)
    k4 = derivatives(dist + h * k3[0], phi + h * k3[1],
                         ddist + h * k3[2], dphi + h * k3[3], rs)
    new_dist = dist + (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) * h / 6
    new_phi = phi + (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) * h / 6
    new_ddist = ddist + (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) * h / 6
    new_dphi = dphi + (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) * h / 6
    return new_dist, new_phi, new_ddist, new_dphi


@ti.func
def derivatives(dist: ti.f32, phi: ti.f32, ddist: ti.f32, dphi: ti.f32,
                rs: ti.f32):
    return ddist, dphi, (dphi * dphi * (dist - 1.5*rs)), ((-2.0 * ddist * dphi) / dist)
    


@ti.func
def event_horizon_hit(dist: ti.f32, rs: ti.f32) -> ti.i32:
    return dist <= rs

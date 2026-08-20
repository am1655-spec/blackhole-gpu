# Instructions

This repo lives at `~/projects/blackhole-gpu` on this machine. It's a
plain local git repo (`git init` already run, nothing committed yet) --
not linked to the second-brain vault, not tracked anywhere. Touch it or
don't; nothing is watching.

## One-time setup

Taichi's published wheels don't cover Python 3.14 yet (this machine's
default `python3`), so this needs a Python 3.12 venv instead.

**1. Install Python 3.12** (needs your sudo password, so run it yourself
rather than through me — in the Claude Code prompt you can type
`! sudo dnf install -y python3.12` to run it interactively):

```
sudo dnf install -y python3.12
```

**2. Create the venv and install dependencies:**

```
cd ~/projects/blackhole-gpu
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**3. Confirm the GPU path works:**

```
python src/hello_gpu.py
```

A window should pop up with a shifting color gradient, and the terminal
should print which backend Taichi picked. On this desktop that should
say `cuda` (the RTX 2070); on the laptop it'll say `vulkan` (Arc iGPU).
Either is fine — that's the whole point of using Taichi instead of
CUDA-only code. Close the window or Ctrl+C to quit.

If this step works, the environment is done and you never need to touch
Python-version/venv stuff again — every future session is just:

```
cd ~/projects/blackhole-gpu && source .venv/bin/activate
```

## The phased plan

Each phase is a real stopping point — nothing below assumes you finish
the next one.

- **[done] Phase 0 — `src/hello_gpu.py`.** Proves venv → Taichi → GPU
  → window all connect. Nothing to do here beyond the setup above.

- **[you are here] Phase 1 — `src/render.py` + `src/geodesic.py`.**
  Static lensed image, no animation. `render.py` (camera, per-pixel
  loop, background pattern, window/PNG output) is already written and
  runs right now — try `python src/render.py`. What you'll see is
  *wrong*, on purpose: `geodesic.step()` in `geodesic.py` is a stub
  that applies no gravity. That function is the actual task — port the
  RK4 Schwarzschild integration from your old `Ray.java` /
  `SchwartzSim.java` into it (same math, `@ti.func` instead of a Java
  method). `geodesic.py`'s docstring has the equations and the mapping
  from the old field names. Once it's real, rerunning `render.py`
  should show the checkerboard warping around the event horizon instead
  of the placeholder swirl.
  **This phase alone is a complete, shareable thing** — a single lensed
  image — even if nothing after it happens.

- **Phase 2 — make it live.** Swap the fixed-camera single render for
  a loop that recomputes each frame with a changing parameter (camera
  distance, black hole mass / `RS`, or orbit angle), using `ti.GUI`'s
  event loop (see `hello_gpu.py` for the loop shape) or mouse input via
  `gui.get_events()`. This is the "watch it update in real time on the
  GPU" payoff.

- **Phase 3 — stretch, only if still fun.** Multiple black holes
  (mirrors the config system in the old Java version), an accretion
  disk, particle trails, motion blur. Nothing here is planned out —
  pick whatever looks fun if you get here.

## Notes

- **Laptop vs. desktop.** All the code above targets `ti.gpu`, which
  auto-picks CUDA here and Vulkan on the laptop. That means the actual
  writing/debugging work isn't tied to being at this machine — you can
  develop and test `geodesic.py` on the laptop and only need this box
  to see it run on the 2070 specifically (mostly just matters for frame
  rate once Phase 2 happens; Phase 1 will look identical either way).
- **Resuming after a gap.** Nothing here expires. `git status` will
  show you exactly what's uncommitted whenever you come back; commit
  checkpoints whenever you want, or never — up to you.
- **If `hello_gpu.py` doesn't find `cuda` on this desktop:** check
  `nvidia-smi` still reports the 2070 (driver may need a re-check after
  a kernel update — see the vault's `Linux Switch — Decision` page for
  the akmod-nvidia rebuild gotcha you already hit once).

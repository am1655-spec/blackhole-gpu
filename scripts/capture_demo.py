"""Scripted demo capture added by Codex; original renderer is unchanged."""
import sys, math, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import render as r
import taichi as ti
parser=argparse.ArgumentParser()
parser.add_argument('output_dir', type=Path)
parser.add_argument('--frames', type=int, default=180)
args=parser.parse_args()
if args.frames < 1:
    parser.error('--frames must be positive')
out=args.output_dir
out.mkdir(parents=True, exist_ok=True)
count=args.frames
for i in range(count):
    t=i/max(count-1,1)
    # A closed camera path: a gentle zoom with a small look-around.
    z=80-25*(.5-.5*math.cos(2*math.pi*t))
    yaw=.025*math.sin(2*math.pi*t)
    pitch=.012*math.sin(4*math.pi*t)
    r.render(ti.math.vec3(0.,0.,z),pitch,yaw)
    ti.sync()
    ti.tools.imwrite(r.pixels.to_numpy(),str(out/f'{i:04d}.png'))
print('Captured',count,'frames; backend',ti.lang.impl.current_cfg().arch)

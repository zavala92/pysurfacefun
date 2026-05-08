"""
Complex Ginzburg-Landau reaction-diffusion on the cow surface.

The solution is complex-valued.  For visualization we plot ``real(u)``, which
matches the color pattern typically shown for this example.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pysurfacefun as psf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cow-file",
        default="/Users/gentianzavalani/Downloads/cow.csv",
        help="Rhino CSV file with n*n rows per patch",
    )
    parser.add_argument("--rhino-n", type=int, default=8, help="points per patch in the CSV file")
    parser.add_argument("--solve-n", type=int, default=8, help="polynomial grid used for the solve")
    parser.add_argument("--steps", type=int, default=2000, help="number of semi-implicit Euler steps")
    parser.add_argument("--print-every", type=int, default=10, help="print every this many steps")
    parser.add_argument("--plot", action="store_true", help="plot real(u) with Matplotlib at the end")
    parser.add_argument("--vtu", default="", help="optional VTU filename for real(u) at the end")
    args = parser.parse_args()

    cow_file = Path(args.cow_file).expanduser()
    if not cow_file.exists():
        raise FileNotFoundError(cow_file)

    delta = 5e-4
    c = 1.5
    dt = 0.03

    def N(u):
        return u - (1.0 + c * 1j) * u * (abs(u) ** 2)

    print(f"Loading {cow_file}")
    dom = psf.from_rhino(str(cow_file), args.rhino_n)
    if args.solve_n != args.rhino_n:
        print(f"Resampling cow patches from n={args.rhino_n} to n={args.solve_n}")
        dom = psf.resample_mesh(dom, args.solve_n)

    print(f"patches = {dom.npatches}, n = {dom.n}, order = {dom.order}")
    print(f"bounding box = {psf.boundingbox(dom)}")

    bb = psf.boundingbox(dom)
    u_random = psf.randnfun3(0.2, bb, seed=1)
    u = psf.surfacefun(lambda x, y, z: u_random(x, y, z), dom)

    print("Building reusable surface operators")
    t0 = time.perf_counter()
    L = psf.surfaceop(dom, {"lap": -dt * delta, "b": 1.0}, 0.0)
    L.build()
    print(f"build time = {time.perf_counter() - t0:.2f} s")

    for k in range(1, args.steps + 1):
        step_t0 = time.perf_counter()
        u = L.apply(u + dt * N(u))
        if args.print_every > 0 and (k == 1 or k == args.steps or k % args.print_every == 0):
            abs_u = abs(u)
            real_u = psf.real(u)
            print(
                f"k = {k:4d}, step time = {time.perf_counter() - step_t0:.2f} s, "
                f"real(u) range = [{psf.minEst(real_u):.3e}, {psf.maxEst(real_u):.3e}], "
                f"|u| max = {psf.maxEst(abs_u):.3e}"
            )

    if args.vtu:
        psf.write_vtu(args.vtu, psf.real(u), point_name="real_u")
        print(f"wrote {args.vtu}")

    if args.plot:
        psf.plot_surface(
            psf.real(u),
            title="cow complex Ginzburg-Landau: real(u)",
            vmin=-1.0,
            vmax=1.0,
        )


if __name__ == "__main__":
    main()

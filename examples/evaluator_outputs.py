"""Write repeatable diagnostics and snapshots with an evaluator."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pysurfacefun as psf


def main() -> None:
    dom = psf.icosphere_tri(n=7, nref=1)
    u = psf.field(lambda x, y, z: x * y * z, dom)
    residual = psf.lap(u) + 12 * u

    evaluator = psf.Evaluator(
        "notebook_outputs/evaluator_demo",
        prefix="tri_sphere",
        handlers=[
            psf.JSONLinesOutputHandler(),
            psf.NPZOutputHandler(),
            psf.VTKOutputHandler(nvis=9),
        ],
        metadata={"example": "evaluator_outputs"},
    )
    evaluator.add_task("linf_residual", lambda state: psf.norm(state["residual"], "inf"))
    evaluator.add_task("u", lambda state: state["u"])
    record = evaluator.evaluate(0, state={"u": u, "residual": residual})

    print(f"wrote {evaluator.output_dir / evaluator.manifest}")
    for file_record in record.files:
        print(f"wrote {evaluator.output_dir / file_record['path']}")


if __name__ == "__main__":
    main()

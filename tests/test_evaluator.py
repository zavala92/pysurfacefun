import json
from pathlib import Path
import tempfile

import numpy as np

import pysurfacefun as psf


def test_evaluator_writes_jsonl_npz_and_manifest_for_quad_field():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        dom = psf.sphere(n=5, nref=0)
        u = psf.field(lambda x, y, z: x * y * z, dom)

        evaluator = psf.Evaluator(
            out,
            prefix="quad",
            handlers=[psf.JSONLinesOutputHandler(), psf.NPZOutputHandler()],
            metadata={"case": "quad-smoke"},
        )
        evaluator.add_task("mass", lambda state: psf.integral(state["u"]), every=1)
        evaluator.add_task("u", lambda state: state["u"], every=2)

        record0 = evaluator.evaluate(0, state={"u": u})
        record1 = evaluator.evaluate(1, state={"u": u})

        scalar_lines = [
            json.loads(line)
            for line in (out / "scalars.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        npz_path = out / "arrays" / "quad_u_000000.npz"
        skipped_npz_path = out / "arrays" / "quad_u_000001.npz"
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))

        assert "u" in record0.values
        assert "u" not in record1.values
        assert len(scalar_lines) == 2
        assert scalar_lines[0]["values"]["mass"] == psf.integral(u)
        assert npz_path.exists()
        assert not skipped_npz_path.exists()
        with np.load(npz_path) as data:
            assert data["kind"].item() == "SurfaceFunction"
            assert data["npatches"].item() == dom.npatches
            assert "value_patch_0000" in data
            assert "x_patch_0000" in data
        assert manifest["prefix"] == "quad"
        assert manifest["metadata"]["case"] == "quad-smoke"
        assert any(file_record["kind"] == "npz" for row in manifest["history"] for file_record in row["files"])

        rerun = psf.Evaluator(out, prefix="quad", handlers=[psf.JSONLinesOutputHandler()])
        rerun.add_task("mass", lambda state: psf.integral(state["u"]))
        rerun.evaluate(0, state={"u": u})
        rerun_lines = [
            json.loads(line)
            for line in (out / "scalars.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(rerun_lines) == 1


def test_evaluator_vtk_handler_writes_triangular_snapshot():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        dom = psf.icosphere_tri(n=5, nref=0)
        u = psf.field(lambda x, y, z: x * y * z, dom)

        evaluator = psf.Evaluator(
            out,
            prefix="tri",
            handlers=[psf.VTKOutputHandler(nvis=5)],
        )
        evaluator.add_task("u", lambda state: state["u"])
        record = evaluator.evaluate(3, state={"u": u})

        vtu_path = out / "vtk" / "tri_u_000003.vtu"
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))

        assert vtu_path.exists()
        assert "VTKFile" in vtu_path.read_text(encoding="utf-8")
        assert record.files == [{"kind": "vtu", "path": "vtk/tri_u_000003.vtu", "task": "u"}]
        assert manifest["history"][0]["files"] == record.files

"""Independent DOM-control comparison, production geometry and lazy tensor check."""
import json
from pathlib import Path
import statistics

import dataset_pipeline as d
import synthetic_training as t
from synthetic_job import runtime_identity

ROOT = d.REPO / "work/dataset/bootstrap/fidelity"


def check():
    Image = d.pillow()
    records = json.loads((ROOT / "records.json").read_text())
    generation = d.read_json(ROOT / "generation.json")
    d.require(generation["renderer_sha256"] == d.digest(d.REPO/"scripts/synthetic-render.mjs") and generation["control_sha256"] == d.digest(d.REPO/"scripts/synthetic-fidelity.mjs"), "generation code changed; rerender fidelity")
    controls = d.read_json(ROOT / "controls.json")
    by_index = {r["recipe"]["index"]: r for r in records}
    expected = {(r["recipe"]["index"], i) for r in records if r["recipe"]["kind"] == "boards" for i in range(len(r["recipe"]["boards"]))}
    d.require(len(records) == 79 and len(by_index) == 79, "missing fidelity pages")
    d.require({(c["index"], c["board"]) for c in controls} == expected and len(controls) == len(expected), "missing or repeated controls")
    calibration = [r for r in records if r["recipe"]["index"] >= 100000]
    d.require(len(calibration) == 39, "missing calibration records")
    d.require({(r["recipe"]["boards"][0]["set"], r["recipe"]["boards"][0]["labels"][0]) for r in calibration} == {(s,p) for s in ("chessnut","fantasy","rhosgfx") for p in d.LABELS}, "class/design calibration incomplete")
    for r in calibration[:3]:
        d.require(d.digest(ROOT / "rebuild" / r["image"]) == r["image_sha256"], "rebuild differs")
    errors, measured = [], []
    templates = {}
    for r in calibration:
        board = r["recipe"]["boards"][0]
        ref = d.load_image(ROOT / f"control-{r['recipe']['index']}-0.png")
        for parity in (0,1):
            templates[(board["set"], board["labels"][0], parity)] = ref.crop((parity*96+3,3,parity*96+93,93)).resize((24,24), Image.Resampling.BOX).tobytes()
    identities_checked = 0
    sheets = {s: Image.new("RGB", (1152, 232), "white") for s in ("chessnut","fantasy","rhosgfx")}
    for control in controls:
        record = by_index[control["index"]]
        board = record["recipe"]["boards"][control["board"]]
        actual = t.load_board(ROOT / record["image"], record, control["board"])
        if control["index"] >= 100000 and board["labels"][0] != ".":
            number = d.LABELS.index(board["labels"][0])-1
            sheets[board["set"]].paste(actual["grid"].crop((0,0,192,96)), ((number%6)*192,(number//6)*116+20))
            from PIL import ImageDraw
            ImageDraw.Draw(sheets[board["set"]]).text(((number%6)*192,(number//6)*116), board["labels"][0], fill="black")
        reference = d.load_image(ROOT / control["image"]).resize((768,768), Image.Resampling.BICUBIC)
        tile_maes = []
        comparison_side = 24 if control["index"] >= 100000 else 48
        for i in range(64):
            x, y = i % 8 * 96, i // 8 * 96
            # Exclude grid edge antialiasing; retain all actual glyph content.
            a = actual["grid"].crop((x+3, y+3, x+93, y+93)).resize((comparison_side, comparison_side), Image.Resampling.BOX)
            b = reference.crop((x+3, y+3, x+93, y+93)).resize((comparison_side, comparison_side), Image.Resampling.BOX)
            mae = sum(abs(v-w) for v,w in zip(a.tobytes(),b.tobytes()))/(comparison_side**2*3)
            tile_maes.append(mae)
            if control["index"] >= 100000:
                pixels = a.tobytes()
                parity = (i//8+i%8)%2
                distances = {label: sum(abs(v-w) for v,w in zip(pixels,templates[(board["set"],label,parity)])) for label in d.LABELS}
                if min(distances, key=distances.get) != board["labels"][i]:
                    errors.append(f"independent piece identity mismatch at calibration {control['index']} square {i}")
                identities_checked += 1
            # Actual candidate tensor center values independently recover RGB.
            raw = actual["grid"].getpixel((x+48,y+48))
            for channel,(mean,std) in enumerate(zip((.485,.456,.406),(.229,.224,.225))):
                value = actual["tensor"][i*3*96*96+channel*96*96+48*96+48]
                if abs((value*std+mean)*255-raw[channel]) > .0001:
                    errors.append("tensor ordering/normalization disagreement")
        calibration = control["index"] >= 100000
        # Fixed before this comparison: tight same-scale control, bounded relaxed
        # resampling/hatch tolerance on small transformed full-page diagrams.
        ceiling = 3 if calibration else 18
        if max(tile_maes) > ceiling:
            errors.append(f"control {control['index']} board {control['board']} exceeds {ceiling} MAE")
        measured.append({"index": control["index"], "board": control["board"], "set": board["set"],
                         "mean_mae": statistics.mean(tile_maes), "max_mae": max(tile_maes),
                         "calibration": calibration})
    report = {"state": "passed" if not errors else "failed", "errors": errors,
              "controls": len(measured), "tiles": len(measured)*64, "measurements": measured,
              "calibration_controls": 39, "rebuild_pages": 3, "runtime": runtime_identity(),
              "independent_identity_squares": identities_checked,
              "renderer_sha256": d.digest(d.REPO / "scripts/synthetic-render.mjs"),
              "control_sha256": d.digest(d.REPO / "scripts/synthetic-fidelity.mjs"),
              "checker_sha256": d.digest(__file__), "training_sha256": d.digest(d.REPO / "python/synthetic_training.py"),
              "provenance_sha256": d.digest(d.REPO / "provenance/public-bootstrap.json"),
              "truth": "agent-validated synthetic; no human truth claim"}
    for name,sheet in sheets.items():
        sheet.save(ROOT/f"production-{name}.png")
    d.write_json(ROOT / "report.json", report)
    print(json.dumps({k:v for k,v in report.items() if k not in {"measurements", "runtime"}},indent=2))
    d.require(not errors, "independent fidelity failed; no bulk synthesis")


if __name__ == "__main__":
    check()

"""Independent DOM-control comparison, production geometry and lazy tensor check."""
import json
from pathlib import Path
import statistics

import dataset_pipeline as d
import synthetic_training as t
from synthetic_job import runtime_identity
from synthetic_effect_reference import apply_effect

ROOT = d.REPO / "work/dataset/bootstrap/fidelity"


def tile_bytes(grid, index):
    x,y=index%8*96,index//8*96
    return grid.crop((x+3,y+3,x+93,y+93)).resize((24,24),d.pillow().Resampling.BOX).tobytes()


def audit_effects(records):
    audits=[r for r in records if r["recipe"]["index"]>=300000]
    d.require(len(audits)==12,"missing smallest-board effect audits")
    d.require({(r["recipe"]["boards"][0]["set"],r["recipe"]["condition"]["degradation"]["variant"]) for r in audits} == {(s,e) for s in ("chessnut","fantasy","rhosgfx") for e in ("blank","paper","faded","soft")},"incomplete effect/design coverage")
    minimum=float("inf")
    checked=0
    for r in audits:
        recipe=r["recipe"]; board=recipe["boards"][0]
        page=d.load_image(ROOT/f"control-{recipe['index']}-0.png")
        clean=d.rectify(page,board["corners"])
        effected=d.rectify(apply_effect(page,recipe["condition"]["degradation"]),board["corners"])
        actual=t.load_board(ROOT/r["image"],r,0)["grid"]
        templates=[{},{}]
        for i,label in enumerate(board["labels"]):
            key=(label,(i//8+i%8)%2)
            for target,grid in zip(templates,(clean,effected)):
                target.setdefault(key,tile_bytes(grid,i))
        d.require(all(len(target)==26 for target in templates),"missing class/background audit")
        for i,label in enumerate(board["labels"]):
            parity=(i//8+i%8)%2
            for target,grid in zip(templates,(effected,actual)):
                pixels=tile_bytes(grid,i)
                distances={p:sum(abs(v-w) for v,w in zip(pixels,target[(p,parity)])) for p in d.LABELS}
                margin=min(v for p,v in distances.items() if p!=label)-distances[label]
                minimum=min(minimum,margin)
                d.require(margin>0,f"effect identity ambiguity at audit {recipe['index']} square {i}")
            checked+=1
    return {"controls":12,"squares":checked,"minimum_class_margin":minimum,"state":"passed"}


def check():
    Image = d.pillow()
    records = json.loads((ROOT / "records.json").read_text())
    generation = d.read_json(ROOT / "generation.json")
    for key, path in [("effects_sha256", "scripts/synthetic-degradation.mjs"), ("perspective_sha256", "scripts/synthetic-perspective.mjs")]:
        d.require(generation[key] == d.digest(d.REPO/path), "effect code changed; rerender fidelity")
    d.require(generation["renderer_sha256"] == d.digest(d.REPO/"scripts/synthetic-render.mjs") and generation["control_sha256"] == d.digest(d.REPO/"scripts/synthetic-fidelity.mjs"), "generation code changed; rerender fidelity")
    controls = d.read_json(ROOT / "controls.json")
    by_index = {r["recipe"]["index"]: r for r in records}
    expected = {(r["recipe"]["index"], i) for r in records if r["recipe"]["kind"] == "boards" for i in range(len(r["recipe"]["boards"]))}
    d.require(len(records) == 91 and len(by_index) == 91, "missing fidelity pages")
    d.require({(c["index"], c["board"]) for c in controls} == expected and len(controls) == len(expected), "missing or repeated controls")
    calibration = [r for r in records if 100000 <= r["recipe"]["index"] < 300000]
    d.require(len(calibration) == 39, "missing calibration records")
    d.require({(r["recipe"]["boards"][0]["set"], r["recipe"]["boards"][0]["labels"][0]) for r in calibration} == {(s,p) for s in ("chessnut","fantasy","rhosgfx") for p in d.LABELS}, "class/design calibration incomplete")
    for r in calibration[:3]:
        d.require(d.digest(ROOT / "rebuild" / r["image"]) == r["image_sha256"], "rebuild differs")
    errors, measured = [], []
    templates = {}
    for r in calibration:
        board = r["recipe"]["boards"][0]
        ref = d.rectify(d.load_image(ROOT / f"control-{r['recipe']['index']}-0.png"), board["corners"])
        for parity in (0,1):
            templates[(board["set"], board["labels"][0], parity)] = ref.crop((parity*96+3,3,parity*96+93,93)).resize((24,24), Image.Resampling.BOX).tobytes()
    identities_checked = 0
    sheets = {s: Image.new("RGB", (1152, 232), "white") for s in ("chessnut","fantasy","rhosgfx")}
    for control in controls:
        record = by_index[control["index"]]
        board = record["recipe"]["boards"][control["board"]]
        actual = t.load_board(ROOT / record["image"], record, control["board"])
        if 100000 <= control["index"] < 300000 and board["labels"][0] != ".":
            number = d.LABELS.index(board["labels"][0])-1
            sheets[board["set"]].paste(actual["grid"].crop((0,0,192,96)), ((number%6)*192,(number//6)*116+20))
            from PIL import ImageDraw
            ImageDraw.Draw(sheets[board["set"]]).text(((number%6)*192,(number//6)*116), board["labels"][0], fill="black")
        reference_page = d.load_image(ROOT / control["image"])
        d.require(reference_page.size == (record["recipe"]["width"], record["recipe"]["height"]), "control page dimensions differ")
        reference_page = apply_effect(reference_page,record["recipe"]["condition"]["degradation"])
        homography = record["recipe"]["condition"].get("perspective")
        if homography:
            # Independently solve inverse page mapping for Pillow, not CSS matrix3d.
            width, height = reference_page.size
            equations, values = [], []
            for x,y in [(0,0),(width,0),(width,height),(0,height)]:
                z=homography[6]*x+homography[7]*y+1
                u=(homography[0]*x+homography[1]*y+homography[2])/z
                v=(homography[3]*x+homography[4]*y+homography[5])/z
                equations.extend([[u,v,1,0,0,0,-x*u,-x*v],[0,0,0,u,v,1,-y*u,-y*v]])
                values.extend([x,y])
            reference_page=reference_page.transform((width,height),Image.Transform.PERSPECTIVE,d.solve(equations,values),Image.Resampling.BICUBIC,fillcolor="white")
        reference = d.rectify(reference_page, board["corners"])
        tile_maes = []
        comparison_side = 24 if control["index"] >= 100000 else 48
        for i in range(64):
            x, y = i % 8 * 96, i // 8 * 96
            # Exclude grid edge antialiasing; retain all actual glyph content.
            a = actual["grid"].crop((x+3, y+3, x+93, y+93)).resize((comparison_side, comparison_side), Image.Resampling.BOX)
            b = reference.crop((x+3, y+3, x+93, y+93)).resize((comparison_side, comparison_side), Image.Resampling.BOX)
            mae = sum(abs(v-w) for v,w in zip(a.tobytes(),b.tobytes()))/(comparison_side**2*3)
            tile_maes.append(mae)
            if 100000 <= control["index"] < 300000:
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
        calibration = 100000 <= control["index"] < 300000
        # Fixed before this comparison: tight same-scale control, bounded relaxed
        # resampling/hatch tolerance on small transformed full-page diagrams.
        ceiling = 3 if calibration else 18
        if max(tile_maes) > ceiling:
            errors.append(f"control {control['index']} board {control['board']} exceeds {ceiling} MAE")
        measured.append({"index": control["index"], "board": control["board"], "set": board["set"],
                         "mean_mae": statistics.mean(tile_maes), "max_mae": max(tile_maes),
                         "calibration": calibration})
    try:
        effect_audit = audit_effects(records)
    except d.Invalid as error:
        effect_audit = {"state":"failed","error":str(error)}
        errors.append(str(error))
    report = {"state": "passed" if not errors else "failed", "errors": errors,
              "effect_audit": effect_audit,
              "controls": len(measured), "tiles": len(measured)*64, "measurements": measured,
              "calibration_controls": 39, "rebuild_pages": 3, "runtime": runtime_identity(),
              "independent_identity_squares": identities_checked,
              "renderer_sha256": d.digest(d.REPO / "scripts/synthetic-render.mjs"),
              "control_sha256": d.digest(d.REPO / "scripts/synthetic-fidelity.mjs"),
              "checker_sha256": d.digest(__file__), "training_sha256": d.digest(d.REPO / "python/synthetic_training.py"),
              "provenance_sha256": d.digest(d.REPO / "provenance/public-bootstrap.json"),
              "effects_sha256": generation["effects_sha256"], "perspective_sha256": generation["perspective_sha256"],
              "effect_reference_sha256": d.digest(d.REPO/"python/synthetic_effect_reference.py"),
              "truth": "agent-validated synthetic; no human truth claim"}
    for name,sheet in sheets.items():
        sheet.save(ROOT/f"production-{name}.png")
    d.write_json(ROOT / "report.json", report)
    print(json.dumps({k:v for k,v in report.items() if k not in {"measurements", "runtime"}},indent=2))
    d.require(not errors, "independent fidelity failed; no bulk synthesis")


if __name__ == "__main__":
    check()

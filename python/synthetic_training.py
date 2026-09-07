"""Lazy synthetic page/grid/tile interface using the candidate dataset preprocessing.

This is not a claim of parity with a promoted browser model. Qualification and
real human-reviewed records remain separate from generated truth.
"""
from __future__ import annotations

import array
from pathlib import Path
import sys

if __package__:
    from . import dataset_pipeline as d
else:
    import dataset_pipeline as d


def load_board(image_path, record, board_number):
    recipe = record["recipe"]
    d.require(recipe["kind"] == "boards", "partial/negative pages have no classifier targets")
    d.require(d.digest(image_path) == record["image_sha256"], "image changed")
    image = d.load_image(image_path)
    d.require(image.size == (recipe["width"], recipe["height"]), "page size changed")
    d.require(type(board_number) is int and 0 <= board_number < len(recipe["boards"]), "board index")
    board = recipe["boards"][board_number]
    labels = board["labels"]
    d.require(len(labels) == 64 and all(x in d.LABELS and len(x) == 1 for x in labels), "labels")
    grid = d.rectify(image, board["corners"])
    tensor = array.array("f")
    for index in range(64):
        x, y = index % 8 * 96, index // 8 * 96
        raw = grid.crop((x, y, x+96, y+96)).tobytes()
        for channel, (mean, std) in enumerate(zip((.485, .456, .406), (.229, .224, .225))):
            tensor.extend((raw[i]/255-mean)/std for i in range(channel, len(raw), 3))
    if sys.byteorder != "little":
        tensor.byteswap()
    return {"grid": grid, "tensor": tensor, "labels": [d.LABELS.index(x) for x in labels],
            "shape": [64, 3, 96, 96], "order": "image-relative row-major",
            "split": "train", "truth": "recipe-derived; caller must verify corpus fidelity gate", "browser_parity": "pending"}


def detector_targets(record):
    recipe = record["recipe"]
    d.require(recipe["kind"] in {"boards", "negative"}, "unsupported page is not a negative")
    result = []
    for board in recipe["boards"]:
        xs, ys = zip(*board["corners"])
        result.append([(min(xs)+max(xs))/2/recipe["width"],
                       (min(ys)+max(ys))/2/recipe["height"],
                       (max(xs)-min(xs))/recipe["width"],
                       (max(ys)-min(ys))/recipe["height"]])
    return result

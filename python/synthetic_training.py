"""Lazy synthetic page/grid/tile interface using the candidate dataset preprocessing.

This is not a claim of parity with a promoted browser model. Qualification and
real human-reviewed records remain separate from generated truth.
"""
from __future__ import annotations

from pathlib import Path

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
    grid = d.rectify_classifier_grid(image, board["corners"])
    tensor = d.classifier_preprocessing.classifier_tensor(grid.tobytes())
    return {"grid": grid, "tensor": tensor, "labels": [d.LABELS.index(x) for x in labels],
            "shape": [64, 3, 96, 96], "order": "image-relative row-major",
            "split": "train", "truth": "recipe-derived; caller must verify corpus fidelity gate",
            "browser_contract": "deterministic-bilinear-rgb96-v1"}


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

# Existing public chess datasets and historical increment

Primary-source screen, 2026-09-07. This is a scope/rights review, not a downloaded
dataset, label audit or recognition comparison.

| Candidate | Primary evidence | Role and remaining restriction |
| --- | --- | --- |
| SLCC | [ChessQueries](https://github.com/JSeytre/chessqueries) | Physical broadcast boards, not printed diagrams. Annotation noncommercial terms do not grant rights to broadcast frames, which are not redistributed. No acquisition or adaptation started. |
| ChessReD | [Original repository](https://github.com/tmasouris/end-to-end-chess-recognition), [data record](https://data.4tu.nl/datasets/99b5c721-280b-450b-b058-b2900b69a90f) | Smartphone physical-board photos, largely one board/set lineage. Authoritative data-license page was inaccessible to the lookup tool; conflicting secondary license descriptions were not treated as permission. Not admitted. |
| chesscog | [Original repository](https://github.com/georg-wolflein/chesscog), [dataset DOI](https://doi.org/10.17605/OSF.IO/XF3KA) | Blender 3D boards. Repository MIT code license does not establish dataset, model or 3D-asset rights. Data record was not accessible in this screen; not admitted. |

These could inform a separately bounded transfer or physical-board screen. None
replaces printed-page training, source-held-out development, or qualification.

## Reviewed historical acquisition

[Public metadata](../../provenance/public-historical-increment.json) pins three
original scans and the captured rights evidence. Each selection is PDF pages
20–43, frozen before inference, including text-only pages. These are candidates,
not accepted truth. No human annotation or recognition score is published here.

Capablanca's 1921 Harcourt edition supplies one additional TRAIN source/design
group. De Witt's 1880 manual and Staunton's 1848 handbook have closely related
engraved king/pawn artwork in the inspected pages; conservatively keep both in
one reserved QUALIFICATION group. Do not claim two independent groups from two
book titles. Their rights/lineage previews are not model evaluation. No proposals
or scoring may run on this reserved group before the candidate freeze.

This adds historical hatch/ink/scan/page-layout coverage. It does not provide the
missing modern-document diversity or a reserved development group. All three
remain subject to page review, fuller artwork/duplicate audit and preserved
single-human acceptance before they can serve as real truth. Unseen pretraining
overlap remains unknown. The Staunton scan lacks some later printed pages; the
fixed PDF selection is not a claim to reconstruct a complete original edition.

The Commons permanent revisions bind the evidence content; server-generated HTML
may change even for the same revision, so the captured HTML SHA-256 identifies
the reviewed local evidence snapshot, not guaranteed byte-identical future HTML.
The original PDF hashes must match exactly. No glyph/font extraction or original,
crop, dataset or model publication is authorized by this metadata review.

## Additional printed-page review

- [Manual de Xadrez](https://commons.wikimedia.org/wiki/File:Manual_xadrez.pdf)
  is a 52-page modern educational manual attributed to Hélio Neto. Commons
  asserts an author public-domain dedication, but the upload is not clearly by
  that author and the additional government-work rationale does not establish
  rights for this 2018 work. Obtain corroborating primary dedication evidence
  before admission; embedded artwork lineage also needs visual inspection.
- [Chess Wikibook PDF](https://commons.wikimedia.org/wiki/File:Chess.pdf)
  supplies a fixed modern textbook layout under stated CC BY-SA 3.0/GFDL terms.
  Shared Wikimedia diagram artwork may overlap other sources. The byte-pinned
  2006 PDF is now admitted for fixed pages 20–43, TRAIN layout coverage only,
  under [reviewed reconstruction metadata](../../provenance/public-wikibook-increment.json).
  A bounded pixel inspection confirmed printed color-board/text layout. It is
  not an independently established held-out artwork group; no human labels exist.
- [ChessPrint](https://github.com/Nairwolf/chessprint) is a printable exercise
  generator, not a fixed real-page corpus. Its bundled artwork has separate
  licenses and overlaps the Lichess ecosystem; it does not close real-source
  or qualification-diversity gates.

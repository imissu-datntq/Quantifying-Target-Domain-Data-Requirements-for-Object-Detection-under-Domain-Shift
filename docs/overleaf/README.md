# IEEE paper (Overleaf project)

English IEEE-format (IEEEtran, conference, two columns) version of the updated results report
`docs/bao_cao_ket_qua_cap_nhat_2026-11-23.docx`. Every number comes from that report, and the figures are byte-identical copies
of `docs/results/` (refresh both when the analysis is rerun).

## Import into Overleaf

1. Zip this folder's contents (`main.tex` must be at the top level of the zip), or use `docs/overleaf_ieee.zip`.
2. Overleaf: **New Project → Upload Project** → choose the zip.
3. Menu → Settings: **Compiler: pdfLaTeX**, **Main document: main.tex**. BibTeX runs automatically.

`IEEEtran.cls` and `IEEEtran.bst` are part of Overleaf's TeX Live, so they are not included.

## Layout

| Path | Content |
| --- | --- |
| `main.tex` | Preamble, notation macros, title, abstract, keywords, section includes |
| `sections/01_introduction_related.tex` | I. Introduction, II. Related work |
| `sections/02_data_method.tex` | III. Data and evaluation protocol, IV. Method |
| `sections/03_main_results.tex` | V. Main result: the control-arm decomposition (RQ5) |
| `sections/04_curves_thresholds.tex` | VI. Data-efficiency curves (RQ1, RQ2, ceilings), VII. Thresholds (RQ3, RQ4) |
| `sections/05_secondary_discussion_conclusion.tex` | VIII. Secondary results, IX. Discussion, X. Limitations, XI. Conclusion |
| `sections/06_appendix.tex` | Appendix: reproducibility |
| `references.bib` | Bibliography (ALDI cited as arXiv:2403.12029 only) |
| `figures/` | The six figures of the report |

## Before submission

- The author block in `main.tex` is a placeholder.
- Author lists, venues and pages in `references.bib` were checked against CVF open access, arXiv, Springer and NeurIPS
  proceedings on 2026-11-23. ALDI stays cited as arXiv:2403.12029 only, by the authors' choice.

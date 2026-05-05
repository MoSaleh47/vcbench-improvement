# NeurIPS LaTeX Paper Folder

This folder contains a NeurIPS-format LaTeX version of the current VCBench paper
draft.

The Overleaf template requested by the user was:

```text
https://www.overleaf.com/latex/templates/neurips-2024/tpsbbrdqcmsh
```

As of 2026-05-05, that URL serves the official "Formatting Instructions For
NeurIPS 2026" template and uses:

```latex
\usepackage{neurips_2026}
```

The source page states that the only supported style file is
`neurips_2026.sty`. I did not commit a third-party mirror of that style file.
To compile:

1. Open the official Overleaf template above.
2. Upload `main.tex`, `references.bib`, and `checklist.tex` from this folder.
3. Keep the official `neurips_2026.sty` supplied by the Overleaf template.
4. Compile with `pdflatex` + `bibtex`, or use Overleaf's default build.

Files:

- `main.tex`: paper source in NeurIPS format.
- `references.bib`: BibTeX references used by the paper.
- `checklist.tex`: NeurIPS-style checklist answers.

Local compilation was not run in this workspace because `pdflatex`/`latexmk`
were not installed.

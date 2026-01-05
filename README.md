# Dynamic MTT

This code generates the figures used for simulating dynamic catalytic phenomena using the matrix-tree-theorem.

## Dependencies

This package relies on [snakemake][https://snakemake.readthedocs.io/en/stable/index.html] and [conda][https://anaconda.org/anaconda/conda] to log and perform workflows.

## Usage
To build the package and generate figures, run `snakemake --sdm conda --cores 2`. Figures are stored in `workflow/resources/figures/`.

## Limitations
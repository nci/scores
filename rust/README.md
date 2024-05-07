This is a completely experimental part of scores - use at your own risk.

## Quick Description

Run the following, ensuring "maturin" is the `build-backend` configured in
`pyproject.toml`.

If you want to use the default `build-backend` such as hatchling, you'll need to
run an additional step to build the binaries (`maturin develop --release`).

```sh
# ACTIVATE A VIRTUAL/CONDA ENV BEFORE THIS
pip install maturin numba
maturin develop --release  # technically not required if build-backend is maturin
pip install -e .
```

## Included Scores

- **FSS: Fractions Skill Score**
    - does a comparision of rust v.s. numba v.s. [fast_fss - numpy](https://github.com/nathan-eize/fss/blob/master/fss_core.py)
    ```sh
    # ACTIVATE APPROPRIATE ENV HERE
    cd examples/  # or rust/examples from project root
    python rust_fss_fast.py
    ```
    - reference: https://www.researchgate.net/publication/269222763_Fast_calculation_of_the_Fractions_Skill_Score

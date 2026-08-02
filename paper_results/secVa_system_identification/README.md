# Section Va: System Identification

This case identifies the soft-tentacle parameters, evaluates the learned
residual model, and produces the paper's shape and error summaries. The 22
motion-capture CSV inputs and canonical derived arrays are committed in `data/`.

Install the paper dependencies:

```bash
uv sync --extra paper_results
```

Run parameter identification into a temporary directory first:

```bash
uv run python paper_results/secVa_system_identification/code/identify_soft_tentacle_parameters.py \
  --output-data-dir /tmp/soromox-secVa-parameters --no-show
```

The residual workflow consumes the fitted model, force arrays, parameter result
arrays, and motion-capture CSVs from one data directory. To regenerate the
canonical case-local arrays explicitly:

```bash
uv run python paper_results/secVa_system_identification/code/identify_soft_tentacle_residual.py \
  --force --no-show
```

Plot the committed or regenerated arrays without rerunning identification:

```bash
uv run python paper_results/secVa_system_identification/code/plot_system_id_residual.py
```

The parameter and residual optimizations are computationally intensive. Their
CLIs refuse to overwrite generated data unless `--force` is supplied.

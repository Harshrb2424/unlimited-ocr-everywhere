# Contributing

Thanks for helping make Unlimited-OCR-2 easier to run on ordinary hardware.

## Good first contributions

- **Hardware reports.** You ran a notebook on a GPU/CPU/Mac we did not test?
  Open a hardware report issue with your specs, settings, and result.
- **Error messages.** If a failure confused you, propose clearer text.
- **Docs.** Fix a step that assumed too much, add a screenshot, clarify a flag.
- **New tiers.** A Kaggle notebook, a Docker image, a RunPod template, a
  Windows-specific setup script.
- **Bug fixes** in the notebooks, CLI, or post-processing pipeline.

## Before you open a PR

1. Run the validator:
   ```bash
   python tools/validate_notebooks.py
   ```
2. If you changed a notebook, change the template in `tools/templates/` and
   regenerate:
   ```bash
   python tools/build_notebooks.py
   ```
   Do not hand-edit the generated `.ipynb` files unless the change is truly
   notebook-only (for example, an output or a metadata tweak).
3. Keep the hardware honesty. If a tier is experimental, say so. Do not remove
   warnings to make a table look better.
4. Update the README table and `docs/HARDWARE.md` if you add a tier.

## Notebook conventions

- One notebook per hardware tier.
- The first code cell installs dependencies; the second configures; the third
  loads the model; the rest are shared helpers.
- Every notebook must be standalone. Do not require a clone.
- Keep `transformers==4.52.1` pinned.
- Include a tier-specific troubleshooting section.
- Do not add outputs or execution counts to generated notebooks.

## Commit messages

Use a short imperative subject:

```
Add Kaggle T4 notebook
Fix MPS OOM message
Document CUDA 12.4 wheel
```

## Reporting a bug

Use the bug report template. Include:

- Notebook or CLI command
- OS, Python, GPU/CPU, memory
- Exact traceback
- Whether it failed during install, model load, or inference
- `name.meta.json` if the failure is document-specific

## Code of conduct

Be kind. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

By contributing, you agree that your contribution is licensed under the MIT
License, the same as this repository.

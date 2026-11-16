#!/bin/bash
# Source in the submitter AND job: do not rely on the submitting shell's active env.
PNG_CONDA_ENV=${PNG_CONDA_ENV:-lama-h200}
export PNG_CONDA_ENV
if [[ "$PNG_CONDA_ENV" != current ]]; then
  if [[ -n ${CONDA_EXE:-} && -x "$CONDA_EXE" ]]; then
    png_conda_base=$("$CONDA_EXE" info --base)
  elif type conda >/dev/null 2>&1; then
    png_conda_base=$(conda info --base)
  else
    echo 'Conda is unavailable. Initialize/load Conda in your login shell before submitting.' >&2
    return 2
  fi
  source "$png_conda_base/etc/profile.d/conda.sh"
  conda activate "$PNG_CONDA_ENV"
  unset png_conda_base
fi

NP_INC=$(python - <<'PY'
import numpy as np; print(np.get_include())
PY
)

CFLAGS="-I${NP_INC}" python -m Cython.Build.Cythonize -i -3 src/soccer_pitch/pitchgeom.pyx
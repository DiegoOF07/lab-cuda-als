# lab-cuda-als

Comparación CPU vs GPU de ALS (*Alternating Least Squares*) para recomendar 10 películas por usuario sobre MovieLens 32M.

- **CPU:** NumPy + SciPy.
- **GPU:** CuPy (cuBLAS, cuSPARSE, cuSOLVER) con **el mismo código de ALS**; solo cambia el módulo `xp`.
- **Top-10 en GPU:** kernel CUDA propio (`src/kernels.cu`), compilado en tiempo de ejecución con NVRTC a través de `cupy.RawModule`. No requiere `nvcc`.

## Requisitos

- GPU NVIDIA con driver reciente (probado con CUDA 13.5 en WSL2, CuPy 14.2 y Python 3.12).
- ~8 GB de RAM y ~4 GB de memoria de GPU libres para el 100 % de los datos.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install numpy scipy pandas matplotlib "cupy-cuda13x[ctk]"   # con driver CUDA 12.x: cupy-cuda12x[ctk]
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceProperties(0)['name'])"
```

## Datos

```bash
mkdir -p data && cd data
curl -LO https://files.grouplens.org/datasets/movielens/ml-32m.zip && unzip ml-32m.zip
```

`data/` no se versiona. Si el repo está en una carpeta sincronizada (OneDrive) o en `/mnt/c` desde WSL, conviene guardar los datos en el sistema de archivos de Linux y pasar la ruta con `--data`. Así la etapa de lectura no queda inflada.

## Uso

```bash
cd src
python bench.py --data ../data/ml-32m --out ../results   # 10 %, 50 % y 100 %: ~45 min
python plots.py --results ../results
```

Opciones útiles de `bench.py`:

| Opción | Valor por defecto | Descripción |
|---|---|---|
| `--fracs` | `0.1 0.5 1.0` | fracciones del dataset |
| `--reps` | `3` | repeticiones por medición (se reporta la mediana) |
| `--cpu-reps-full` | `1` | repeticiones en CPU con el 100 % |
| `--devices` | `cpu gpu` | dispositivos a medir |
| `--k`, `--iters`, `--lam` | `32`, `10`, `0.05` | hiperparámetros de ALS |
| `--no-gpu-argpartition` | | omite el top-10 de referencia con `cupy.argpartition` |

Prueba rápida: `python bench.py --data ../data/ml-32m --out /tmp/res --fracs 0.02 --reps 1 --iters 2`.

## Salida (`results/`)

| Archivo | Contenido |
|---|---|
| `results.csv` | una fila por (fracción, etapa, dispositivo): mediana en `tiempo_s`, cada repetición en `tiempos` y RMSE de test |
| `summary.csv` | speedup por etapa y total, y Ley de Amdahl (`p_paralelo`, `amdahl_max`, `amdahl_predicho`) |
| `env.json` | hardware, versiones y parámetros de la corrida |
| `top10_muestra.csv` | top-10 de los 5 primeros usuarios con 100 % de los datos |
| `tiempos_por_etapa.png`, `escalado.png`, `speedup.png` | gráficas |

## Etapas medidas

| Etapa | Dispositivo | Qué mide |
|---|---|---|
| `lectura_csv` | CPU | `pandas.read_csv` del archivo completo. Se lee una sola vez y el mismo valor se reporta en cada fracción, porque toda corrida lee el archivo entero |
| `construir_csr` | CPU | submuestreo, reindexado, split 90/10 y matrices CSR |
| `transferencia` | GPU | copia de `R`, `Rᵀ` y del conjunto de test a la GPU |
| `als_10_iter` | CPU / GPU | 10 iteraciones de ALS (ALS-WR, regularización `λ·n_u`) |
| `top10` | CPU / GPU | `U @ Vᵀ` por bloques más selección del top-10. En CPU usa `argpartition` de NumPy; en GPU, el kernel propio |
| `top10_argpartition` | GPU | referencia: top-10 con `cupy.argpartition` (1 repetición) |

En GPU se sincroniza el dispositivo antes de detener el reloj, y antes de medir se hace un calentamiento al 1 % que además verifica que el kernel coincida con CuPy.

## Notas metodológicas

- **Verificación:** CPU y GPU parten de la misma inicialización de `V` (semilla 42), así que el RMSE debe coincidir hasta ~1e-6.
- **Paralelismo en CPU:** el producto disperso × denso de SciPy y el `solve` por lotes de NumPy corren prácticamente en un hilo. Solo `U @ Vᵀ` usa BLAS multihilo. El speedup del ALS es contra esa implementación.
- **Top-10:** no excluye las películas ya vistas. Las películas con 1 o 2 calificaciones reciben factores casi idénticos, así que hay empates frecuentes y el orden entre empatados es arbitrario. Por eso la verificación compara puntajes, no índices.
- **Test:** se descartan los pares cuyo usuario o película no aparece en el entrenamiento (*cold start*).
- **Submuestreo:** las fracciones son prefijos de una misma permutación, de modo que el 10 % está contenido en el 50 %, y este en el 100 %.

## Estructura

```
src/
├── load.py       # lectura del CSV, submuestreo, split y CSR; copia a la GPU
├── als.py        # ALS genérico (xp = numpy | cupy) y RMSE
├── kernels.cu    # kernel CUDA de selección top-10 por fila
├── topk.py       # top-10 con argpartition (CPU/GPU) y con el kernel
├── bench.py      # benchmark completo → results.csv, env.json
└── plots.py      # gráficas y summary.csv (speedup, Amdahl)
```

## Fuentes

- F. M. Harper y J. A. Konstan. 2015. *The MovieLens Datasets: History and Context*. ACM TiiS 5(4).
- Y. Zhou et al. 2008. *Large-scale Parallel Collaborative Filtering for the Netflix Prize* (ALS-WR).

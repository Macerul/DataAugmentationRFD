"""
=============================================================
DISTANCE MATRIX BENCHMARK  v2.0
=============================================================
Distanza: D[i,j] = sum_k |a_k - b_k|  (L1 / Manhattan)

Metodi:
  01 Naive Python puro          - baseline O(n²·d)
  02 NumPy broadcasting         - vettorizzazione N×N×D in RAM
  03 NumPy chunk (memory-safe)  - broadcasting a blocchi 64×64
  04 NumPy einsum               - contrazione tensoriale
  05 SciPy cdist L1             - wrapper C, metric='cityblock'
  06 SciPy pdist L1+squareform  - triangolo superiore + simmetria
  07 Multiprocessing Pool       - N processi OS paralleli
  08 ThreadPoolExecutor         - N thread (scipy rilascia GIL)
  09 Tile-based cache-friendly  - blocchi 32×32 in cache L1
  10 Parquet I/O pipeline       - write→read parquet→calcola→salva
  11 GPU-sim SIMD float32       - tile 128, AVX2/SSE4 friendly
  12 Vectorized L1 stride       - riga per riga, basso picco RAM

Ogni metodo ha un timeout di 3 minuti (--timeout per cambiarlo).

Sorgenti dati:
  --csv PATH           CSV esistente
  --parquet PATH       Parquet esistente
  --synth N D          Dataset sintetico N righe × D colonne

Uso:
  python distance_matrix_benchmark.py --csv data.csv
  python distance_matrix_benchmark.py --synth 1000 50
  python distance_matrix_benchmark.py --synth 500 20 --timeout 60
=============================================================
"""

import sys, os, time, gc, argparse, hashlib, warnings, traceback
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
import pandas as pd
import psutil
import scipy.spatial.distance as spd

warnings.filterwarnings("ignore")

# ───────────────────────────── COSTANTI ─────────────────────────────
DEFAULT_TIMEOUT  = 180
NAIVE_SKIP_N     = 300
MAX_WORKERS      = min(32, max(1, multiprocessing.cpu_count()))

# ─────────────────────────── UTILS ──────────────────────────────────
def get_mem():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2

def fingerprint(mat):
    return hashlib.md5(np.round(mat.astype(np.float64), 4).tobytes()).hexdigest()[:12]

def save_csv(mat, path):
    pd.DataFrame(mat).to_csv(path, index=False)

def run_with_timeout(fn, args, timeout):
    import threading
    result = [None]; err = [None]
    def t():
        try: result[0] = fn(*args)
        except Exception as e: err[0] = e
    th = threading.Thread(target=t, daemon=True)
    th.start(); th.join(timeout)
    if th.is_alive(): return None, True
    if err[0]: raise err[0]
    return result[0], False

# ───────────────────────── METODI ───────────────────────────────────

def method_naive_python(X):
    n, d = X.shape
    D = np.zeros((n, n))
    data = X.tolist()
    for i in range(n):
        ri = data[i]
        for j in range(i+1, n):
            rj = data[j]
            s = sum(abs(ri[k]-rj[k]) for k in range(d))
            D[i][j] = D[j][i] = s
    return D

def method_numpy_broadcasting(X):
    diff = X[:,np.newaxis,:] - X[np.newaxis,:,:]
    return np.abs(diff).sum(axis=2)

def method_numpy_chunk(X, chunk=64):
    n = X.shape[0]
    D = np.zeros((n,n))
    for i in range(0,n,chunk):
        Xi = X[i:i+chunk]
        for j in range(0,n,chunk):
            Xj = X[j:j+chunk]
            diff = Xi[:,np.newaxis,:] - Xj[np.newaxis,:,:]
            D[i:i+chunk, j:j+chunk] = np.abs(diff).sum(axis=2)
    return D

def method_numpy_einsum(X):
    diff = X[:,np.newaxis,:] - X[np.newaxis,:,:]
    np.abs(diff, out=diff)
    return np.einsum("ijk->ij", diff)

def method_scipy_cdist_l1(X):
    return spd.cdist(X, X, metric="cityblock")

def method_scipy_pdist_l1(X):
    return spd.squareform(spd.pdist(X, metric="cityblock"))

# worker deve stare a livello modulo per pickle
def _mp_worker(args):
    Xb, shape, i0, i1 = args
    X = np.frombuffer(Xb, dtype=np.float64).reshape(shape)
    return i0, i1, spd.cdist(X[i0:i1], X, metric="cityblock")

def method_multiprocessing(X):
    n = X.shape[0]
    Xc = np.ascontiguousarray(X, dtype=np.float64)
    Xb = Xc.tobytes()
    cs = max(1, -(-n // MAX_WORKERS))
    chunks = [(Xb, X.shape, i, min(i+cs,n)) for i in range(0,n,cs)]
    D = np.zeros((n,n))
    with Pool(processes=MAX_WORKERS) as pool:
        for i0,i1,block in pool.map(_mp_worker, chunks):
            D[i0:i1,:] = block
    return D

def _thr_worker(X, i0, i1):
    return i0, i1, spd.cdist(X[i0:i1], X, metric="cityblock")

def method_threadpool(X):
    n = X.shape[0]
    cs = max(1, -(-n // MAX_WORKERS))
    segs = [(i, min(i+cs,n)) for i in range(0,n,cs)]
    D = np.zeros((n,n))
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(_thr_worker, X, s, e) for s,e in segs]
        for f in futs:
            i0,i1,block = f.result()
            D[i0:i1,:] = block
    return D

def method_tile_cache(X, tile=32):
    n = X.shape[0]
    D = np.zeros((n,n))
    for i in range(0,n,tile):
        ei = min(i+tile,n); Xi = X[i:ei]
        for j in range(0,n,tile):
            ej = min(j+tile,n); Xj = X[j:ej]
            diff = Xi[:,np.newaxis,:] - Xj[np.newaxis,:,:]
            D[i:ei,j:ej] = np.abs(diff).sum(axis=2)
    return D

def method_parquet_pipeline(X, output_dir):
    """Scrive X in parquet, rilegge, calcola L1, salva matrice in parquet."""
    fin  = os.path.join(output_dir, "_in.parquet")
    fout = os.path.join(output_dir, "_out.parquet")
    pd.DataFrame(X).to_parquet(fin)
    Xl = pd.read_parquet(fin).to_numpy(dtype=np.float64)
    D  = spd.cdist(Xl, Xl, metric="cityblock")
    pd.DataFrame(D.astype(np.float32)).to_parquet(fout)
    for f in [fin, fout]:
        try: os.remove(f)
        except: pass
    return D

def method_gpu_sim_float32(X):
    """
    Simula pipeline GPU-like (CUDA L1 kernel) su CPU:
    - float32: doppia larghezza SIMD (AVX2: 8 floats vs 4 doubles)
    - Tile 128 = dimensione warp CUDA tipica
    - Layout C-contiguo = coalescenza di memoria come in CUDA
    """
    Xf = np.ascontiguousarray(X, dtype=np.float32)
    n  = Xf.shape[0]
    D  = np.empty((n,n), dtype=np.float32)
    T  = 128
    for i in range(0,n,T):
        ei = min(i+T,n); Ai = Xf[i:ei]
        for j in range(0,n,T):
            ej = min(j+T,n); Bj = Xf[j:ej]
            diff = Ai[:,np.newaxis,:] - Bj[np.newaxis,:,:]
            np.abs(diff, out=diff)
            D[i:ei,j:ej] = diff.sum(axis=2)
    return D.astype(np.float64)

def method_vectorized_l1_stride(X):
    """
    Riga per riga: np.abs(X - X[i]).sum(axis=1)
    Picco RAM = O(n·d) invece di O(n²·d).
    Ottimale per n grande, d piccolo.
    """
    n = X.shape[0]
    D = np.empty((n,n))
    for i in range(n):
        D[i] = np.abs(X - X[i]).sum(axis=1)
    return D

# ─────────────────────── REGISTRY ───────────────────────────────────

METHODS = [
    ("01_naive_python",         method_naive_python,        "Python puro – O(n²·d) baseline"),
    ("02_numpy_broadcasting",   method_numpy_broadcasting,  "NumPy broadcasting N×N×D"),
    ("03_numpy_chunk",          method_numpy_chunk,         "NumPy chunk 64×64 (memory-safe)"),
    ("04_numpy_einsum",         method_numpy_einsum,        "NumPy einsum abs+contrazione"),
    ("05_scipy_cdist_l1",       method_scipy_cdist_l1,      "SciPy cdist cityblock (L1, C)"),
    ("06_scipy_pdist_l1",       method_scipy_pdist_l1,      "SciPy pdist L1 + squareform"),
    ("07_multiprocessing",      method_multiprocessing,     f"Multiprocessing Pool ({MAX_WORKERS}w)"),
    ("08_threadpool",           method_threadpool,          f"ThreadPoolExecutor ({MAX_WORKERS}t)"),
    ("09_tile_cache",           method_tile_cache,          "Tile cache-friendly 32×32"),
    ("10_parquet_pipeline",     None,                       "Parquet I/O pipeline (I/O incluso)"),
    ("11_gpu_sim_float32",      method_gpu_sim_float32,     "GPU-sim SIMD float32 tile-128"),
    ("12_vectorized_l1_stride", method_vectorized_l1_stride,"Vectorized L1 stride – basso RAM"),
]

# ─────────────────────── CARICAMENTO DATI ───────────────────────────

def load_data(args):
    if args.synth:
        n, d = args.synth
        X = np.random.default_rng(42).standard_normal((n, d))
        print(f"🎲 Dataset sintetico: {n} righe × {d} col  "
              f"| matrice ≈ {n*n*8/1024**2:.1f} MB")
        return X
    if args.csv:
        print(f"📂 CSV: {args.csv}")
        df = pd.read_csv(args.csv)
    elif args.parquet:
        print(f"📂 Parquet: {args.parquet}")
        df = pd.read_parquet(args.parquet)
    else:
        raise ValueError("Specifica --csv, --parquet o --synth N D")
    X = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    n, d = X.shape
    print(f"   {n} righe × {d} col | matrice ≈ {n*n*8/1024**2:.1f} MB")
    return X

# ─────────────────────── RUNNER PRINCIPALE ──────────────────────────

def run_benchmark(X, output_dir, timeout_sec, skip_naive):
    n, d = X.shape
    os.makedirs(output_dir, exist_ok=True)
    results = []
    ref_fp  = None

    print(f"\n{'═'*65}")
    print(f"  n={n}  d={d}  timeout={timeout_sec}s  workers={MAX_WORKERS}")
    print(f"{'═'*65}")

    for key, fn, desc in METHODS:
        is_naive   = "naive"   in key
        is_parquet = "parquet" in key

        if is_naive and (skip_naive or n > NAIVE_SKIP_N):
            print(f"\n  ⏭  [{key}] SALTATO (n={n} > {NAIVE_SKIP_N})")
            results.append(_row(key, desc, "SALTATO","SALTATO","SALTATO","N/A",n,d))
            continue

        print(f"\n▶  [{key}]  {desc}")

        call_fn   = (lambda X, od=output_dir: method_parquet_pipeline(X, od)) if is_parquet else fn
        call_args = (X,)

        gc.collect()
        mem0 = get_mem()
        t0   = time.perf_counter()

        try:
            mat, timed_out = run_with_timeout(call_fn, call_args, timeout_sec)
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            mem_d   = max(get_mem()-mem0, 0)
            print(f"     ❌  ERRORE {elapsed:.2f}s: {exc}")
            results.append(_row(key,desc,f"ERR({elapsed:.2f}s)",f"{mem_d:.1f}","ERR","ERR",n,d))
            gc.collect(); continue

        elapsed = time.perf_counter() - t0
        mem_d   = max(get_mem()-mem0, 0)

        if timed_out:
            print(f"     ⏰  TIMEOUT (>{timeout_sec}s)")
            results.append(_row(key,desc,f"TIMEOUT(>{timeout_sec}s)",f"{mem_d:.1f}","TO","TIMEOUT",n,d))
            gc.collect(); continue

        if mat is None:
            results.append(_row(key,desc,"ERR_NONE",f"{mem_d:.1f}","ERR","ERR",n,d))
            gc.collect(); continue

        fp = fingerprint(mat)
        if ref_fp is None:
            ref_fp = fp; corretto = "REF"
        elif fp == ref_fp:
            corretto = "✓"
        elif "float32" in key or "gpu_sim" in key:
            corretto = "~OK(f32)"
        else:
            corretto = "⚠DIVERSO"

        print(f"     ✅  {elapsed:.4f}s | RAM +{mem_d:.1f}MB | FP:{fp} | {corretto}")

        mat_path = os.path.join(output_dir, f"distance_matrix_{key}.csv")
        save_csv(mat, mat_path)
        print(f"     💾  {mat_path}")

        results.append(_row(key,desc,f"{elapsed:.4f}",f"{mem_d:.1f}",fp,corretto,n,d))
        gc.collect()

    # ── salva & stampa ──
    bench_df   = pd.DataFrame(results)
    bench_path = os.path.join(output_dir, "benchmark_results.csv")
    bench_df.to_csv(bench_path, index=False)
    _summary(bench_df, bench_path)
    return bench_df

def _row(key, desc, tempo, mem, fp, corretto, n, d):
    return {"metodo":key,"descrizione":desc,"tempo_sec":tempo,
            "mem_delta_mb":mem,"fingerprint":fp,"corretto":corretto,
            "n_righe":n,"n_feature":d}

def _summary(df, path):
    print(f"\n{'═'*65}")
    print(f"📊  {path}")
    print(f"{'═'*65}")
    df2 = df.copy()
    df2["_t"] = pd.to_numeric(df2["tempo_sec"], errors="coerce")
    df2 = df2.dropna(subset=["_t"]).sort_values("_t")
    print(f"{'Metodo':<32} {'Tempo(s)':<11} {'RAM(MB)':<10} Corretto")
    print("─"*65)
    for _, r in df2.iterrows():
        print(f"{r['metodo']:<32} {float(r['_t']):<11.4f} {str(r['mem_delta_mb']):<10} {r['corretto']}")
    if len(df2) >= 2:
        f = df2.iloc[0]; s = df2.iloc[-1]
        print(f"\n🏆  Più veloce: {f['metodo']}  ({float(f['_t']):.4f}s)")
        if float(f["_t"]) > 0:
            print(f"⚡  Speedup:    {float(s['_t'])/float(f['_t']):.1f}× su {s['metodo']}")

# ─────────────────────── ENTRY POINT ────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Benchmark matrice distanze L1  D[i,j]=Σ|a_k-b_k|",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python distance_matrix_benchmark.py --csv data.csv
  python distance_matrix_benchmark.py --parquet data.parquet
  python distance_matrix_benchmark.py --synth 1000 30
  python distance_matrix_benchmark.py --synth 500 20 --timeout 60 --output-dir out/
        """)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--csv",     metavar="PATH")
    src.add_argument("--parquet", metavar="PATH")
    src.add_argument("--synth",   metavar=("N","D"), nargs=2, type=int)
    p.add_argument("--output-dir", default="benchmark_output")
    p.add_argument("--timeout",    type=float, default=DEFAULT_TIMEOUT)
    p.add_argument("--skip-naive", action="store_true")
    args = p.parse_args()

    if not any([args.csv, args.parquet, args.synth]):
        if os.path.exists("sample_data.csv"):
            args.csv = "sample_data.csv"
        else:
            print("Nessuna sorgente → sintetico 300×20"); args.synth = [300,20]

    X = load_data(args)
    run_benchmark(X, args.output_dir, args.timeout, args.skip_naive)

if __name__ == "__main__":
    main()
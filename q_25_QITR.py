"""
 QITR - Quantum Imaginary Time Regression
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import random
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.utils import algorithm_globals

SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED

CSV_DRAWN = "/data/loto7hh_4582_k22.csv"
CSV_ALL   = "/data/kombinacijeH_39C7.csv"

MIN_VAL = [1, 2, 3, 4, 5, 6, 7]
MAX_VAL = [33, 34, 35, 36, 37, 38, 39]
NUM_QUBITS = 5
ITE_STEPS = 8
ITE_TAU = 0.2
LAMBDA_REG = 0.01


def load_draws():
    df = pd.read_csv(CSV_DRAWN)
    return df.values


def build_empirical(draws, pos):
    n_states = 1 << NUM_QUBITS
    freq = np.zeros(n_states)
    for row in draws:
        v = int(row[pos]) - MIN_VAL[pos]
        if v >= n_states:
            v = v % n_states
        freq[v] += 1
    return freq / freq.sum()


def build_hamiltonian(draws, pos):
    n_states = 1 << NUM_QUBITS
    emp = build_empirical(draws, pos)

    H = np.zeros((n_states, n_states))

    for i in range(n_states):
        H[i, i] = -np.log(max(emp[i], 1e-10))

    for k in range(len(draws) - 1):
        v1 = int(draws[k][pos]) - MIN_VAL[pos]
        v2 = int(draws[k + 1][pos]) - MIN_VAL[pos]
        if v1 >= n_states:
            v1 = v1 % n_states
        if v2 >= n_states:
            v2 = v2 % n_states
        H[v1, v2] -= 0.01
        H[v2, v1] -= 0.01

    H = (H + H.T) / 2.0
    return H


def imaginary_time_evolution(H, steps=ITE_STEPS, tau=ITE_TAU):
    n = H.shape[0]
    psi = np.ones(n, dtype=complex) / np.sqrt(n)

    for step in range(steps):
        psi = psi - tau * (H @ psi)

        norm = np.linalg.norm(psi)
        if norm > 0:
            psi /= norm

    return np.abs(psi) ** 2


def quantum_ite_circuit(H_diag, step):
    qc = QuantumCircuit(NUM_QUBITS)

    for i in range(NUM_QUBITS):
        qc.h(i)

    angles = H_diag[:NUM_QUBITS] * ITE_TAU * (step + 1)
    for i in range(NUM_QUBITS):
        qc.ry(angles[i], i)

    for i in range(NUM_QUBITS - 1):
        qc.cx(i, i + 1)

    for i in range(NUM_QUBITS):
        qc.rz(angles[i] * 0.5, i)

    for i in range(NUM_QUBITS - 1):
        qc.cx(i, i + 1)
    qc.cx(NUM_QUBITS - 1, 0)

    for i in range(NUM_QUBITS):
        qc.ry(-angles[i] * 0.3, i)

    return qc


def quantum_thermal_features(H, X_feats):
    H_diag = np.diag(H)[:NUM_QUBITS]

    all_feats = []
    for step in range(ITE_STEPS):
        qc_step = quantum_ite_circuit(H_diag, step)
        step_feats = []
        for x in X_feats:
            qc = QuantumCircuit(NUM_QUBITS)
            for i in range(NUM_QUBITS):
                qc.ry(x[i], i)
            qc.compose(qc_step, inplace=True)
            sv = Statevector.from_instruction(qc)
            step_feats.append(sv.probabilities())
        all_feats.append(np.array(step_feats))

    return np.hstack(all_feats)


def value_to_features(v):
    theta = v * np.pi / 31.0
    return np.array([theta * (k + 1) for k in range(NUM_QUBITS)])


def ridge_fit_predict(X, y, lam=LAMBDA_REG):
    alpha = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
    return X @ alpha


def greedy_combo(dists):
    combo = []
    used = set()
    for pos in range(7):
        ranked = sorted(enumerate(dists[pos]),
                        key=lambda x: x[1], reverse=True)
        for mv, score in ranked:
            actual = int(mv) + MIN_VAL[pos]
            if actual > MAX_VAL[pos]:
                continue
            if actual in used:
                continue
            if combo and actual <= combo[-1]:
                continue
            combo.append(actual)
            used.add(actual)
            break
    return combo


def main():
    draws = load_draws()
    print(f"Ucitano izvucenih kombinacija: {len(draws)}")

    df_all_head = pd.read_csv(CSV_ALL, nrows=3)
    print(f"Graf svih kombinacija: {CSV_ALL}")
    print(f"  Primer: {df_all_head.values[0].tolist()} ... "
          f"{df_all_head.values[-1].tolist()}")

    n_states = 1 << NUM_QUBITS
    X_feats = np.array([value_to_features(v) for v in range(n_states)])

    print(f"\n--- QITR ({NUM_QUBITS}q, {ITE_STEPS} ITE koraka, "
          f"tau={ITE_TAU}) ---")

    dists_ite = []
    dists_qf = []
    for pos in range(7):
        print(f"  Poz {pos+1}...", end=" ", flush=True)

        H = build_hamiltonian(draws, pos)
        p_ite = imaginary_time_evolution(H)

        F = quantum_thermal_features(H, X_feats)
        y = build_empirical(draws, pos)
        p_qf = ridge_fit_predict(F, y)
        p_qf = p_qf - p_qf.min()
        if p_qf.sum() > 0:
            p_qf /= p_qf.sum()

        combined = 0.5 * p_ite + 0.5 * p_qf
        combined = combined - combined.min()
        if combined.sum() > 0:
            combined /= combined.sum()
        dists_ite.append(combined)

        top_idx = np.argsort(combined)[::-1][:3]
        info = " | ".join(
            f"{i + MIN_VAL[pos]}:{combined[i]:.3f}" for i in top_idx)
        print(f"top: {info}")

    combo = greedy_combo(dists_ite)

    print(f"\n{'='*50}")
    print(f"Predikcija (QITR, deterministicki, seed={SEED}):")
    print(combo)
    print(f"{'='*50}")


if __name__ == "__main__":
    main()


"""
Ucitano izvucenih kombinacija: 4582
Graf svih kombinacija: /data/kombinacijeH_39C7.csv
  Primer: [1, 2, 3, 4, 5, 6, 7] ... [1, 2, 3, 4, 5, 6, 9]

--- QITR (5q, 8 ITE koraka, tau=0.2) ---
  Poz 1... top: 32:0.084 | 29:0.084 | 26:0.084
  Poz 2... top: 33:0.500 | 8:0.039 | 9:0.039
  Poz 3... top: 13:0.079 | 14:0.077 | 12:0.075
  Poz 4... top: 23:0.079 | 21:0.076 | 18:0.076
  Poz 5... top: 26:0.078 | 29:0.077 | 27:0.074
  Poz 6... top: 7:0.173 | 10:0.170 | 8:0.169
  Poz 7... top: 8:0.132 | 11:0.123 | 13:0.117

==================================================
Predikcija (QITR, deterministicki, seed=39):
[32, 33, x, y, z, 37, 38]
==================================================
"""



"""
QITR - Quantum Imaginary Time Regression

Hamiltonijan iz podataka: dijagonala = -log(frekvencija), van-dijagonala = tranzicione veze izmedju uzastopnih izvlacenja
Imaginary Time Evolution (ITE): psi -= tau * H @ psi sa normalizacijom - konvergira ka osnovnom stanju H (najfrekventnije kombinacije)
Kvantni termalni feature-i: 8 ITE koraka enkodiranih u kvantna kola sa razlicitom dubinom - hvata evoluciju od uniformnog ka termalnom stanju
Feature matrica: 8 koraka x 32 verovatnoce = 256 dimenzija
Kombinovano: 50% ITE distribucija + 50% kvantna regresija nad termalnim feature-ima
Inspirisano kvantnom termodinamikom i variational ITE
Deterministicki
"""

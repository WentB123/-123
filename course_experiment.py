"""Bounded, complete course reproduction using saved official QAOA circuits.

Run with .runtime/python310/python.exe course_experiment.py. Original research
files and incomplete large-budget jobs are never overwritten or resumed.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.metadata
import json
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from qiskit import qpy
from qiskit.primitives import BackendSamplerV2
from qiskit_aer import AerSimulator

from project_metrics import evaluate_bits, first_front, hypervolume
from run_official_mps import hypervolume_3d, verify_postprocessor

OUT = ROOT / 'output/course_reproduction_20261103'
TEMPLATES = ROOT / 'output/official_reproduction/mps_42q_3obj_fixed_circuits'
sys.path.insert(0, str(ROOT / 'qamoo-main'))
from qamoo.configs.configs import ProblemSpecification


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',default='course_reproduction_20261103',help='New output folder name; existing runs are protected.')
    args=parser.parse_args()
    if Path(args.tag).name!=args.tag or args.tag in ['.','..']:
        parser.error('tag must be a single directory name')
    OUT=ROOT/'output'/args.tag
    OUT.mkdir(exist_ok=False, parents=True)
    inputs = OUT / 'inputs'
    inputs.mkdir()
    results = OUT / 'runs'
    results.mkdir()
    problem = ProblemSpecification(data_folder=(ROOT / 'qamoo-main/data').as_posix()+'/',
                                   num_qubits=42, num_objectives=3, num_swap_layers=0, problem_id=0)
    graphs, ref = problem.objective_graphs, np.asarray(problem.lower_bounds)
    depths, seeds, shots, weights, chi = [1, 3, 6], list(range(20261103, 20261108)), 1000, 10, 20
    manifest = []
    for p in depths:
        folder = inputs / f'p{p}'
        folder.mkdir()
        for name in ['transpiled_parametrized_circuit.qpy', 'parameter_dicts.json', 'objective_weights.json']:
            src = TEMPLATES / f'p{p}_seed42' / name
            dst = folder / name
            dst.write_bytes(src.read_bytes())
            manifest.append({'source':src.relative_to(ROOT).as_posix(), 'copy':dst.relative_to(OUT).as_posix(), 'sha256':sha(src)})
    for src in sorted((ROOT/'qamoo-main/data/problems/42q/problem_set_42q_0s_3o_0').glob('*.json')):
        dst = inputs / src.name
        dst.write_bytes(src.read_bytes())
        manifest.append({'source':src.relative_to(ROOT).as_posix(), 'copy':dst.relative_to(OUT).as_posix(), 'sha256':sha(src)})
    for src in [ROOT/'qamoo-main/qamoo/algorithms/qaoa.py', ROOT/'qamoo-main/data/problems/27q/problem_set_27q_0s_3o_0/JuliQAOA_angles.json']:
        manifest.append({'source':src.relative_to(ROOT).as_posix(), 'sha256':sha(src)})
    sources = json.loads((ROOT/'data/sources.json').read_text(encoding='utf-8'))
    for s in sources:
        assert sha(ROOT/'data'/s['file']) == s['sha256'], s['file']
    dump(OUT/'input_manifest.json', manifest)
    dump(OUT/'config.json', {'depths':depths,'seeds':seeds,'shots_per_circuit':shots,'circuits':weights,'chi':chi,
                            'reference_point':ref.tolist(),'n_qubits':42,'objectives':3,
                            'fixed_weight_subset':'first 10 official vectors, shared across depths and repeats',
                            'scope':'new ideal-circuit approximate MPS sampling; parameters transferred, not retrained',
                            'versions':{x:importlib.metadata.version(x) for x in ['numpy','qiskit','qiskit-aer','moocore']},
                            'python':sys.version,'source_archives_verified':len(sources),
                            'paper_budget_per_run':25000000,'course_budget_per_run':weights*shots})
    verify_postprocessor()
    rows, curves, verification, projection = [], [], [], {}
    for seed in seeds:
        for p in [0]+depths:
            label = 'Random' if p == 0 else f'QAOA p={p}'
            t = time.perf_counter()
            if p == 0:
                # Same uniform bit distribution and dtype as the author's random baseline.
                bits = np.random.RandomState(seed).randint(0, 2, size=(weights*shots, 42), dtype='b')
            else:
                folder = inputs/f'p{p}'
                with (folder/'transpiled_parametrized_circuit.qpy').open('rb') as f:
                    circuit = qpy.load(f)[0]
                bindings = json.loads((folder/'parameter_dicts.json').read_text())
                values = [[b[k.name] for k in circuit.parameters] for b in bindings]
                backend = AerSimulator(method='matrix_product_state', matrix_product_state_max_bond_dimension=chi, max_parallel_threads=1)
                sampler = BackendSamplerV2(backend=backend, options={'seed_simulator':seed})
                measured = sampler.run([(circuit, values, shots)]).result()[0].data.meas.get_bitstrings()
                bits = np.array([[int(x) for x in s[::-1]] for s in measured], dtype=np.int8)
            sampling = time.perf_counter()-t
            assert bits.shape == (10000,42) and np.isin(bits, [0,1]).all()
            t = time.perf_counter()
            objectives = evaluate_bits(bits, graphs)
            front = first_front(objectives)
            hv = hypervolume(objectives[front], ref)
            independent = hypervolume_3d(objectives[front], ref)
            assert np.isclose(hv, independent, atol=1e-7, rtol=0)
            for idx in [0, 17, 990, 4000, 9999]:
                for j,g in enumerate(graphs):
                    expected = sum(d['weight'] for u,v,d in g.edges(data=True) if bits[idx,u] != bits[idx,v])
                    assert np.isclose(objectives[idx,j], expected, atol=1e-12)
            progress = []
            for n in range(shots, weights*shots+1, shots):
                progress.append(hypervolume(objectives[:n], ref))
                curves.append({'method':label,'p':p,'seed':seed,'shots':n,'hypervolume':progress[-1]})
            assert np.all(np.diff(progress)>=-1e-8)
            np.savez_compressed(results/f'p{p}_seed{seed}.npz', bits=bits, objectives=objectives, front=front)
            # First repeat retained for transparent 2D projection in the teaching animation.
            if seed == seeds[0]:
                projection[str(p)] = {'points':objectives[::20,:2].round(6).tolist(),
                                      'front':objectives[front,:2].round(6).tolist()}
            row = {'method':label,'p':p,'seed':seed,'shots':len(bits),'hypervolume':hv,
                   'non_dominated_vectors':len(front),'sampling_wall_s':sampling,
                   'processing_wall_s':time.perf_counter()-t}
            rows.append(row)
            verification.append({'method':label,'seed':seed,'hv_independent_abs_error':abs(hv-independent),'shape_ok':True,'objective_scalar_checks':15,'hv_monotone':True})
            print(json.dumps(row), flush=True)
    for name,data in [('results.csv',rows),('curves.csv',curves)]:
        with (OUT/name).open('w', newline='', encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    summary=[]
    for p in [0]+depths:
        subset=[r for r in rows if r['p']==p]
        summary.append({'method':subset[0]['method'],'p':p,'repeats':len(subset),
                        **{f'{key}_{stat}':float(fn([r[key] for r in subset])) for key in ['hypervolume','non_dominated_vectors','sampling_wall_s'] for stat,fn in [('mean',np.mean),('std',lambda x:np.std(x,ddof=1)),('min',np.min),('max',np.max)]}})
    dump(OUT/'summary.json',summary)
    dump(OUT/'projection.json',projection)
    dump(OUT/'verification.json',{'complete':True,'runs':len(rows),'samples':sum(r['shots'] for r in rows),
                                  'checks':verification,'note':'five measurement repeats on one fixed problem and weight subset; SD is descriptive, not a confidence interval'})
    print('COURSE EXPERIMENT COMPLETE',flush=True)


if __name__ == '__main__':
    main()

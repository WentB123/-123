"""Resume-safe expanded execution of official QAMOO inputs and algorithms.

Commands: prepare, benchmark, worker, run, status. All outputs stay in the project.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'qamoo-main')]
os.environ['MPLCONFIGDIR'] = str(ROOT / '.jupyter/matplotlib310')
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
BASE = ROOT / 'output/project_reproduction'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def replace_with_retry(source, target):
    # Windows readers/virus scanners can briefly deny FILE_SHARE_DELETE.
    for attempt in range(80):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 79:
                raise
            time.sleep(0.025)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    replace_with_retry(tmp, path)


def archive_prefix(objectives, p):
    problem = f'qamoo-main/data/problems/42q/problem_set_42q_0s_{objectives}o_0/results/'
    suffix = (f'qaoa_aer_simulator_matrix_product_state_v2_p_{p}_5000_samples_5000_shots_50_0/'
              if objectives == 3 else
              'qaoa_aer_simulator_matrix_product_state_v2_p_6_20000_samples_5000_shots_a_20_0/')
    return problem + suffix


def prepare():
    BASE.mkdir(parents=True, exist_ok=True)
    with ZipFile(ROOT / 'qamoo-main.zip') as archive:
        for objectives, p in [(3, p) for p in range(1, 7)] + [(4, 6)]:
            prefix = archive_prefix(objectives, p)
            dest = BASE / 'inputs' / f'{objectives}o_p{p}'
            if (dest / 'manifest.json').exists():
                continue
            dest.mkdir(parents=True, exist_ok=True)
            manifest = []
            for filename in ['transpiled_parametrized_circuit.qpy', 'parameter_dicts.json']:
                source = prefix + filename
                if filename.endswith('.json') and source not in archive.namelist():
                    zipped = archive.read(source + '.zip')
                    with ZipFile(io.BytesIO(zipped)) as inner:
                        names = [n for n in inner.namelist() if n.endswith('parameter_dicts.json')]
                        if len(names) != 1:
                            raise ValueError(names)
                        data = inner.read(names[0])
                    source += '.zip (inner parameter_dicts.json)'
                else:
                    data = archive.read(source)
                (dest / filename).write_bytes(data)
                manifest.append({'source': source, 'target': filename, 'bytes': len(data),
                                 'sha256': hashlib.sha256(data).hexdigest()})
            save(dest / 'manifest.json', manifest)
            print(f'Prepared original circuit and parameter bindings: {objectives} objectives, p={p}', flush=True)


def spec(objectives):
    from qamoo.configs.configs import ProblemSpecification
    return ProblemSpecification(data_folder=(ROOT / 'qamoo-main/data').as_posix() + '/',
                                num_qubits=42, num_objectives=objectives, num_swap_layers=0, problem_id=0)


def atomic_npz(path, **arrays):
    import numpy as np
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    replace_with_retry(tmp, path)


def latest_checkpoint(folder):
    import numpy as np
    candidates = list(folder.glob('checkpoint_*.npz'))
    # Earlier runner versions stored front state in each batch file.
    batches = sorted(folder.glob('batch_*.npz'))
    if batches:
        with np.load(batches[-1], allow_pickle=False) as ck:
            if 'front_values' in ck:
                candidates.append(batches[-1])
    if not candidates:
        return None
    def end(path):
        with np.load(path, allow_pickle=False) as ck:
            return int(ck['end_circuit'])
    return max(candidates, key=end)


def batch_and_checkpoint(folder, index, raw_fields, state_fields):
    import numpy as np
    import shutil
    if shutil.disk_usage(folder).free < 2 * 1024**3:
        raise OSError('Less than 2 GiB free disk; previous completed checkpoint remains resumable.')
    # Store every new measurement once; rotate just two complete Pareto checkpoints.
    # If a crash occurs after the batch write but before its checkpoint, replay that batch.
    atomic_npz(folder / f'batch_{index:06d}.npz', **raw_fields)
    atomic_npz(folder / f'checkpoint_{index % 2}.npz', batch_index=np.array(index), **state_fields)


def qaoa_worker(job):
    import numpy as np
    from qiskit import qpy
    from qiskit.primitives import BackendSamplerV2
    from qiskit_aer import AerSimulator
    from project_metrics import first_front, hypervolume, evaluate_bits
    folder = BASE / 'runs' / job['id']
    folder.mkdir(parents=True, exist_ok=True)
    config_path = folder / 'config.json'
    if config_path.exists() and read(config_path) != job:
        raise ValueError('Job settings differ from the saved checkpoint configuration.')
    save(config_path, job)
    if (folder / 'complete.json').exists():
        return read(folder / 'complete.json')
    problem = spec(job['objectives'])
    graphs, reference = problem.objective_graphs, problem.lower_bounds
    source = BASE / 'inputs' / f"{job['objectives']}o_p{job['p']}"
    for item in read(source / 'manifest.json'):
        assert hashlib.sha256((source / item['target']).read_bytes()).hexdigest() == item['sha256']
    with (source / 'transpiled_parametrized_circuit.qpy').open('rb') as stream:
        circuit = qpy.load(stream)[0]
    bindings = read(source / 'parameter_dicts.json')
    assert len(bindings) >= job['circuits']
    param_names = [p.name for p in circuit.parameters]
    backend = AerSimulator(method='matrix_product_state',
                           matrix_product_state_max_bond_dimension=job['chi'], max_parallel_threads=1)
    front_values = np.empty((0, job['objectives']))
    front_bits = np.empty((0, 42), dtype=np.int8)
    front_positions = np.empty(0, dtype=np.int64)
    completed, sampling_s, processing_s, volume, batch_index = 0, 0., 0., 0., 0
    last = latest_checkpoint(folder)
    if last:
        with np.load(last, allow_pickle=False) as ck:
            completed = int(ck['end_circuit'])
            front_values, front_bits, front_positions = ck['front_values'], ck['front_bits'], ck['front_positions']
            sampling_s, processing_s, volume = float(ck['sampling_s']), float(ck['processing_s']), float(ck['hypervolume'])
            batch_index = int(ck['batch_index']) + 1 if 'batch_index' in ck else int(last.stem.split('_')[1]) + 1
    state = {**job, 'status': 'running', 'completed_circuits': completed,
             'completed_shots': completed * job['shots'], 'target_shots': job['circuits'] * job['shots'],
             'non_dominated_vectors': len(front_values), 'hypervolume': volume,
             'sampling_wall_s': sampling_s, 'processing_wall_s': processing_s, 'updated_unix': time.time()}
    save(folder / 'status.json', state)
    for start in range(completed, job['circuits'], job['batch_circuits']):
        end = min(start + job['batch_circuits'], job['circuits'])
        values = [[row[name] for name in param_names] for row in bindings[start:end]]
        batch_seed = int(np.random.SeedSequence([job['seed'], start]).generate_state(1)[0])
        sampler = BackendSamplerV2(backend=backend, options={'seed_simulator': batch_seed})
        t = time.perf_counter()
        result = sampler.run([(circuit, values, job['shots'])]).result()
        sampling_s += time.perf_counter() - t
        t = time.perf_counter()
        packed = result[0].data.meas.array.reshape(-1, 6)
        bits = np.unpackbits(packed, axis=1)[:, -42:][:, ::-1].astype(np.int8)
        assert bits.shape == ((end - start) * job['shots'], 42)
        # Independent check of packed bit order on the first measurement in each batch.
        assert ''.join(map(str, bits[0][::-1])) == result[0].data.meas.get_bitstrings((0,))[0]
        objective_values = evaluate_bits(bits, graphs)
        positions = np.arange(start * job['shots'], end * job['shots'], dtype=np.int64)
        merged_values = np.vstack((front_values, objective_values))
        merged_bits = np.vstack((front_bits, bits))
        merged_positions = np.r_[front_positions, positions]
        kept = first_front(merged_values)
        front_values, front_bits, front_positions = merged_values[kept], merged_bits[kept], merged_positions[kept]
        new_volume = hypervolume(front_values, reference)
        assert new_volume >= volume - 1e-7
        volume = new_volume
        processing_s += time.perf_counter() - t
        shared = dict(start_circuit=np.array(start), end_circuit=np.array(end),
                      hypervolume=np.array(volume), batch_seed=np.array(batch_seed))
        batch_and_checkpoint(folder, batch_index,
                             dict(raw_packed_bits=packed, **shared),
                             dict(front_values=front_values, front_bits=front_bits, front_positions=front_positions,
                                  sampling_s=np.array(sampling_s), processing_s=np.array(processing_s), **shared))
        completed = end
        state = {**job, 'status': 'running', 'completed_circuits': end,
                 'completed_shots': end * job['shots'], 'target_shots': job['circuits'] * job['shots'],
                 'non_dominated_vectors': len(front_values), 'hypervolume': volume,
                 'sampling_wall_s': sampling_s, 'processing_wall_s': processing_s,
                 'updated_unix': time.time()}
        save(folder / 'status.json', state)
        print(json.dumps(state), flush=True)
        batch_index += 1
        if (BASE / 'STOP_AFTER_BATCH').exists():
            state['status'] = 'stopped_at_checkpoint'
            save(folder / 'status.json', state)
            return state
    state['status'] = 'complete'
    save(folder / 'complete.json', state)
    save(folder / 'status.json', state)
    return state


def classical_worker(job):
    import numpy as np
    from project_metrics import install_explicit_pygmo_adapter, first_front, hypervolume, evaluate_bits
    adapter = install_explicit_pygmo_adapter()
    from qamoo.algorithms import classical_algorithms as official
    from qamoo.configs.configs import RandomConfig, WeightedSumConfig, RandomEpsConstraintConfig, GoemansWilliamsonConfig
    folder = BASE / 'runs' / job['id']
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        return read(folder / 'complete.json')
    save(folder / 'config.json', job)
    problem = spec(job['objectives'])
    methods = {'random': (RandomConfig, official.random),
               'weighted_sum': (WeightedSumConfig, official.weighted_sum),
               'epsilon_constraint': (RandomEpsConstraintConfig, official.random_eps_constraint),
               'goemans_williamson': (GoemansWilliamsonConfig, official.goemans_williamson)}
    cls, function = methods[job['method']]
    all_weights = None
    if job['method'] != 'random':
        all_weights = read(ROOT / f"qamoo-main/data/objective_weights/objective_weights_{job['objectives']}o_0.json")
        assert len(all_weights) >= job['circuits']
    front_values, front_bits = np.empty((0, job['objectives'])), np.empty((0, 42), dtype=np.int8)
    front_positions = np.empty(0, dtype=np.int64)
    completed, wall, processing_s, volume, batch_index = 0, 0., 0., 0., 0
    last = latest_checkpoint(folder)
    if last:
        with np.load(last, allow_pickle=False) as ck:
            completed, wall = int(ck['end_circuit']), float(ck['algorithm_s'])
            processing_s, volume = float(ck['processing_s']), float(ck['hypervolume'])
            front_values, front_bits, front_positions = ck['front_values'], ck['front_bits'], ck['front_positions']
            batch_index = int(ck['batch_index']) + 1 if 'batch_index' in ck else int(last.stem.split('_')[1]) + 1
    state = {**job, 'status': 'running', 'completed_circuits': completed,
             'completed_shots': completed * job['shots'], 'hypervolume': volume,
             'non_dominated_vectors': len(front_values), 'algorithm_wall_s': wall,
             'processing_wall_s': processing_s, 'hv_adapter': adapter, 'updated_unix': time.time()}
    save(folder / 'status.json', state)
    graphs, reference = problem.objective_graphs, problem.lower_bounds
    for start in range(completed, job['circuits'], job['batch_circuits']):
        end = min(job['circuits'], start + job['batch_circuits'])
        # Author functions are unmodified. Slice their weight sequence and redirect generated output.
        properties = {'results_folder': property(lambda self: (folder / 'author_output').as_posix() + '/')}
        if all_weights is not None:
            properties['objective_weights'] = property(lambda self: all_weights[start:end])
        config_cls = type('LocalConfig', (cls,), properties)
        config = config_cls(problem=problem, run_id=job['seed'], num_samples=end-start)
        if hasattr(config, 'shots'):
            config.shots = job['shots']
        batch_seed = int(np.random.SeedSequence([job['seed'], start]).generate_state(1)[0])
        if hasattr(config, 'eps_seed'):
            config.eps_seed = batch_seed
        np.random.seed(batch_seed)
        started = time.perf_counter()
        function(config, overwrite_results=True)
        wall += time.perf_counter() - started
        t = time.perf_counter()
        samples = np.load(folder / 'author_output/samples.npy')
        values = evaluate_bits(samples, graphs)
        merged_values = np.vstack((front_values, values))
        merged_bits = np.vstack((front_bits, samples))
        merged_positions = np.r_[front_positions, np.arange(start*job['shots'], end*job['shots'])]
        kept = first_front(merged_values)
        front_values, front_bits, front_positions = merged_values[kept], merged_bits[kept], merged_positions[kept]
        volume = hypervolume(front_values, reference)
        # Qiskit-compatible six-byte big-endian encoding, with six leading padding bits.
        padded = np.pad(samples[:, ::-1], ((0, 0), (6, 0)))
        packed = np.packbits(padded, axis=1)
        processing_s += time.perf_counter() - t
        extra = {}
        if (folder / 'author_output/hv_estimate.json').exists():
            extra['batch_hv_estimate_json'] = np.array(json.dumps(read(folder / 'author_output/hv_estimate.json')))
        shared = dict(start_circuit=np.array(start), end_circuit=np.array(end),
                      hypervolume=np.array(volume), batch_seed=np.array(batch_seed))
        batch_and_checkpoint(folder, batch_index, dict(raw_packed_bits=packed, **shared, **extra),
                             dict(front_values=front_values, front_bits=front_bits, front_positions=front_positions,
                                  algorithm_s=np.array(wall), processing_s=np.array(processing_s), **shared))
        state.update(completed_circuits=end, completed_shots=end*job['shots'], hypervolume=volume,
                     non_dominated_vectors=len(front_values), algorithm_wall_s=wall,
                     processing_wall_s=processing_s, updated_unix=time.time())
        save(folder / 'status.json', state)
        print(json.dumps(state), flush=True)
        batch_index += 1
        if (BASE / 'STOP_AFTER_BATCH').exists():
            state['status'] = 'stopped_at_checkpoint'
            save(folder / 'status.json', state)
            return state
    state['status'] = 'complete'
    save(folder / 'complete.json', state)
    save(folder / 'status.json', state)
    print(json.dumps(state), flush=True)
    return state


def status():
    plan = read(BASE / 'plan.json') if (BASE / 'plan.json').exists() else []
    states = []
    for job in plan:
        folder = BASE / 'runs' / job['id']
        current = folder / 'status.json'
        states.append(read(current) if current.exists() else {**job, 'status': 'queued'})
    save(BASE / 'progress.json', {'jobs': states, 'updated_unix': time.time()})
    lines = ['# 项目复现进度', '', '此文件由运行器自动更新。只有 complete 才表示该任务已完成。', '',
             '| 任务 | 状态 | 已处理样本 | 目标样本 | 当前 HV |', '|---|---|---:|---:|---:|']
    for s in states:
        target = s['circuits'] * s.get('shots', 1)
        lines.append(f"| {s['id']} | {s['status']} | {s.get('completed_shots', 0):,} | {target:,} | {s.get('hypervolume', 0):.6f} |")
    (BASE / 'PROGRESS.md').write_text('\n'.join(lines), encoding='utf-8')
    columns = ['id', 'method', 'objectives', 'p', 'chi', 'seed', 'status', 'circuits', 'shots',
               'completed_shots', 'hypervolume', 'non_dominated_vectors',
               'sampling_wall_s', 'algorithm_wall_s', 'processing_wall_s']
    temp_csv = BASE / f'results.csv.{os.getpid()}.tmp'
    with temp_csv.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(states)
    try:
        replace_with_retry(temp_csv, BASE / 'results.csv')
    except PermissionError:
        # Excel may hold a report open for a long time; preserve execution and retry next refresh.
        temp_csv.unlink(missing_ok=True)
    return states


def _run_unlocked(plan_path, workers):
    plan = read(plan_path)
    previous = BASE / 'plan.json'
    if previous.exists() and read(previous) != plan:
        raise ValueError('Different saved plan. Preserve the existing experiment before changing its queue.')
    save(previous, plan)
    save(BASE / 'suite_status.json', {'status': 'running', 'total': len(plan), 'updated_unix': time.time()})
    status()
    logs = BASE / 'logs'
    logs.mkdir(exist_ok=True)

    def dispatch(job):
        if (BASE / 'STOP_AFTER_BATCH').exists():
            return 0
        folder = BASE / 'runs' / job['id']
        if (folder / 'complete.json').exists():
            return 0
        job_file = logs / (job['id'] + '.job.json')
        save(job_file, job)
        for attempt in range(3):
            with (logs / (job['id'] + '.log')).open('a', encoding='utf-8') as log:
                code = subprocess.call([sys.executable, '-u', str(Path(__file__).resolve()), 'worker',
                                        '--job', str(job_file)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if not code or (BASE / 'STOP_AFTER_BATCH').exists():
                break
            print(f"Retrying {job['id']} after exit {code} (attempt {attempt+1}/3)", flush=True)
            time.sleep(2)
        if code:
            save(folder / 'status.json', {**job, 'status': 'failed', 'exit_code': code,
                                         'log': str(logs / (job['id'] + '.log'))})
        return code

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(dispatch, job): job for job in plan}
        while futures:
            done, _ = concurrent.futures.wait(futures, timeout=10,
                                             return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                job = futures.pop(future)
                print(f"Finished {job['id']}: exit {future.result()}", flush=True)
            states = status()
    success = all(s['status'] == 'complete' for s in states)
    suite_state = 'complete' if success else ('stopped_at_checkpoint' if (BASE / 'STOP_AFTER_BATCH').exists() else 'has_failures')
    save(BASE / 'suite_status.json', {'status': suite_state,
                                     'completed': sum(s['status'] == 'complete' for s in states),
                                     'total': len(plan), 'updated_unix': time.time()})


def run(plan_path, workers):
    import msvcrt
    BASE.mkdir(parents=True, exist_ok=True)
    with (BASE / 'controller.lock').open('a+b') as lock:
        lock.write(b'0')
        lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            stop = BASE / 'STOP_AFTER_BATCH'
            if stop.exists():
                stop.unlink()
            save(BASE / 'controller.json', {'pid': os.getpid(), 'workers': workers,
                                           'plan': str(Path(plan_path).resolve()), 'started_unix': time.time()})
            _run_unlocked(plan_path, workers)
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'benchmark', 'worker', 'run', 'status', 'stop'])
    parser.add_argument('--job')
    parser.add_argument('--plan', default=str(ROOT / 'expanded_plan.json'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'worker':
        job = read(args.job)
        if job['method'] == 'qaoa':
            qaoa_worker(job)
        else:
            classical_worker(job)
    elif args.command == 'benchmark':
        job = {'id': 'benchmark_3o_p6_5000shots', 'method': 'qaoa', 'objectives': 3,
               'p': 6, 'chi': 50, 'circuits': 2, 'shots': 5000, 'seed': 2026, 'batch_circuits': 2}
        qaoa_worker(job)
    elif args.command == 'run':
        run(args.plan, args.workers)
    elif args.command == 'stop':
        (BASE / 'STOP_AFTER_BATCH').touch()
        print('Stop requested: workers will finish their current batch and preserve checkpoints.')
    else:
        print(json.dumps(status(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

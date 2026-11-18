"""Export a bounded selection of actual expanded-run measurements as readable JSONL."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from project_reproduction import BASE, read, spec, latest_checkpoint
from project_metrics import evaluate_bits
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True, help='Job id from expanded_plan.json')
parser.add_argument('--output', required=True)
parser.add_argument('--limit', type=int, default=1000, help='Maximum records; 0 exports all available records.')
parser.add_argument('--front-only', action='store_true')
args = parser.parse_args()
if Path(args.run).name != args.run or args.limit < 0:
    parser.error('Invalid run id or limit.')
folder = BASE / 'runs' / args.run
job = read(folder / 'config.json')
graphs = spec(job['objectives']).objective_graphs
out = Path(args.output).resolve()
out.parent.mkdir(parents=True, exist_ok=True)
count = 0
with out.open('w', encoding='utf-8') as stream:
    files = [latest_checkpoint(folder)] if args.front_only else sorted(folder.glob('batch_*.npz'))
    for path in files:
        if path is None:
            continue
        with np.load(path, allow_pickle=False) as data:
            if args.front_only:
                bits, values, positions = data['front_bits'], data['front_values'], data['front_positions']
            else:
                bits = np.unpackbits(data['raw_packed_bits'], axis=1)[:, -42:][:, ::-1]
                if args.limit:
                    bits = bits[:args.limit-count]
                values = evaluate_bits(bits, graphs)
                start = int(data['start_circuit']) * job['shots']
                positions = np.arange(start, start+len(bits))
            for index, sample, objectives in zip(positions, bits, values):
                stream.write(json.dumps({'run': args.run, 'method': job['method'],
                                         'data_kind': 'new_local_measurements',
                                         'sample_index': int(index), 'seed': job['seed'],
                                         'solution_or_circuit_index': int(index) // job['shots'],
                                         'bits_node_0_to_41': sample.tolist(),
                                         'objectives': objectives.tolist()}, allow_nan=False) + '\n')
                count += 1
                if args.limit and count >= args.limit:
                    break
        if args.limit and count >= args.limit:
            break
print(f'Exported {count} records to {out}')

"""Extract official source and small inputs, leaving historical results archived."""
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent


def main():
    archive_path = ROOT / 'qamoo-main.zip'
    inventory = []
    with ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            name = member.filename
            is_data = '/data/' in name
            keep = not is_data or '/data/initial_layouts/' in name or '/data/objective_weights/' in name
            keep |= '/data/problems/' in name and '/results/' not in name and member.file_size < 100000
            if not keep:
                continue
            dest = (ROOT / name).resolve()
            if not dest.is_relative_to(ROOT) or not name.startswith('qamoo-main/'):
                raise ValueError(name)
            content = archive.read(member)  # ZIP CRC is checked here.
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and dest.read_bytes() != content:
                raise ValueError(f'Refusing to overwrite modified upstream file: {dest}')
            dest.write_bytes(content)
            inventory.append({'file': name, 'bytes': len(content),
                              'sha256': hashlib.sha256(content).hexdigest()})
    out = ROOT / 'output' / 'official_reproduction'
    out.mkdir(parents=True, exist_ok=True)
    # Hash streaming avoids loading the 2.48 GB archive into RAM.
    digest = hashlib.sha256()
    with archive_path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    manifest = {'archive': archive_path.name, 'archive_sha256': digest.hexdigest(),
                'upstream': 'https://github.com/stefan-woerner/qamoo',
                'git_commit': None, 'note': 'User-supplied main ZIP; commit not established.',
                'files': inventory}
    (out / 'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    graphs = []
    edge_rows = []
    for path in sorted((ROOT / 'qamoo-main/data/problems').glob('*/*/problem_graph_*.json')):
        graph = json.loads(path.read_text())
        objective = int(path.stem.rsplit('_', 1)[1])
        graphs.append({'source_file': path.relative_to(ROOT).as_posix(),
                       'problem': path.parent.name, 'objective': objective, 'graph': graph})
        for link in graph['links']:
            edge_rows.append({'problem': path.parent.name, 'objective': objective,
                              'node_u': link['source'], 'node_v': link['target'],
                              'weight': link['weight']})
    (out / 'problem_graphs.json').write_text(json.dumps(graphs, indent=2), encoding='utf-8')
    with (out / 'problem_edges.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(edge_rows[0]))
        writer.writeheader()
        writer.writerows(edge_rows)
    print(f'Verified/extracted {len(inventory)} files; exported {len(graphs)} objective graphs, {len(edge_rows)} weighted edges.')


if __name__ == '__main__':
    main()

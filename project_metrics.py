"""Validated, bounded-memory metric helpers for the expanded reproduction."""
import sys
import types

import numpy as np
import moocore


def first_front(values):
    if not len(values):
        return np.empty(0, dtype=int)
    return np.flatnonzero(moocore.is_nondominated(values, maximise=True, keep_weakly=False))


def hypervolume(values, lower_reference):
    if not len(values):
        return 0.
    return float(moocore.hypervolume(values, ref=lower_reference, maximise=True))


def evaluate_bits(bits, graphs):
    return np.column_stack([
        sum(float(d['weight']) * (bits[:, u] != bits[:, v]) for u, v, d in graph.edges(data=True))
        for graph in graphs
    ])


def install_explicit_pygmo_adapter():
    """Satisfy ONLY the author's hypervolume API using moocore, without claiming pygmo installed."""
    try:
        import pygmo  # noqa: F401
        return 'pygmo'
    except ImportError:
        pass
    module = types.ModuleType('pygmo')
    module.__doc__ = 'Local reproduction adapter: hypervolume only, implemented with moocore.'

    class Hypervolume:
        def __init__(self, points):
            self.points = np.asarray(points)

        def compute(self, reference):
            return float(moocore.hypervolume(self.points, ref=reference))

    module.hypervolume = Hypervolume
    module.REPRODUCTION_ADAPTER = 'moocore 0.3.2, hypervolume API only'
    sys.modules['pygmo'] = module
    return module.REPRODUCTION_ADAPTER

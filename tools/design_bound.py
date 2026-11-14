#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_solution import SolutionCheck, RES, SHARED_BUSES, SWITCH_ETH_PORTS, REPO


def _cheapest_mix(types, load, h, n_min):
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds
    ids = sorted(types)
    res = [r for r in load if load[r] > 0]
    A = [[types[t][r] * (h if r in ('cpu', 'tops') else 1.0) for t in ids] for r in res] + [[1.0] * len(ids)]
    lb = [load[r] for r in res] + [float(n_min)]
    sol = milp(c=np.array([types[t]['cost'] for t in ids]), constraints=LinearConstraint(np.array(A), lb, np.inf),
               integrality=np.ones(len(ids)), bounds=Bounds(0, np.inf))
    if sol.x is None:
        return {}, float('inf')
    mix = {t: int(round(v)) for t, v in zip(ids, sol.x) if round(v) > 0}
    return mix, float(sum(types[t]['cost'] * n for t, n in mix.items()))


def _ceil(x):
    return int(math.ceil(x - 1e-6 * max(1.0, abs(x))))


def _must_hot(c):
    ftti = {}
    for ch in c.chains:
        for s in ch.get('swcs', []):
            ftti[s] = min(ftti.get(s, float('inf')), float(ch.get('ftti_ms', float('inf'))))
    return {s for s in c.crit if ftti.get(s, float('inf')) < c.recovery}


def bound_settings(c):
    d = {'time_limit_s': 300.0, 'threads': 8, 'failover': True}
    try:
        with open(os.path.join(c.config_dir, 'solver.json'), encoding='utf-8') as fh:
            d.update({k: v for k, v in (json.load(fh).get('bound') or {}).items() if not k.startswith('_')})
    except (OSError, ValueError):
        pass
    return d


def design_bound(c):
    h = max(c.derate(s) for s in c.cands)
    must_hot = _must_hot(c)
    nominal = list(c.scs) + sorted(must_hot)
    images = sorted(s for s in c.crit if s not in must_hot)

    def inst_r(r, inst):
        if r != 'rom':
            return inst
        levels = {c.scs[s]['asil'] for s in inst}
        return inst + [s for s in images if c.scs[s]['asil'] in levels]
    out = {'h': h, 'critical': len(c.crit), 'must_hot': len(must_hot)}
    parts, rows, part_cost = {}, [], 0.0
    for a in sorted({sc['asil'] for sc in c.scs.values()}):
        inst = [s for s in nominal if c.scs[s]['asil'] == a]
        n, by = 1, {}
        n_min = 2 if any(s in c.crit for s in inst) else 1
        if c.types:
            load = {r: sum(c.demand(s, r) for s in inst_r(r, inst)) for r in c.res}
            mix, cost_a = _cheapest_mix(c.types, load, h, n_min)
            for r in c.res:
                top = max(t[r] for t in c.types.values()) * (h if r in ('cpu', 'tops') else 1.0)
                if load[r] > 0:
                    by[r] = load[r] / top
                    n = max(n, _ceil(by[r]))
            n = max(n, n_min)
            part_cost += cost_a
            parts[a] = n
            rows.append({'asil': a, 'components': sum(1 for s in c.scs.values() if s['asil'] == a),
                         'critical': sum(1 for s in c.crit if c.scs[s]['asil'] == a),
                         'must_hot': sum(1 for s in must_hot if c.scs[s]['asil'] == a), 'load': by, 'bound': n,
                         'mix': mix, 'mix_cost': cost_a})
            continue
        for r in RES:
            cap = float(c.part.get(f'{r}_cap', 0) or 0) * (h if r == 'cpu' else 1.0)
            if cap > 0:
                by[r] = sum(c.demand(s, r) for s in inst_r(r, inst)) / cap
                n = max(n, _ceil(by[r]))
        n = max(n, n_min)
        parts[a] = n
        part_cost += n * float(c.part.get('cost', 0) or 0)
        rows.append({'asil': a, 'components': sum(1 for s in c.scs.values() if s['asil'] == a),
                     'critical': sum(1 for s in c.crit if c.scs[s]['asil'] == a),
                     'must_hot': sum(1 for s in must_hot if c.scs[s]['asil'] == a), 'load': by, 'bound': n})
    out['partitions'], out['partition_rows'] = sum(parts.values()), rows
    units = 1
    site_max = c.site_cfg.get('max_partitions')
    if site_max:
        units = max(units, _ceil(out['partitions'] / float(site_max)))
    for r in RES:
        cap = c.site_cfg.get(f'{r}_cap')
        if cap is not None:
            units = max(units, _ceil(sum(c.demand(s, r) for s in inst_r(r, nominal)) / (float(cap) * (h if r == 'cpu' else 1.0))))
    if c.crit or (c.flags.get('redundancy') and any(sc['redundant_with'] in c.scs for sc in c.scs.values())):
        units = max(units, 2)
    max_pa = int(c.part.get('max_per_asil_per_site', 3) or 0)
    if max_pa > 0:
        units = max(units, max(_ceil(n / float(max_pa)) for n in parts.values()))
    out['units'] = units
    hw = {}
    for f, price in c.hw_price.items():
        users = [s for s in c.scs if f in c.scs[s]['hw']]
        if not users:
            continue
        n = 2 if any(s in c.crit for s in users) else 1
        spec = c.hw_cap.get(f)
        if spec and spec.get('types'):
            gtypes = {str(t['id']): {'tops': float(t['tops']), 'cost': float(t.get('cost') or 0)} for t in spec['types']}
            load = sum(c.demand(s, 'tops') for s in nominal if f in c.scs[s]['hw'])
            mix, mix_cost = _cheapest_mix(gtypes, {'tops': load}, h, n)
            units = sum(mix.values())
            hw[f] = (units, mix_cost / units if units else 0.0)
            continue
        if spec:
            n = max(n, _ceil(sum(c.demand(s, 'tops') for s in nominal if f in c.scs[s]['hw']) / (float(spec['tops']) * h)))
        hw[f] = (n, price)
    out['hardware'] = hw
    by_if = defaultdict(int)
    for d in {d for sc in c.scs.values() for d in sc['devices']}:
        by_if[c.devices[d]['iface']] += 1
    ports = {}
    for t, k in sorted(by_if.items()):
        port = float(c.ct.get(t, {}).get('port', 0.0))
        if t in SHARED_BUSES:
            ports[t] = (1, port, port)
        elif t == 'ETH' and 'SWITCH' in c.hw_price:
            each = min(port, c.hw_price['SWITCH'] / SWITCH_ETH_PORTS)
            ports[t] = (k, each, k * each)
        else:
            ports[t] = (k, port, k * port)
    out['ports'] = ports
    out['cost'] = {'partitions': part_cost,
                   'units': units * float(c.site_cfg.get('fixed_cost', 0) or 0),
                   'hardware': sum(n * p for n, p in hw.values()),
                   'ports': sum(v[2] for v in ports.values()),
                   'network': 0.0}
    out['bound'] = sum(out['cost'].values())
    return out


def _k_cells_model(c, K, items, dem, res, kinds, kind_cap, kind_cost, unit_max, h, must_hot, slot_classes, tl, threads, presolve=None):
    import gurobipy as gp
    from gurobipy import GRB
    crit = [s for s in items if s in c.crit]
    cold = [s for s in crit if s not in must_hot]
    m = gp.Model('design_bound')
    m.Params.OutputFlag = 0
    m.Params.Threads = int(threads)
    m.Params.TimeLimit = float(tl)
    m.Params.Seed = 1
    m.Params.MIPFocus = 3
    m.Params.Symmetry = 2
    if presolve is not None:
        m.Params.Presolve = presolve
    cells = range(K)
    Q = m.addVars(cells, kinds, vtype=GRB.INTEGER, lb=0, ub=unit_max)
    x = m.addVars(items, cells, vtype=GRB.BINARY)
    y = m.addVars(crit, cells, vtype=GRB.BINARY)
    z = m.addVars(cold, cells, cells, lb=0.0, ub=1.0)
    for k in cells:
        m.addConstr(gp.quicksum(Q[k, t] for t in kinds) <= unit_max)
        for s in items:
            m.addConstr(x[s, k] <= gp.quicksum(Q[k, t] for t in kinds))
        for s in crit:
            m.addConstr(y[s, k] <= gp.quicksum(Q[k, t] for t in kinds))
    def _binds(r):
        der = h if r in ('cpu', 'tops') else 1.0
        total = sum(dem[s][r] for s in items) + sum(dem[s][r] for s in crit)
        smallest = min(kind_cap[t][r] * der for t in kinds)
        return total > 0 and total > smallest * (1.0 - 1e-9)
    for s in items:
        m.addConstr(x.sum(s, '*') == 1)
    for s in crit:
        m.addConstr(y.sum(s, '*') == 1)
        for k in cells:
            m.addConstr(x[s, k] + y[s, k] <= 1)
    for s in cold:
        for f in cells:
            for k in cells:
                if f != k:
                    m.addConstr(z[s, f, k] >= x[s, f] + y[s, k] - 1)
    binding = [r for r in res if any(dem[s][r] > 0 for s in items) and _binds(r)]
    for k in cells:
        for r in binding:
            der = h if r in ('cpu', 'tops') else 1.0
            cap = gp.quicksum(kind_cap[t][r] * der * Q[k, t] for t in kinds if kind_cap[t][r] > 0)
            nom = gp.quicksum(dem[s][r] * x[s, k] for s in items) \
                + gp.quicksum(dem[s][r] * y[s, k] for s in crit if s in must_hot or r == 'rom')
            m.addConstr(nom <= cap)
            if r == 'rom':
                continue
            for f in cells:
                if f != k:
                    m.addConstr(nom + gp.quicksum(dem[s][r] * z[s, f, k] for s in cold) <= cap)
        for d_cls, members in (slot_classes or {}).items():
            slots = gp.quicksum(int((kind_cap[t]['tops'] * h) // d_cls) * Q[k, t] for t in kinds if (kind_cap[t]['tops'] * h) // d_cls >= 1)
            held = gp.quicksum(x[s, k] for s in members) + gp.quicksum(y[s, k] for s in members if s in must_hot)
            m.addConstr(held <= slots)
            for f in cells:
                if f != k:
                    m.addConstr(held + gp.quicksum(z[s, f, k] for s in members if s in cold) <= slots)
    cost = [gp.quicksum(kind_cost[t] * Q[k, t] for t in kinds) for k in cells]
    for k in range(K - 1):
        m.addConstr(cost[k] >= cost[k + 1])
    m.setObjective(gp.quicksum(cost), GRB.MINIMIZE)
    m.optimize()
    if m.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        return None
    mix = defaultdict(int)
    used = 0
    if m.SolCount > 0:
        for k in cells:
            if any(Q[k, t].X > 0.5 for t in kinds):
                used += 1
            for t in kinds:
                if Q[k, t].X > 0.5:
                    mix[t] += int(round(Q[k, t].X))
    return {'lb': float(m.ObjBound), 'ub': float(m.ObjVal) if m.SolCount > 0 else float('inf'), 'mix': dict(mix),
            'cells': used, 'optimal': m.Status == GRB.OPTIMAL, 'time': float(m.Runtime)}


def _progress(label, r_, floor):
    if r_ is None:
        print(f"  [bound] {label}: infeasible", file=sys.stderr, flush=True)
    else:
        print(f"  [bound] {label}: lb {r_['lb']:.2f} best {r_['ub']:.2f} floor {floor:.2f} {r_['time']:.0f} s"
              + ("" if r_['optimal'] else " (time limit)"), file=sys.stderr, flush=True)


def _module_specs(c, dem, h, must_hot):
    gpu = {f: spec for f, spec in c.hw_cap.items() if spec.get('types') and any(f in c.scs[s]['hw'] and dem[s]['tops'] > 0 for s in c.scs)}
    mods = {}
    for f, spec in gpu.items():
        gt = {str(t['id']): {'tops': float(t['tops']), 'cost': float(t.get('cost') or 0)} for t in spec['types']}
        users = [s for s in c.scs if f in c.scs[s]['hw'] and dem[s]['tops'] > 0]
        small = min(t['tops'] for t in gt.values()) * h
        classes = defaultdict(list)
        for s in users:
            if dem[s]['tops'] > small + 1e-9:
                classes[round(dem[s]['tops'], 6)].append(s)
        load = {'tops': sum(dem[s]['tops'] for s in users) + sum(dem[s]['tops'] for s in users if s in must_hot)}
        mods[f] = {'kinds': sorted(gt), 'cap': {t: {'tops': gt[t]['tops']} for t in gt}, 'cost': {t: gt[t]['cost'] for t in gt},
                   'unit_max': int(spec.get('max_units_per_site') or 4), 'users': users, 'classes': dict(classes),
                   'floor': _cheapest_mix(gt, load, h, 1)[1]}
    return mods


def gpu_module_floor(c, b, settings):
    h, must_hot = b['h'], _must_hot(c)
    dem = {s: {'tops': c.demand(s, 'tops')} for s in c.scs}
    tl, threads = float(settings.get('time_limit_s', 300.0)), int(settings.get('threads', 8))
    out = {}
    for f, mo in _module_specs(c, dem, h, must_hot).items():
        r_ = _k_cells_model(c, len(c.cands), mo['users'], dem, ['tops'], mo['kinds'], mo['cap'], mo['cost'], mo['unit_max'], h,
                            must_hot, mo['classes'], tl, threads)
        _progress(f"{f} K={len(c.cands)} (module set)", r_, mo['floor'])
        if r_ is not None and r_.get('mix'):
            out[f] = r_
    return out


def failover_bound(c, b, settings, presolve=None):
    h, must_hot = b['h'], _must_hot(c)
    res = list(c.res)
    dem = {s: {r: c.demand(s, r) for r in res + ['tops']} for s in c.scs}
    levels = sorted({sc['asil'] for sc in c.scs.values()})
    tl, threads = float(settings.get('time_limit_s', 300.0)), int(settings.get('threads', 8))
    cell_cost = float(c.site_cfg.get('fixed_cost', 0) or 0)
    if c.types:
        kinds = sorted(c.types)
        kind_cap = {t: {r: float(c.types[t].get(r, 0) or 0) for r in res} for t in kinds}
        kind_cost = {t: float(c.types[t]['cost']) for t in kinds}
    else:
        kinds = ['partition']
        kind_cap = {'partition': {r: float(c.part.get(f'{r}_cap', 0) or 0) for r in res}}
        kind_cost = {'partition': float(c.part.get('cost', 0) or 0)}
    if not any(kind_cap[t][r] > 0 for t in kinds for r in res):
        return None
    site_max = int(c.site_cfg.get('max_partitions') or 0) or len(c.scs)
    mix_types = {t: dict(kind_cap[t], cost=kind_cost[t]) for t in kinds}
    floors, load_of = {}, {}
    for a in levels:
        inst = [s for s in c.scs if c.scs[s]['asil'] == a]
        load = {r: sum(dem[s][r] for s in inst) + sum(dem[s][r] for s in inst if s in must_hot)
                + (sum(dem[s][r] for s in inst if s in c.crit and s not in must_hot) if r == 'rom' else 0.0) for r in res}
        load_of[a] = load
        floors[a] = _cheapest_mix(mix_types, load, h, 1)[1]
    mods = _module_specs(c, dem, h, must_hot)
    hw_other = sum(n * p for f, (n, p) in b['hardware'].items() if f not in mods)
    ports = b['cost']['ports']
    cables = 0.0
    for d in {d for sc in c.scs.values() for d in sc['devices']}:
        n = c.devices[d]['iface']
        if n in SHARED_BUSES or d not in c.devices:
            continue
        cables += c.dev_factor * min(c.dist(d, k) for k in c.cands) * float(c.ct.get(n, {}).get('cost', 0.0))
    rest0 = hw_other + ports + cables
    K_all = len(c.cands)
    tail_level, tail_mod = {}, {}
    k0 = 2 if c.crit else 1
    if K_all > k0:
        for a in levels:
            items = [s for s in c.scs if c.scs[s]['asil'] == a]
            r_ = _k_cells_model(c, K_all, items, dem, res, kinds, kind_cap, kind_cost, site_max, h, must_hot, None, tl, threads, presolve)
            _progress(f"ASIL {a} K={K_all} (floor)", r_, floors[a])
            if r_ is not None:
                tail_level[a] = r_
                floors[a] = max(floors[a], r_['lb'])
        for f, mo in mods.items():
            r_ = _k_cells_model(c, K_all, mo['users'], dem, ['tops'], mo['kinds'], mo['cap'], mo['cost'], mo['unit_max'], h, must_hot,
                                mo['classes'], tl, threads, presolve)
            _progress(f"{f} K={K_all} (floor)", r_, mo['floor'])
            if r_ is not None:
                tail_mod[f] = r_
                mo['floor'] = max(mo['floor'], r_['lb'])
    floor_sum = sum(floors.values()) + sum(mo['floor'] for mo in mods.values())
    seen = defaultdict(list)
    rows, best = [], None
    done_level, done_mod = {}, {}
    for K in range(k0, K_all + 1):
        if K == K_all:
            for a in levels:
                done_level.setdefault(a, tail_level.get(a))
            for f in mods:
                done_mod.setdefault(f, tail_mod.get(f))
        rest = rest0 + K * cell_cost
        if best is not None and floor_sum + rest >= best['total'] - 1e-6:
            break
        parts, feasible = {}, True
        for a in levels:
            if done_level.get(a) is not None:
                parts[a] = done_level[a]
                continue
            items = [s for s in c.scs if c.scs[s]['asil'] == a]
            r_ = _k_cells_model(c, K, items, dem, res, kinds, kind_cap, kind_cost, site_max, h, must_hot, None, tl, threads, presolve)
            _progress(f"ASIL {a} K={K}", r_, floors[a])
            if r_ is not None and r_['optimal']:
                seen[a].append(r_['lb'])
            if r_ is None:
                feasible = False
                break
            parts[a] = r_
            if r_['optimal'] and r_['lb'] <= floors[a] + 1e-6:
                done_level[a] = r_
        mres = {}
        if feasible:
            for f, mo in mods.items():
                if done_mod.get(f) is not None:
                    mres[f] = done_mod[f]
                    continue
                r_ = _k_cells_model(c, K, mo['users'], dem, ['tops'], mo['kinds'], mo['cap'], mo['cost'], mo['unit_max'], h, must_hot,
                                    mo['classes'], tl, threads, presolve)
                _progress(f"{f} K={K}", r_, mo['floor'])
                if r_ is not None and r_['optimal']:
                    seen[f].append(r_['lb'])
                if r_ is None:
                    feasible = False
                    break
                mres[f] = r_
                if r_['optimal'] and r_['lb'] <= mo['floor'] + 1e-6:
                    done_mod[f] = r_
        if not feasible:
            rows.append({'K': K, 'feasible': False})
            continue
        p_sum = sum(v['lb'] for v in parts.values())
        m_sum = sum(v['lb'] for v in mres.values())
        total = p_sum + m_sum + rest
        row = {'K': K, 'feasible': True, 'partitions': p_sum, 'modules': m_sum, 'cells': K * cell_cost, 'hw_ports': hw_other + ports,
               'cables': cables, 'total': total, 'optimal': all(v['optimal'] for v in list(parts.values()) + list(mres.values()))}
        rows.append(row)
        if best is None or total < best['total'] - 1e-9:
            best = dict(row, levels=parts, modules_by_feature=mres)
    bad = [(a, floors[a], min(v)) for a, v in seen.items() if a in floors and v and floors[a] > min(v) + 1e-6]
    bad += [(f, mods[f]['floor'], min(v)) for f, v in seen.items() if f in mods and v and mods[f]['floor'] > min(v) + 1e-6]
    if bad:
        print(f"  [bound] WARNING: inconsistent floors {bad}" + ("; computing again with presolve off" if presolve is None else ""),
              file=sys.stderr, flush=True)
        if presolve is None:
            return failover_bound(c, b, settings, presolve=0)
    if best is None:
        return None
    return {'bound': best['total'], 'K': best['K'], 'rows': rows, 'levels': best['levels'], 'modules': best['modules_by_feature'],
            'cables': cables, 'hw_other': hw_other, 'ports': ports, 'optimal': best['optimal'], 'floors': floors,
            'module_floors': {f: mo['floor'] for f, mo in mods.items()}}


def report(c, b, fb=None):
    s = c.sol
    cost = float(s['total_cost'])
    bound = max(b['bound'], fb['bound']) if fb else b['bound']
    gap = (cost - bound) / cost if cost else float('inf')
    rel = os.path.relpath(c.path, REPO) if c.path.startswith(REPO) else c.path
    which = (f"failover-aware over K={fb['K']} cells; counting {b['bound']:.2f}" if fb and fb['bound'] >= b['bound']
             else (f"counting; failover-aware {fb['bound']:.2f}" if fb else "counting"))
    lines = [f"[bound] {rel}: cost {cost:.2f}, bound {bound:.2f} ({which}), gap {100 * gap:.2f} % "
             f"(rho {c.rho:g}, {len(c.scs)} components, {len(c.cands)} cells, {b['critical']} critical of which "
             f"{b['must_hot']} must be hot, best derating {b['h']:g})"]
    design = {'partitions': (int(s.get('num_partitions', 0)), float(s.get('partition_cost', 0) or 0)),
              'units': (int(s.get('num_locations_used', 0)), float(s.get('site_cost', 0) or 0)),
              'hardware': (sum(c.hw_units.values()), float(s.get('hardware_cost', 0) or 0)),
              'ports': (sum(c.ports.values()), float(s.get('interface_cost', 0) or 0)),
              'network': (len(s.get('comm_links', [])), float(s.get('cable_cost', 0) or 0) + float(s.get('comm_cost', 0) or 0))}
    bound_n = {'partitions': b['partitions'], 'units': b['units'], 'hardware': sum(n for n, _ in b['hardware'].values()),
               'ports': sum(v[0] for v in b['ports'].values()), 'network': 0}
    lines.append(f"  {'part':<12}{'design':>8}{'cost':>12}{'bound':>8}{'cost':>12}{'slack':>10}")
    for k in ('partitions', 'units', 'hardware', 'ports', 'network'):
        n, v = design[k]
        lines.append(f"  {k:<12}{n:>8}{v:>12.2f}{bound_n[k]:>8}{b['cost'][k]:>12.2f}{v - b['cost'][k]:>10.2f}")
    lines.append(f"  {'total':<12}{'':>8}{cost:>12.2f}{'':>8}{b['bound']:>12.2f}{cost - b['bound']:>10.2f}")
    counts = defaultdict(int)
    for key, n in (s.get('partition_counts') or {}).items():
        m = re.match(r'^(?P<site>.+)_asil(?P<asil>\d+)$', key)
        counts[int(m['asil'])] += int(n)
    lines.append("  partitions per ASIL (bound = nominal load over the partition caps, rounded up):")
    lines.append(f"  {'ASIL':<6}{'comps':>7}{'crit':>6}{'hot':>5}{'cpu':>8}{'ram':>8}{'rom':>8}{'bound':>7}{'design':>8}")
    for r in b['partition_rows']:
        ld = r['load']
        lines.append(f"  {r['asil']:<6}{r['components']:>7}{r['critical']:>6}{r['must_hot']:>5}"
                     f"{ld.get('cpu', 0):>8.2f}{ld.get('ram', 0):>8.2f}{ld.get('rom', 0):>8.2f}{r['bound']:>7}{counts.get(r['asil'], 0):>8}"
                     + (f"   cheapest blocks {r['mix']} = {r['mix_cost']:.2f}" if 'mix' in r else ""))
    hw = ", ".join(f"{f} {n} x {p:g}" for f, (n, p) in b['hardware'].items()) or "none"
    ports = ", ".join(f"{t} {k} x {each:g}" for t, (k, each, _) in b['ports'].items()) or "none"
    lines.append(f"  hardware bound: {hw}; port bound: {ports}")
    if fb:
        lines.append("  failover-aware bound by the number K of cells with partitions (per ASIL level the cheapest blocks over K "
                     "interchangeable cells with the model's pooled nominal and failover rules; GPU modules with the slot rule; "
                     "K cells; other hardware and ports as above; device cables to the nearest cell):")
        lines.append(f"  {'K':>4}{'partitions':>12}{'modules':>10}{'cells':>8}{'hw+ports':>10}{'cables':>9}{'total':>11}")
        for r in fb['rows']:
            if not r['feasible']:
                lines.append(f"  {r['K']:>4}  infeasible")
                continue
            lines.append(f"  {r['K']:>4}{r['partitions']:>12.2f}{r['modules']:>10.2f}{r['cells']:>8.0f}{r['hw_ports']:>10.2f}{r['cables']:>9.2f}{r['total']:>11.2f}"
                         + ("" if r['optimal'] else "   (a model at its time limit: dual bound used)") + ("   <- min" if r['K'] == fb['K'] else ""))
        for a, v in sorted(fb['levels'].items()):
            lines.append(f"  ASIL {a} at K={fb['K']}: blocks {v['mix']} in {v['cells']} cells = {v['lb']:.2f}"
                         + ("" if v['optimal'] else f" (time limit; best found {v['ub']:.2f})"))
        for f, v in sorted(fb['modules'].items()):
            lines.append(f"  {f} at K={fb['K']}: modules {v['mix']} in {v['cells']} cells = {v['lb']:.2f}"
                         + ("" if v['optimal'] else f" (time limit; best found {v['ub']:.2f})"))
        lines.append(f"  device cables to the nearest candidate cell {fb['cables']:.2f}; bus lines, backbone and switches 0")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description='Lower bound on the cost of any valid design, from a relaxation that keeps the capacity and failover rules '
                                             'and drops the geometry, the latencies and the network, and the gap of the given design to it.')
    ap.add_argument('path', help='solution_<i>.json or a run directory')
    ap.add_argument('--config_dir', default=None, help='default: the run\'s config dir (run_info.json), else configs')
    ap.add_argument('--architecture', default=None, help='default: the run\'s architecture JSON (run_info.json)')
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.path, 'solution_*.json')), key=lambda p: int(re.findall(r'(\d+)\.json$', p)[0])) \
        if os.path.isdir(a.path) else [a.path]
    for f in files:
        c = SolutionCheck(f, a.config_dir, a.architecture)
        c.load()
        c.parse()
        b = design_bound(c)
        st = bound_settings(c)
        fb = failover_bound(c, b, st) if st.get('failover', True) else None
        print(report(c, b, fb), flush=True)
    return 0 if files else 1


if __name__ == '__main__':
    sys.exit(main())

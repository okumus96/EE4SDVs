import copy
import json
import os
import re
import time
from collections import defaultdict, deque

from optimizer import UNCERTAINTY_BLOCKS, process_memory_gb
from optimizer_utils import is_fail_operational
import optimizer_utils


def _polish_worker(job):
    import sys
    t0 = time.time()
    if job.get('distance_table'):
        optimizer_utils.set_distance_table(*job['distance_table'])
    if job.get('steiner_cache'):
        optimizer_utils._STEINER_CACHE.update(job['steiner_cache'])
    if job.get('log'):
        sys.stdout = sys.stderr = open(job['log'], 'w', buffering=1)
    from optimizer import AssignmentOptimizer
    opt = AssignmentOptimizer()
    res = opt.optimize(*job['args'], **job['kwargs'])
    return job['cell'], res, (dict(opt.last_values) if res is not None else None), round(time.time() - t0, 1)

DEFAULTS = {'max_group': 60, 'min_group': 20, 'piece_time_s': 300, 'piece_gap': 0.01, 'piece_phase2_s': 10,
            'max_backtrack': 3, 'lock_gap': 0.10, 'piece_time_max_s': 1200, 'static_capacity': False,
            'polish_rounds': 0, 'polish_time_s': 600, 'polish_partners': 10, 'polish_mode': 'cell', 'polish_max_free': 120, 'polish_parallel': 3, 'polish_passes': 4,
            'polish_stages': None, 'polish_allow_cells': None, 'polish_fix_patterns': None, 'polish_order': 'fewest', 'build_parallel': 1, 'build_lookahead': 8,
            'polish_split_cells': True, 'polish_all': False, 'polish_min_gain': 0.001,
            'step_gurobi_params': 'NoRelHeurTime=20', 'final_network_s': 0, 'final_network_gap': 0.0,
            'gateway_in_build': False,
            'polish_gpu': 'capped', 'polish_gpu_time_s': 3600, 'polish_gpu_fix_patterns': None,
            'merge_rounds': 10, 'merge_time_s': 1800, 'merge_targets': 'every'}
GPU_ROUND_FIX = ['^attachS_', '^attachA_', '^bus_', '^busLen_', '^busHigh_', '^busUnit_', '^busDev_', '^busGw_',
                 '^switch_node_', '^hw_.*_SWITCH$']
_GPU_CAP = {}


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        self.parent[self.find(i)] = self.find(j)


def plan_pieces(scs, comm_matrix, chains, fail_cfg, failures_on, cfg):
    ids = [sc.id for sc in scs]
    present = set(ids)
    sc_by_id = {sc.id: sc for sc in scs}
    max_g, min_g = int(cfg['max_group']), int(cfg['min_group'])

    accel_pieces = []
    if cfg.get('accel_first', True):
        accel = sorted((s for s in ids if float(getattr(sc_by_id[s], 'tops_req', 0) or 0) > 0),
                       key=lambda s: (-float(sc_by_id[s].tops_req), s))
        accel_pieces = [accel[k:k + max_g] for k in range(0, len(accel), max_g)]
        present -= set(accel)
        ids = [s for s in ids if s in present]

    units = []
    for ch in (chains or []):
        members = [s for s in ch.get('swcs', []) if s in present]
        if members:
            units.append((members, float(ch.get('net_budget_ms', float('inf')))))
    for sc in scs:
        p = sc.redundant_with
        if p and p in present and sc.id < p:
            units.append(([sc.id, p], float('inf')))
    uf = _UnionFind(len(units))
    owner = {}
    for k, (members, _b) in enumerate(units):
        for s in members:
            if s in owner:
                uf.union(k, owner[s])
            else:
                owner[s] = k
    groups = defaultdict(list)
    for k in range(len(units)):
        groups[uf.find(k)].append(k)

    def is_crit(members):
        return failures_on and any(is_fail_operational(sc_by_id[s], fail_cfg) for s in members)

    ordered = []
    for ks in groups.values():
        ks = sorted(ks, key=lambda k: (units[k][1], k))
        members, seen = [], set()
        for k in ks:
            for s in units[k][0]:
                if s not in seen:
                    seen.add(s)
                    members.append(s)
        ordered.append({'members': members, 'units': ks, 'budget': units[ks[0]][1], 'crit': is_crit(members)})
    ordered.sort(key=lambda g: (not g['crit'], g['budget'], g['members'][0]))

    pieces = []
    for g in ordered:
        if len(g['members']) <= max_g:
            pieces.append(list(g['members']))
            continue
        cur, seen = [], set()
        for k in g['units']:
            new = [s for s in units[k][0] if s not in seen]
            if cur and len(cur) + len(new) > max_g:
                pieces.append(cur)
                cur = []
            cur += new
            seen.update(new)
        if cur:
            pieces.append(cur)

    in_piece = {s: k for k, p in enumerate(pieces) for s in p}
    vol, nbr = defaultdict(float), defaultdict(set)
    for lk in comm_matrix:
        a, b = lk['src'], lk['dst']
        if a in present and b in present and a != b:
            vol[a, b] += float(lk.get('volume', 0.0))
            vol[b, a] += float(lk.get('volume', 0.0))
            nbr[a].add(b)
            nbr[b].add(a)
    pool = []
    for s in ids:
        if s in in_piece:
            continue
        for _v, t in sorted(((vol[s, t], t) for t in nbr[s] if t in in_piece), reverse=True):
            if len(pieces[in_piece[t]]) < max_g:
                pieces[in_piece[t]].append(s)
                in_piece[s] = in_piece[t]
                break
        else:
            pool.append(s)
    pool_set = set(pool)
    pool.sort(key=lambda s: (-sum(vol[s, t] for t in nbr[s]), s))
    taken = set()
    for seed in pool:
        if seed in taken:
            continue
        cur, q, queued = [], deque([seed]), {seed}
        while q and len(cur) < max_g:
            s = q.popleft()
            cur.append(s)
            taken.add(s)
            for t in sorted(nbr[s] & pool_set, key=lambda t: (-vol[s, t], t)):
                if t not in queued and t not in taken:
                    queued.add(t)
                    q.append(t)
        pieces.append(cur)
    total_cpu = sum(sc.cpu_req for sc in scs) or 1
    total_ram = sum(sc.ram_req for sc in scs) or 1

    def hardness(piece):
        heavy = (sum(sc_by_id[s].cpu_req for s in piece) > 0.1 * total_cpu) or (sum(sc_by_id[s].ram_req for s in piece) > 0.1 * total_ram)
        crit = sum(1 for s in piece if is_fail_operational(sc_by_id[s], fail_cfg)) if failures_on else 0
        traffic = sum(vol[s, t] for s in piece for t in nbr[s])
        budget = min((units[k][1] for s in piece if s in owner for k in [owner[s]]), default=float('inf'))
        return ((not heavy) if cfg.get('heavy_first', True) else False, -crit, -traffic, budget)
    pieces.sort(key=hardness)
    merged = []
    for p in pieces:
        if merged and (len(merged[-1]) < min_g or len(p) < min_g) and len(merged[-1]) + len(p) <= max_g:
            merged[-1] = merged[-1] + p
        else:
            merged.append(p)
    merged = accel_pieces + merged
    assert sorted(s for p in merged for s in p) == sorted(sc.id for sc in scs), "every component in exactly one piece"
    return merged


def sub_architecture(scs, comm_matrix, sensors, actuators, chains, keep):
    keep = set(keep)
    present = {sc.id for sc in scs}
    sub = []
    for sc in scs:
        if sc.id not in keep:
            continue
        c = copy.copy(sc)
        if c.redundant_with and c.redundant_with not in keep:
            c.redundant_with = None
        sub.append(c)
    comm = [lk for lk in comm_matrix if lk['src'] in keep and lk['dst'] in keep]
    dev_s = {d for sc in sub for d in (sc.sensors or [])}
    dev_a = {d for sc in sub for d in (sc.actuators or [])}
    sens = [s for s in sensors if s.id in dev_s]
    acts = [a for a in actuators if a.id in dev_a]
    chs = [ch for ch in (chains or []) if any(s in present for s in ch.get('swcs', [])) and all(s in keep for s in ch.get('swcs', []) if s in present)]
    return sub, comm, sens, acts, chs


def build_by_pieces(opt, scs, locations, sensors, actuators, cable_types, comm_matrix, *, partitions, hardwares,
                    interfaces, backbone, uncertainty, chains, gurobi_params, cfg, phase2_time):
    cfg = {**DEFAULTS, **(cfg or {})}
    if cfg.get('static_capacity', False):
        backbone = {**(backbone or {}), 'upfront_paths': True, 'cell_cut_capacity': True}
    params = (uncertainty or {}).get('params', {})
    blocks_on = any((uncertainty or {}).get(k) for k in UNCERTAINTY_BLOCKS)
    levels = [float(a) for a in params.get('rho', {}).get('levels', [1.0])] if blocks_on else [None]
    print("=" * 80)
    print(f"FIRST DESIGN BY PIECES: rho in {levels}, max_group {cfg['max_group']}, min_group {cfg['min_group']}, "
          f"{cfg['piece_time_s']} s per step (--time_limit not used)")
    print("=" * 80)
    solutions = []
    for alpha in levels:
        unc = dict(uncertainty or {})
        if alpha is not None:
            unc['rho'] = alpha
            unc['growth_alpha'] = alpha
            if alpha <= 0.0:
                unc['failures'] = False
                unc['redundancy'] = False
            print(f"\n[Pieces] rho = {alpha:.2f}")
            unc = optimizer_utils.with_gradual_standbys(unc, scs, chains)
        sol = _build_level(opt, scs, locations, sensors, actuators, cable_types, comm_matrix, partitions=partitions,
                           hardwares=hardwares, interfaces=interfaces, backbone=backbone, uncertainty=unc,
                           chains=chains, gurobi_params=gurobi_params, cfg=cfg, phase2_time=phase2_time)
        if sol is None:
            print(f"[Pieces] rho = {alpha}: no design; sweep stopped")
            break
        solutions.append(sol)
    return solutions


def _build_level(opt, scs, locations, sensors, actuators, cable_types, comm_matrix, *, partitions, hardwares,
                 interfaces, backbone, uncertainty, chains, gurobi_params, cfg, phase2_time):
    failures_on = bool(uncertainty.get('failures'))
    fail_cfg = uncertainty.get('params', {}).get('failures', {}) if failures_on else {}
    pieces = plan_pieces(scs, comm_matrix, chains, fail_cfg, failures_on, cfg)
    backbone_full = backbone
    if (backbone or {}).get('bus_gateway') and not cfg.get('gateway_in_build', True):
        backbone = {**backbone, 'bus_gateway': False}
        print("[Pieces] bus gateway off during the build (pieces.gateway_in_build false); on from the polish", flush=True)
    present = {sc.id for sc in scs}
    by_id = {sc.id: sc for sc in scs}
    accel_steps = 0
    for p in pieces:
        if p and all(float(getattr(by_id[s], 'tops_req', 0) or 0) > 0 for s in p):
            accel_steps += 1
        else:
            break
    traffic = defaultdict(float)
    for lk in comm_matrix:
        traffic[lk['src']] += float(lk.get('volume', 0.0))
        traffic[lk['dst']] += float(lk.get('volume', 0.0))

    def group_info(g):
        return {'fail_op': sum(1 for s in g if failures_on and is_fail_operational(by_id[s], fail_cfg)),
                'traffic_mbit': round(sum(traffic[s] for s in g), 1),
                'devices': sum(len(by_id[s].sensors or []) + len(by_id[s].actuators or []) for s in g),
                'chains': sum(1 for ch in (chains or []) if any(s in g for s in ch.get('swcs', []))),
                'cpu': sum(by_id[s].cpu_req for s in g), 'ram_kb': sum(by_id[s].ram_req for s in g)}
    chain_sets = [{s for s in ch.get('swcs', []) if s in present} for ch in (chains or [])]
    print(f"[Pieces] {len(scs)} components in {len(pieces)} pieces: sizes {[len(p) for p in pieces]}")
    step_params = ','.join(x for x in (gurobi_params, cfg.get('step_gurobi_params')) if x)
    placed_x, placed_r, steps, sol = {}, {}, [], None
    opt.last_values = None
    k, unlock, t0 = 0, 0, time.time()
    if cfg.get('resume_dump'):
        with open(cfg['resume_dump']) as f:
            dump = json.load(f)
        if {s for p in dump['pieces'] for s in p} != present:
            raise ValueError(f"[Pieces] resume_dump {cfg['resume_dump']}: its groups do not hold the components of this instance")
        pieces = [list(p) for p in dump['pieces']]
        placed_x, placed_r, k = dict(dump['placed_x']), dict(dump['placed_r']), int(dump['step'])
        print(f"[Pieces] resumed from {cfg['resume_dump']}: steps 1..{k} placed ({len(placed_x)} components, "
              f"{len(placed_r)} standbys, cost {float(dump.get('cost') or 0):.2f}), next step {k + 1}/{len(pieces)}", flush=True)
    msg_nbr = defaultdict(set)
    for lk in comm_matrix:
        if lk['src'] != lk['dst']:
            msg_nbr[lk['src']].add(lk['dst'])
            msg_nbr[lk['dst']].add(lk['src'])
    chains_of = defaultdict(set)
    for ci, cs in enumerate(chain_sets):
        for s_ in cs:
            chains_of[s_].add(ci)

    def independent(a, b):
        sa, sb = set(a), set(b)
        return not any(msg_nbr[s_] & sb for s_ in sa) and not (set().union(*(chains_of[s_] for s_ in sa)) & set().union(*(chains_of[s_] for s_ in sb)))

    dev_obj = {d.id: d for d in list(sensors) + list(actuators)}
    loc_obj = {loc.id: loc for loc in locations}
    pair_vol = defaultdict(float)
    for lk in comm_matrix:
        pair_vol[lk['src'], lk['dst']] += float(lk.get('volume', 0.0))
        pair_vol[lk['dst'], lk['src']] += float(lk.get('volume', 0.0))

    def _own_cells(k0, m):
        cells = [loc.id for loc in locations]
        share = max(2, -(-len(cells) // m))
        used = defaultdict(int)
        for s_, c in placed_x.items():
            used[c] += 1
        for s_, c in placed_r.items():
            used[c] += 1
        score = {}
        for i in range(m):
            g = pieces[k0 + i]
            devs = [dev_obj[d] for s_ in g for d in (by_id[s_].sensors or []) + (by_id[s_].actuators or []) if d in dev_obj]
            partners = [(placed_x[t], 1.0 + pair_vol[s_, t]) for s_ in g for t in msg_nbr[s_] if t in placed_x]
            if devs or partners:
                score[i] = {c: sum(optimizer_utils.get_distance(loc_obj[c].location, d.location) for d in devs)
                               + sum(w * optimizer_utils.get_distance(loc_obj[c].location, loc_obj[pc].location) for pc, w in partners)
                            for c in cells}
        own = {i: [] for i in range(m)}
        taken = set()
        for c in sorted(cells, key=lambda c_: (used[c_] == 0, c_)):
            ranked = sorted((score[i][c], i) for i in score if len(own[i]) < share)
            if ranked:
                for _v, i in ranked:
                    if c in sorted(cells, key=lambda c_: score[i][c_])[:share]:
                        own[i].append(c)
                        taken.add(c)
                        break
        rest = sorted((c for c in cells if c not in taken), key=lambda c_: (used[c_] == 0, used[c_], c_))
        for c in rest:
            i = min(range(m), key=lambda i_: (i_ in score, len(own[i_])))
            own[i].append(c)
        for i in range(m):
            while len(own[i]) < 2:
                donor = max(range(m), key=lambda j: len(own[j]))
                own[i].append(own[donor].pop())
        return own

    def _parallel_batch(k0, m, b_no):
        nonlocal sol, placed_x, placed_r
        import concurrent.futures as cf
        import multiprocessing as mp
        threads = None
        for item in (gurobi_params or '').split(','):
            if item.strip().lower().startswith('threads='):
                threads = int(item.split('=')[1])
        per_job = max(1, (threads or os.cpu_count() or m) // m)
        locked = {s_ for p_ in pieces[:k0] for s_ in p_}
        fixed = {'x': {s_: c for s_, c in placed_x.items() if s_ in locked}, 'r': {s_: c for s_, c in placed_r.items() if s_ in locked}}
        own = _own_cells(k0, m)
        log_dir = cfg.get('dump_dir') or '.'
        os.makedirs(log_dir, exist_ok=True)
        jobs = []
        for i in range(m):
            keep_i = [s_ for p_ in pieces[:k0] for s_ in p_] + list(pieces[k0 + i])
            sub_i, comm_i, sens_i, acts_i, chs_i = sub_architecture(scs, comm_matrix, sensors, actuators, chains, keep_i)
            jobs.append({'cell': i, 'log': os.path.join(log_dir, f"build_step{k0 + i + 1}.log"),
                         'distance_table': (list(optimizer_utils._DISTANCE_TABLE[0]), optimizer_utils._DISTANCE_TABLE[1]) if optimizer_utils._DISTANCE_TABLE else None,
                         'steiner_cache': dict(optimizer_utils._STEINER_CACHE),
                         'args': (sub_i, locations, sens_i, acts_i, cable_types, comm_i),
                         'kwargs': dict(partitions=partitions, hardwares=hardwares, interfaces=interfaces,
                                        time_limit=float(cfg['piece_time_s']), mip_gap=float(cfg['piece_gap']),
                                        uncertainty=uncertainty, backbone=backbone, phase2_time=float(cfg['piece_phase2_s']),
                                        phase2_cap=float(cfg['piece_phase2_s']),
                                        gurobi_params=','.join(x for x in (step_params, f'Threads={per_job}') if x),
                                        chains=chs_i, start=opt.last_values,
                                        fixed={**fixed, 'allow': {s_: own[i] for s_ in pieces[k0 + i]}}, lns=None)})
        print(f"\n[Pieces] parallel batch {b_no}: steps {k0 + 1}-{k0 + m} ({[len(pieces[k0 + i]) for i in range(m)]} new components), "
              f"{len(locked)} locked, {m} processes x {per_job} threads, {float(cfg['piece_time_s']):.0f} s; own cells: "
              + "; ".join(f"step {k0 + i + 1}: {', '.join(own[i])}" for i in range(m)), flush=True)
        t_b = time.time()
        with cf.ProcessPoolExecutor(max_workers=m, mp_context=mp.get_context('spawn')) as ex:
            results = list(ex.map(_polish_worker, jobs))
        if any(res_i is None for _i, res_i, _v, _sec in results):
            print(f"[Pieces] parallel batch {b_no}: a group found no design ({[i + k0 + 1 for i, r_, _v, _s in results if r_ is None]}); "
                  f"placing these {m} groups one by one", flush=True)
            return False
        new_x, new_r = dict(fixed['x']), dict(fixed['r'])
        for i, res_i, _v, _sec in results:
            g = set(pieces[k0 + i])
            new_x.update({s_: c for s_, c in res_i['assignment'].items() if s_ in g})
            new_r.update({s_: v['cell'] for s_, v in (res_i.get('standby') or {}).items() if s_ in g})
        keep_m = [s_ for p_ in pieces[:k0 + m] for s_ in p_]
        sub_m, comm_m, sens_m, acts_m, chs_m = sub_architecture(scs, comm_matrix, sensors, actuators, chains, keep_m)
        last_m = k0 + m == len(pieces)
        t_c = time.time()
        res_m = opt.optimize(sub_m, locations, sens_m, acts_m, cable_types, comm_m, partitions=partitions, hardwares=hardwares,
                             interfaces=interfaces, time_limit=float(cfg['piece_time_s']), mip_gap=float(cfg['piece_gap']),
                             uncertainty=uncertainty, backbone=backbone,
                             phase2_time=(phase2_time if last_m else float(cfg['piece_phase2_s'])),
                             phase2_cap=(None if last_m else float(cfg['piece_phase2_s'])),
                             gurobi_params=step_params, chains=chs_m, start=None, fixed={'x': new_x, 'r': new_r}, lns=None)
        if res_m is None:
            print(f"[Pieces] parallel batch {b_no}: the groups do not fit together (combined check found no design in "
                  f"{time.time() - t_c:.0f} s); placing these {m} groups one by one", flush=True)
            return False
        sol = res_m
        placed_x = dict(sol['assignment'])
        placed_r = {s_: v['cell'] for s_, v in (sol.get('standby') or {}).items()}
        ph_m = sol['solve_info']['phases'][0]
        for i, res_i, _v, sec in results:
            ph_i = res_i['solve_info']['phases'][0]
            steps.append({'step': k0 + i + 1, 'new': len(pieces[k0 + i]), 'free': len(pieces[k0 + i]), 'locked': len(locked),
                          'unlocked': 0, 'seconds': sec, **group_info(pieces[k0 + i]), 'result': f'parallel batch {b_no}',
                          'cost': sol.get('total_cost'), 'gap': ph_i.get('mip_gap'), 'cells': len(set(placed_x.values()))})
        if cfg.get('dump_dir'):
            with open(os.path.join(cfg['dump_dir'], f"pieces_step_{k0 + m}.json"), 'w') as f:
                json.dump({'step': k0 + m, 'pieces': pieces, 'placed_x': placed_x, 'placed_r': placed_r,
                           'cost': sol.get('total_cost')}, f, indent=1)
        print(f"[Pieces] parallel batch {b_no} done in {time.time() - t_b:.0f} s (groups {max(r_[3] for r_ in results):.0f} s, "
              f"combined check {time.time() - t_c:.0f} s): cost {sol.get('total_cost'):.2f}, gap of the check "
              f"{100 * (ph_m.get('mip_gap') or 0):.2f} %, {len(set(placed_x.values()))} cells used", flush=True)
        return True

    seq_until, batch_no = 0, 0
    while k < len(pieces):
        if int(cfg.get('build_parallel', 1)) > 1 and unlock == 0 and k >= seq_until:
            window = list(range(k + 1, min(len(pieces), k + int(cfg.get('build_lookahead', 8)))))
            chosen, skipped = [k], []
            for j in window:
                if len(chosen) < int(cfg['build_parallel']) and all(independent(pieces[j], pieces[i]) for i in chosen + skipped):
                    chosen.append(j)
                else:
                    skipped.append(j)
            if len(chosen) > 1:
                pieces[k:] = [pieces[i] for i in chosen] + [pieces[i] for i in range(k, len(pieces)) if i not in chosen]
                m = len(chosen)
                batch_no += 1
                ok = _parallel_batch(k, m, batch_no)
                if ok:
                    k += m
                    continue
                seq_until = k + m
        free_from = max(0, k - unlock)
        keep = [s for p in pieces[:k + 1] for s in p]
        locked = {s for p in pieces[:free_from] for s in p}
        fixed = {'x': {s: c for s, c in placed_x.items() if s in locked},
                 'r': {s: c for s, c in placed_r.items() if s in locked}}
        sub, comm, sens, acts, chs = sub_architecture(scs, comm_matrix, sensors, actuators, chains, keep)
        n_free = len(keep) - len(locked)
        last = k == len(pieces) - 1
        t_step = float(cfg['piece_time_s']) * (1 + unlock) * float(cfg.get('_t_scale', 1.0))
        if k < accel_steps and cfg.get('accel_time_s'):
            t_step = max(t_step, float(cfg['accel_time_s']))
        print(f"\n[Pieces] step {k + 1}/{len(pieces)}: {n_free} free components ({len(pieces[k])} new"
              f"{', ' + str(unlock) + ' previous piece(s) unlocked' if unlock else ''}), {len(locked)} locked, "
              f"{len(chs)} chains, {len(sens) + len(acts)} devices, {t_step:.0f} s", flush=True)
        t1 = time.time()
        res = opt.optimize(sub, locations, sens, acts, cable_types, comm, partitions=partitions, hardwares=hardwares,
                           interfaces=interfaces, time_limit=t_step, mip_gap=float(cfg['piece_gap']),
                           uncertainty=uncertainty, backbone=backbone,
                           phase2_time=(phase2_time if last else float(cfg['piece_phase2_s'])),
                           phase2_cap=(None if last else float(cfg['piece_phase2_s'])),
                           gurobi_params=step_params, chains=chs, start=opt.last_values, fixed=fixed, lns=None, reuse=True)
        step = {'step': k + 1, 'new': len(pieces[k]), 'free': n_free, 'locked': len(locked), 'unlocked': unlock,
                'seconds': round(time.time() - t1, 1), **group_info(pieces[k])}
        if res is None:
            step['result'] = 'no design'
            steps.append(step)
            t_used = t_step
            while res is None and t_used < float(cfg['piece_time_max_s']) and getattr(opt, 'last_status', None) != 'INFEASIBLE' \
                    and getattr(opt, '_resumable', False):
                extra = min(t_used, float(cfg['piece_time_max_s']) - t_used)
                print(f"[Pieces] step {k + 1}: no design in {t_used:.0f} s; solving on for {extra:.0f} s more (model kept)", flush=True)
                t1 = time.time()
                res = opt.optimize(sub, locations, sens, acts, cable_types, comm, partitions=partitions, hardwares=hardwares,
                                   interfaces=interfaces, time_limit=extra, mip_gap=float(cfg['piece_gap']),
                                   uncertainty=uncertainty, backbone=backbone,
                                   phase2_time=(phase2_time if last else float(cfg['piece_phase2_s'])),
                                   phase2_cap=(None if last else float(cfg['piece_phase2_s'])),
                                   gurobi_params=step_params, chains=chs, start=None, fixed=fixed, lns=None, reuse=True, resume=True)
                t_used += extra
                step = {'step': k + 1, 'new': len(pieces[k]), 'free': n_free, 'locked': len(locked), 'unlocked': unlock,
                        'seconds': round(time.time() - t1, 1), 'resumed': True, **group_info(pieces[k])}
                if res is None:
                    step['result'] = 'no design'
                    steps.append(step)
            t_step = t_used
        if res is None and getattr(opt, 'last_status', None) == 'UNPACKABLE' and not any(s.get('reseeded') for s in steps if s['step'] == k + 1):
            print(f"[Pieces] step {k + 1}: no packable design after the packing cuts; one more solve with Seed=1 ({t_step:.0f} s)", flush=True)
            t1 = time.time()
            res = opt.optimize(sub, locations, sens, acts, cable_types, comm, partitions=partitions, hardwares=hardwares,
                               interfaces=interfaces, time_limit=t_step, mip_gap=float(cfg['piece_gap']),
                               uncertainty=uncertainty, backbone=backbone,
                               phase2_time=(phase2_time if last else float(cfg['piece_phase2_s'])),
                               phase2_cap=(None if last else float(cfg['piece_phase2_s'])),
                               gurobi_params=','.join(x for x in (step_params, 'Seed=1') if x), chains=chs, start=opt.last_values,
                               fixed=fixed, lns=None, reuse=True)
            step = {'step': k + 1, 'new': len(pieces[k]), 'free': n_free, 'locked': len(locked), 'unlocked': unlock,
                    'seconds': round(time.time() - t1, 1), 'reseeded': True, **group_info(pieces[k])}
            if res is None:
                step['result'] = 'no design'
                steps.append(step)
        if res is None:
            unlock += 1
            cfg = {**cfg, '_t_scale': max(1.0, float(cfg['piece_time_max_s']) / float(cfg['piece_time_s']) / (1 + unlock))}
            print(f"[Pieces] step {k + 1}: no design in {t_step:.0f} s; unlocking the previous {unlock} piece(s)", flush=True)
            if unlock > int(cfg['max_backtrack']) or free_from == 0:
                print(f"[Pieces] build stopped at step {k + 1}: no design with {unlock - 1} piece(s) unlocked "
                      f"(max_backtrack {cfg['max_backtrack']})", flush=True)
                return None
            continue
        ph = res['solve_info']['phases'][0]
        t_budget = t_step
        while (ph.get('mip_gap') or 0.0) > float(cfg['lock_gap']) and t_budget < float(cfg['piece_time_max_s']):
            t_budget = min(2 * t_budget, float(cfg['piece_time_max_s']))
            print(f"[Pieces] step {k + 1}: gap {100 * ph['mip_gap']:.1f} % > {100 * float(cfg['lock_gap']):.0f} %: not locked, "
                  f"re-solving with {t_budget:.0f} s", flush=True)
            again = opt.optimize(sub, locations, sens, acts, cable_types, comm, partitions=partitions, hardwares=hardwares,
                                 interfaces=interfaces, time_limit=t_budget, mip_gap=float(cfg['piece_gap']),
                                 uncertainty=uncertainty, backbone=backbone,
                                 phase2_time=(phase2_time if last else float(cfg['piece_phase2_s'])),
                                 phase2_cap=(None if last else float(cfg['piece_phase2_s'])),
                                 gurobi_params=step_params, chains=chs, start=opt.last_values, fixed=fixed, lns=None)
            if again is None:
                break
            if again['total_cost'] <= res['total_cost'] + 1e-6:
                res, ph = again, again['solve_info']['phases'][0]
        if (ph.get('mip_gap') or 0.0) > float(cfg['lock_gap']) and len(pieces[k]) >= 2:
            half = len(pieces[k]) // 2
            print(f"[Pieces] step {k + 1}: gap {100 * ph['mip_gap']:.1f} % after {t_budget:.0f} s: group of {len(pieces[k])} "
                  f"split into {half} + {len(pieces[k]) - half}", flush=True)
            pieces[k:k + 1] = [pieces[k][:half], pieces[k][half:]]
            steps.append({**step, 'result': 'split', 'gap': ph.get('mip_gap'), 'seconds': round(time.time() - t1, 1)})
            continue
        sol = res
        cfg = {**cfg, '_t_scale': 1.0}
        placed_x = dict(sol['assignment'])
        placed_r = {s: v['cell'] for s, v in (sol.get('standby') or {}).items()}
        step['seconds'] = round(time.time() - t1, 1)
        if cfg.get('dump_dir'):
            os.makedirs(cfg['dump_dir'], exist_ok=True)
            with open(os.path.join(cfg['dump_dir'], f"pieces_step_{k + 1}.json"), 'w') as f:
                json.dump({'step': k + 1, 'pieces': pieces, 'placed_x': placed_x, 'placed_r': placed_r,
                           'cost': sol.get('total_cost')}, f, indent=1)
        step.update({'result': ph['status'], 'cost': sol.get('total_cost'), 'gap': ph.get('mip_gap'),
                     'cells': len(set(placed_x.values()))})
        steps.append(step)
        step['memory_gb'] = [round(v, 1) for v in process_memory_gb()]
        print(f"[Pieces] step {k + 1} done in {step['seconds']:.0f} s: cost {step['cost']:.2f}, "
              f"gap {100 * (ph.get('mip_gap') or 0):.2f} %, {step['cells']} cells used; memory {step['memory_gb'][0]:.1f} GB now, "
              f"{step['memory_gb'][1]:.1f} GB peak", flush=True)
        k += 1
        unlock = 0
    build_s = round(time.time() - t0, 1)
    print(f"\n[Pieces] complete design: cost {sol.get('total_cost'):.2f}, {len(pieces)} pieces, "
          f"{sum(1 for s in steps if s['result'] == 'no design')} rejected steps, {sum(1 for s in steps if s['result'] == 'split')} splits, "
          f"{build_s:.0f} s in total")
    polish = []
    gw_conv = None
    if backbone_full is not backbone:
        sol, gw_conv = _gateway_convert(opt, scs, locations, sensors, actuators, cable_types, comm_matrix, partitions=partitions,
                                        hardwares=hardwares, interfaces=interfaces, backbone=backbone_full, uncertainty=uncertainty,
                                        chains=chains, gurobi_params=gurobi_params, cfg=cfg, sol=sol)
    backbone = backbone_full
    if int(cfg.get('polish_rounds', 0)) > 0:
        sol, polish = polish_fn(cfg)(opt, scs, locations, sensors, actuators, cable_types, comm_matrix, partitions=partitions,
                                     hardwares=hardwares, interfaces=interfaces, backbone=backbone, uncertainty=uncertainty,
                                     chains=chains, gurobi_params=gurobi_params, cfg=cfg, pieces=pieces, steps=steps, sol=sol,
                                     traffic=traffic)
    sol, merge, final = merge_and_final(opt, scs, locations, sensors, actuators, cable_types, comm_matrix, partitions=partitions,
                                        hardwares=hardwares, interfaces=interfaces, backbone=backbone, uncertainty=uncertainty,
                                        chains=chains, gurobi_params=gurobi_params, cfg=cfg, sol=sol, pieces=pieces)
    sol['solve_info']['pieces'] = {'sizes': [len(p) for p in pieces], 'steps': steps, 'build_s': build_s, 'polish': polish,
                                   'gateway_convert': gw_conv, 'merge': merge, 'final_network': final,
                                   'total_s': round(time.time() - t0, 1), 'settings': {k: v for k, v in cfg.items() if not k.startswith('_')}}
    _report(steps, polish, pieces, sol, cfg, final, merge)
    _dump_values(cfg, opt)
    return sol


def _dump_values(cfg, opt, name='values.json'):
    if not cfg.get('dump_dir') or not opt.last_values:
        return
    os.makedirs(cfg['dump_dir'], exist_ok=True)
    path = os.path.join(cfg['dump_dir'], name)
    with open(path + '.tmp', 'w') as f:
        json.dump(opt.last_values, f)
    os.replace(path + '.tmp', path)


def _gateway_convert(opt, scs, locations, sensors, actuators, cable_types, comm_matrix, *, partitions, hardwares, interfaces,
                     backbone, uncertainty, chains, gurobi_params, cfg, sol):
    t0 = time.time()
    before = float(sol['total_cost'])
    fixed = {'x': dict(sol['assignment']), 'r': {s: v['cell'] for s, v in (sol.get('standby') or {}).items()}}
    limit = float(cfg.get('gateway_convert_s') or cfg.get('final_network_s') or 1800)
    print(f"\n[Pieces] gateway step: every placement fixed, the network free, the bus gateway on, {limit:.0f} s", flush=True)
    start = dict(opt.last_values or {})
    res = opt.optimize(scs, locations, sensors, actuators, cable_types, comm_matrix, partitions=partitions, hardwares=hardwares,
                       interfaces=interfaces, time_limit=limit, mip_gap=float(cfg['piece_gap']),
                       uncertainty=uncertainty, backbone=backbone, phase2_time=float(cfg['piece_phase2_s']),
                       phase2_cap=float(cfg['piece_phase2_s']),
                       gurobi_params=','.join(x for x in (gurobi_params, cfg.get('step_gurobi_params')) if x),
                       chains=chains, start=start, fixed=fixed, lns=None, reuse=True)
    ph = (((res or {}).get('solve_info') or {}).get('phases') or [{}])[0]
    if res is not None:
        sol = res
    else:
        opt.last_values = start
    entry = {'before': before, 'after': float(sol['total_cost']), 'seconds': round(time.time() - t0, 1),
             'result': 'design with gateways' if res is not None else 'no design', 'status': ph.get('status'),
             'gap': ph.get('mip_gap'), 'gateways': len(sol.get('bus_gateways') or {})}
    print(f"[Pieces] gateway step: {before:.2f} -> {entry['after']:.2f} ({entry['result']}, {entry['gateways']} devices with "
          f"gateways; {entry['status']}, {entry['seconds']:.0f} s)", flush=True)
    if res is not None:
        _dump_values(cfg, opt)
    return sol, entry


def _final_network(opt, scs, locations, sensors, actuators, cable_types, comm_matrix, *, partitions, hardwares, interfaces,
                   backbone, uncertainty, chains, gurobi_params, cfg, sol):
    t0 = time.time()
    before, before_len = float(sol['total_cost']), float(sol.get('cable_length') or 0.0)
    fixed = {'x': dict(sol['assignment']), 'r': {s: v['cell'] for s, v in (sol.get('standby') or {}).items()}}
    gap = float(cfg.get('final_network_gap', 0.0) or 0.0)
    print(f"\n[Pieces] final network clean-up: every placement fixed, the network free, gap {100 * gap:.2f} %, "
          f"{float(cfg['final_network_s']):.0f} s", flush=True)
    start = dict(opt.last_values or {})
    res = opt.optimize(scs, locations, sensors, actuators, cable_types, comm_matrix, partitions=partitions, hardwares=hardwares,
                       interfaces=interfaces, time_limit=float(cfg['final_network_s']), mip_gap=gap,
                       uncertainty=uncertainty, backbone=backbone, phase2_time=float(cfg['piece_phase2_s']),
                       phase2_cap=float(cfg['piece_phase2_s']),
                       gurobi_params=','.join(x for x in (gurobi_params, cfg.get('step_gurobi_params')) if x),
                       chains=chains, start=start, fixed=fixed, lns=None, reuse=True)
    ph = (((res or {}).get('solve_info') or {}).get('phases') or [{}])[0]
    better = res is not None and (float(res['total_cost']) < before - 1e-6 or (float(res['total_cost']) < before + 1e-6 and
                                                                                float(res.get('cable_length') or 0.0) < before_len - 1e-6))
    if better:
        sol = res
    else:
        opt.last_values = start
    switches = sorted(str(h).split('@')[1] for h in sol.get('hw_features', []) if str(h).startswith('SWITCH@'))
    entry = {'before': before, 'after': float(sol['total_cost']), 'seconds': round(time.time() - t0, 1),
             'result': 'improved' if better else ('no gain' if res is not None else 'no design'),
             'status': ph.get('status'), 'gap': ph.get('mip_gap'), 'switches': switches}
    entry['memory_gb'] = [round(v, 1) for v in process_memory_gb()]
    print(f"[Pieces] final network: {before:.2f} -> {entry['after']:.2f} ({entry['result']}; {entry['status']}, gap "
          f"{100 * (entry['gap'] or 0):.3f} %, {entry['seconds']:.0f} s); switches at {switches}", flush=True)
    return sol, entry


def merge_and_final(opt, scs, locations, sensors, actuators, cable_types, comm_matrix, *, partitions, hardwares, interfaces,
                    backbone, uncertainty, chains, gurobi_params, cfg, sol, pieces=None):
    kw = dict(partitions=partitions, hardwares=hardwares, interfaces=interfaces, backbone=backbone, uncertainty=uncertainty,
              chains=chains, gurobi_params=gurobi_params, cfg=cfg)
    merge, final = [], None
    before, before_values = sol, dict(opt.last_values or {})
    if int(cfg.get('merge_rounds', 0) or 0) > 0:
        sol, merge = _merge_step(opt, scs, locations, sensors, actuators, cable_types, comm_matrix, sol=sol, pieces=pieces, **kw)
    if float(cfg.get('final_network_s', 0) or 0) > 0:
        sol, final = _final_network(opt, scs, locations, sensors, actuators, cable_types, comm_matrix, sol=sol, **kw)
    if sol is not before and float(before['total_cost']) < float(sol['total_cost']) - 1e-6:
        print(f"[Pieces] merge step: the merged design ({float(sol['total_cost']):.2f}) is dearer than the design before the "
              f"merge ({float(before['total_cost']):.2f}) after the clean-up: the design before the merge is kept", flush=True)
        merge.append({'round': None, 'close': None, 'result': 'dropped after the clean-up', 'after': float(sol['total_cost']),
                      'before': float(before['total_cost'])})
        sol, opt.last_values = before, before_values
    return sol, merge, final


def reach_cells(scs, locations, sensors, actuators, cable_types):
    max_len = {n: float(getattr(ct, 'max_length_m', float('inf')) or float('inf')) for n, ct in (cable_types or {}).items()}
    shared = {n for n in ('CAN', 'LIN', 'FLEXRAY') if n in (cable_types or {})}
    s_look, a_look = optimizer_utils.build_sensor_lookup(sensors), optimizer_utils.build_actuator_lookup(actuators)
    out = {}
    for sc in scs:
        devs = [s_look.get(d) for d in (getattr(sc, 'sensors', None) or [])] + [a_look.get(d) for d in (getattr(sc, 'actuators', None) or [])]
        devs = [d for d in devs if d is not None and d.location and d.interface != 'ETH' and d.interface not in shared]
        ok = getattr(sc, 'allowed_cells', None)
        out[sc.id] = {L.id for L in locations if (not ok or L.id in ok) and all(
            optimizer_utils.get_distance(d.location, L.location) <= max_len.get(d.interface, float('inf')) + 1e-9 for d in devs)}
    return out


def redundancy_partners(scs):
    ids = {sc.id for sc in scs}
    out = defaultdict(set)
    for sc in scs:
        p = getattr(sc, 'redundant_with', None)
        if p and p in ids:
            out[sc.id].add(p)
            out[p].add(sc.id)
    return out


def merge_moves(placed_x, placed_r, close, reach, partners, target=None):
    size = defaultdict(int)
    for c in list(placed_x.values()) + list(placed_r.values()):
        size[c] += 1
    order = sorted((c for c in size if c != close), key=lambda c: (-size[c], c))
    if target is not None:
        order = [target] + [c for c in order if c != target]
    x, r = dict(placed_x), dict(placed_r)
    moves = []
    for sid in sorted(s for s, c in placed_x.items() if c == close):
        bad = {r.get(sid)} | {x.get(p) for p in partners.get(sid, ())}
        t = next((c for c in order if c in reach.get(sid, set()) and c not in bad), None)
        if t is None:
            return None, None, f"primary {sid} has no cell within reach"
        x[sid] = t
        moves.append(('primary', sid, t))
    for sid in sorted(s for s, c in placed_r.items() if c == close):
        t = next((c for c in order if c in reach.get(sid, set()) and c != x[sid]), None)
        if t is None:
            return None, None, f"standby {sid} has no cell within reach"
        r[sid] = t
        moves.append(('standby', sid, t))
    return x, r, moves


def _merge_step(opt, scs, locations, sensors, actuators, cable_types, comm_matrix, *, partitions, hardwares, interfaces,
                backbone, uncertainty, chains, gurobi_params, cfg, sol, pieces=None):
    t0 = time.time()
    common = dict(partitions=partitions, hardwares=hardwares, interfaces=interfaces, uncertainty=uncertainty, chains=chains)
    bb = {**backbone, 'bus_gateway': False} if (backbone or {}).get('bus_gateway') and not cfg.get('gateway_in_build', True) else backbone
    reach = reach_cells(scs, locations, sensors, actuators, cable_types)
    partners = redundancy_partners(scs) if (uncertainty or {}).get('redundancy') else {}
    given, given_values = sol, dict(opt.last_values or {})

    def rebuild(x, r):
        res = opt.optimize(scs, locations, sensors, actuators, cable_types, comm_matrix, time_limit=float(cfg['merge_time_s']),
                           mip_gap=float(cfg['piece_gap']), phase2_time=float(cfg['piece_phase2_s']), phase2_cap=float(cfg['piece_phase2_s']),
                           gurobi_params=','.join(v for v in (gurobi_params, cfg.get('step_gurobi_params')) if v),
                           fixed={'x': x, 'r': r}, lns=None, reuse=True, backbone=bb, **common)
        return res, dict(opt.last_values or {})

    x = dict(sol['assignment'])
    r = {s: v['cell'] for s, v in (sol.get('standby') or {}).items()}
    print(f"\n[Pieces] merge step: up to {cfg['merge_rounds']} round(s), every used ECU tried ({cfg.get('merge_targets', 'every')} targets), {float(cfg['merge_time_s']):.0f} s per rebuild", flush=True)
    ref, ref_values = rebuild(x, r)
    entries = []
    if ref is None:
        print("[Pieces] merge step: the current design could not be rebuilt, no merge", flush=True)
        opt.last_values = given_values
        return given, [{'round': 0, 'close': None, 'result': 'no reference', 'seconds': round(time.time() - t0, 1)}]
    ref_cost = float(ref['total_cost'])
    print(f"[Pieces] merge step: reference (the current design rebuilt) {ref_cost:.2f}", flush=True)
    kept = False
    for rnd in range(1, int(cfg['merge_rounds']) + 1):
        size = defaultdict(int)
        for c in list(x.values()) + list(r.values()):
            size[c] += 1
        tries = []
        every = cfg.get('merge_targets', 'every') == 'every'
        jobs = []
        for close in sorted(size, key=lambda c: (size[c], c)):
            if not every:
                jobs.append((close, None))
                continue
            if merge_moves(x, r, close, reach, partners)[0] is None:
                jobs.append((close, None))
                continue
            jobs += [(close, tgt) for tgt in sorted(size, key=lambda c: (-size[c], c)) if tgt != close]
        seen = set()
        for close, tgt in jobs:
            t1 = time.time()
            nx, nr, moves = merge_moves(x, r, close, reach, partners, target=tgt)
            entry = {'round': rnd, 'close': close, 'target': tgt, 'components': size[close], 'before': ref_cost}
            if nx is not None:
                key = (close, tuple(sorted(nx.items())), tuple(sorted(nr.items())))
                if key in seen:
                    continue
                seen.add(key)
            if nx is None:
                entry.update(result='skipped', reason=moves, seconds=0.0)
                print(f"[Pieces] merge round {rnd}: {close} ({size[close]} components) skipped: {moves}", flush=True)
                entries.append(entry)
                continue
            res, vals = rebuild(nx, nr)
            entry.update(moves=len(moves), seconds=round(time.time() - t1, 1),
                         after=float(res['total_cost']) if res is not None else None,
                         result='design' if res is not None else 'no design')
            print(f"[Pieces] merge round {rnd}: {close} ({size[close]} components" + (f", into {tgt} first" if tgt else "") + f", {len(moves)} moves) -> "
                  f"{entry['after'] if entry['after'] is not None else 'no design'} (reference {ref_cost:.2f}, {entry['seconds']:.0f} s)", flush=True)
            entries.append(entry)
            if res is not None:
                tries.append({'cost': float(res['total_cost']), 'close': close, 'x': nx, 'r': nr, 'sol': res, 'values': vals, 'entry': entry})
        best = min(tries, key=lambda t: t['cost']) if tries else None
        if best is None or best['cost'] >= ref_cost - 1e-6:
            print(f"[Pieces] merge round {rnd}: no try beats {ref_cost:.2f}, the merge step ends", flush=True)
            break
        best['entry']['result'] = 'kept'
        print(f"[Pieces] merge round {rnd}: kept: {best['close']} emptied" + (f" into {best['entry']['target']} first" if best['entry'].get('target') else "")
              + f", {ref_cost:.2f} -> {best['cost']:.2f}", flush=True)
        x, r, ref, ref_values, ref_cost, kept = best['x'], best['r'], best['sol'], best['values'], best['cost'], True
        if cfg.get('dump_dir'):
            _dump_polish(cfg, pieces or [], best['sol'], {'kind': 'merge', 'round': rnd, 'close': best['close'],
                                                         'target': best['entry'].get('target'), 'before': best['entry']['before'],
                                                         'after': best['cost']}, values=best['values'], name='merge')
    print(f"[Pieces] merge step: {'kept ' + format(ref_cost, '.2f') if kept else 'no merge kept'} ({time.time() - t0:.0f} s)", flush=True)
    if not kept:
        opt.last_values = given_values
        return given, entries
    opt.last_values = ref_values
    return ref, entries


def _report(steps, polish, pieces, sol, cfg, final=None, merge=None):
    lines = [f"{'step':>4} {'new':>4} {'free':>4} {'lock':>5} {'failop':>6} {'Mbit/s':>8} {'dev':>3} {'chn':>3} {'kDMIPS':>6} {'GB':>5} "
             f"{'s':>6} {'gap %':>6} {'cost':>10} {'cells':>5}  result"]
    for st in steps:
        gap = st.get('gap'); cost = st.get('cost')
        lines.append(f"{st['step']:>4} {st['new']:>4} {st['free']:>4} {st['locked']:>5} {st.get('fail_op', 0):>6} {st.get('traffic_mbit', 0):>8.1f} "
                     f"{st.get('devices', 0):>3} {st.get('chains', 0):>3} {st.get('cpu', 0) / 1000:>6.1f} {st.get('ram_kb', 0) / 1048576:>5.1f} "
                     f"{st['seconds']:>6.0f} {(100 * gap if gap is not None else float('nan')):>6.1f} "
                     f"{(cost if cost is not None else float('nan')):>10.1f} {st.get('cells', '-'):>5}  {st['result']}")
    for r in polish:
        lines.append(f"polish {r['round']} (pass {r.get('pass', 1)}): {r.get('kind') or ('cell' if isinstance(r['group'], str) else 'group')} {r['group']} (+{r['partners']} partners, {r['free']} free): "
                     f"{r['before']:.1f} -> {r['after']:.1f} in {r['seconds']:.0f} s ({r['result']})")
    for m in (merge or []):
        lines.append(f"merge round {m.get('round')}: {m.get('close')}" + (f" into {m['target']} first" if m.get('target') else "")
                     + f" ({m.get('components', '-')} components): {m.get('before')} -> {m.get('after')} "
                     f"({m['result']}{', ' + m['reason'] if m.get('reason') else ''})")
    if final:
        lines.append(f"final network: {final['before']:.1f} -> {final['after']:.1f} in {final['seconds']:.0f} s ({final['result']}; "
                     f"{final['status']}, gap {100 * (final['gap'] or 0):.3f} %), switches at {final['switches']}")
    lines.append(f"final cost {sol.get('total_cost'):.2f}; groups {[len(p) for p in pieces]}")
    text = "\n".join(lines)
    print("\n[Pieces] build report\n" + text, flush=True)
    if cfg.get('dump_dir'):
        os.makedirs(cfg['dump_dir'], exist_ok=True)
        with open(os.path.join(cfg['dump_dir'], 'pieces_report.txt'), 'w') as f:
            f.write(text + "\n")


def _polish_cells_parallel(opt, scs, locations, sensors, actuators, cable_types, comm_matrix, *, partitions, hardwares,
                           interfaces, backbone, uncertainty, chains, gurobi_params, cfg, pieces, steps, sol, traffic):
    import concurrent.futures as cf
    import multiprocessing as mp
    par = int(cfg.get('polish_parallel', 3))
    max_free = int(cfg.get('polish_max_free', 120))
    threads = None
    for item in (gurobi_params or '').split(','):
        if item.strip().lower().startswith('threads='):
            threads = int(item.split('=')[1])
    per_job = max(1, (threads or os.cpu_count() or par) // par)
    params = ','.join(x for x in (gurobi_params, cfg.get('step_gurobi_params'), f'Threads={per_job}') if x)
    log_dir = cfg.get('dump_dir') or '.'
    os.makedirs(log_dir, exist_ok=True)
    best_values = dict(opt.last_values or {})
    tried, rounds, n_tried, batch = set(), [], 0, 0
    budget = int(cfg['polish_rounds'])
    ctx = mp.get_context('spawn')
    mode, kind = cfg.get('polish_mode', 'cell'), ('level' if cfg.get('polish_mode', 'cell') == 'level' else 'cell')
    while n_tried < budget:
        placed_x = dict(sol['assignment'])
        placed_r = {s: v['cell'] for s, v in (sol.get('standby') or {}).items()}
        cands = _free_sets(mode, sol, scs, cfg, tried, partitions)[:min(par, budget - n_tried)]
        if not cands:
            print(f"[Pieces] parallel {kind} polish: no {kind} left with at most {max_free} components", flush=True)
            break
        batch += 1
        jobs = []
        for n_free, cell, free in cands:
            fixed = {'x': {s: c for s, c in placed_x.items() if s not in free}, 'r': {s: c for s, c in placed_r.items() if s not in free}}
            jobs.append({'cell': cell, 'n_free': n_free, 'distance_table': (list(optimizer_utils._DISTANCE_TABLE[0]), optimizer_utils._DISTANCE_TABLE[1]) if optimizer_utils._DISTANCE_TABLE else None,
                         'steiner_cache': dict(optimizer_utils._STEINER_CACHE),
                         'log': os.path.join(log_dir, f"polish_batch{batch}_{cell}.log"),
                         'args': (scs, locations, sensors, actuators, cable_types, comm_matrix),
                         'kwargs': dict(partitions=partitions, hardwares=hardwares, interfaces=interfaces,
                                        time_limit=float(cfg.get('polish_time_s', 600)), mip_gap=float(cfg['piece_gap']),
                                        uncertainty=uncertainty, backbone=backbone, phase2_time=float(cfg['piece_phase2_s']),
                                        phase2_cap=float(cfg['piece_phase2_s']), gurobi_params=params, chains=chains,
                                        start=best_values, fixed=fixed, lns=None)})
        print(f"\n[Pieces] parallel {kind} polish batch {batch}: {[c for _n, c, _f in cands]} ({[n for n, _c, _f in cands]} components freed), "
              f"{len(jobs)} processes x {per_job} threads, {float(cfg.get('polish_time_s', 600)):.0f} s", flush=True)
        t1 = time.time()
        with cf.ProcessPoolExecutor(max_workers=len(jobs), mp_context=ctx) as ex:
            results = list(ex.map(_polish_worker, jobs))
        before = float(sol['total_cost'])
        improving = sorted((float(res['total_cost']), cell, res, vals, sec) for cell, res, vals, sec in results
                           if res is not None and float(res['total_cost']) < before - 1e-6)
        taken = improving[0][1] if improving else None
        for cell, res, vals, sec in results:
            n_tried += 1
            entry = {'round': n_tried, 'kind': kind, 'group': cell, 'partners': 0, 'free': next(n for n, c, _f in cands if c == cell),
                     'before': before, 'seconds': sec, 'batch': batch}
            if res is None:
                entry.update({'after': before, 'result': 'no design'})
                tried.add(cell)
            elif float(res['total_cost']) < before - 1e-6:
                entry.update({'after': float(res['total_cost']),
                              'result': 'improved, taken' + _polish_note(mode, cell, sol, res) if cell == taken else 'improved, not taken'})
                if cell == taken:
                    tried.add(cell)
            else:
                entry.update({'after': before, 'result': 'no gain'})
                tried.add(cell)
            rounds.append(entry)
            print(f"[Pieces] polish batch {batch}, {kind} {cell}: {before:.2f} -> {entry['after']:.2f} ({entry['result']}, {sec:.0f} s)", flush=True)
        if improving:
            _c, cell, res, vals, _s = improving[0]
            note = _polish_note(mode, cell, sol, res)
            sol, best_values = res, vals
            if cfg.get('dump_dir'):
                _dump_polish(cfg, pieces, sol, {'stage': mode, 'batch': batch, 'group': cell}, values=best_values)
            print(f"[Pieces] batch {batch}: kept {kind} {cell}, {before:.2f} -> {float(sol['total_cost']):.2f}{note}, "
                  f"{time.time() - t1:.0f} s", flush=True)
        else:
            print(f"[Pieces] batch {batch}: no improvement, {time.time() - t1:.0f} s", flush=True)
    opt.last_values = best_values
    return sol, rounds


def _polish_cells(opt, scs, locations, sensors, actuators, cable_types, comm_matrix, *, partitions, hardwares,
                  interfaces, backbone, uncertainty, chains, gurobi_params, cfg, pieces, steps, sol, traffic):
    rounds = []
    best_values = dict(opt.last_values or {})
    max_free = int(cfg.get('polish_max_free', 120))
    mode = cfg.get('polish_mode', 'cell')
    kind = mode if mode in ('level', 'gpu') else 'cell'
    max_passes, per_pass = max(1, int(cfg.get('polish_passes', 1))), int(cfg['polish_rounds'])
    for n_pass in range(1, max_passes + 1):
        tried, gain = set(), False
        if bool(cfg.get('polish_all', False)):
            per_pass = max(1, len(_free_sets(mode, sol, scs, cfg, set(), partitions)))
        for r in range(per_pass):
            placed_x = dict(sol['assignment'])
            placed_r = {s: v['cell'] for s, v in (sol.get('standby') or {}).items()}
            cands = _free_sets(mode, sol, scs, cfg, tried, partitions)
            if not cands:
                print(f"[Pieces] {kind} polish: no {kind} left with at most {max_free} components", flush=True)
                break
            n_free, cell, free = cands[0]
            tried.add(cell)
            fixed = {'x': {s: c for s, c in placed_x.items() if s not in free}, 'r': {s: c for s, c in placed_r.items() if s not in free}}
            allow = (cfg.get('polish_allow_cells') or {}).get(cell)
            if allow:
                fixed['allow'] = {s: list(allow) for s in free}
                print(f"[Pieces] {kind} polish: the {n_free} freed components of {cell} may only go to {sorted(allow)}", flush=True)
            pats = [re.compile(q) for q in (cfg.get('polish_fix_patterns') or [])]
            if pats:
                fixed['vars'] = {k: v for k, v in best_values.items() if any(q.match(k) for q in pats)}
                print(f"[Pieces] {kind} polish: {len(fixed['vars'])} values of the kept design frozen by polish_fix_patterns", flush=True)
            if cfg.get('polish_unit_cap'):
                fixed['unit_cap'] = dict(cfg['polish_unit_cap'])
                print(f"[Pieces] {kind} polish: unit cap {fixed['unit_cap']} in this round", flush=True)
            t1 = time.time()
            print(f"\n[Pieces] {kind} polish pass {n_pass}, round {r + 1}/{per_pass}: {kind} {cell} ({n_free} components freed), "
                  f"{float(cfg.get('polish_time_s', 600)):.0f} s", flush=True)
            res = opt.optimize(scs, locations, sensors, actuators, cable_types, comm_matrix, partitions=partitions, hardwares=hardwares,
                               interfaces=interfaces, time_limit=float(cfg.get('polish_time_s', 600)), mip_gap=float(cfg['piece_gap']),
                               uncertainty=uncertainty, backbone=backbone, phase2_time=float(cfg['piece_phase2_s']),
                               phase2_cap=float(cfg['piece_phase2_s']),
                               gurobi_params=','.join(x for x in (gurobi_params, cfg.get('step_gurobi_params')) if x),
                               chains=chains, start=best_values, fixed=fixed, lns=None, reuse=True)
            before = float(sol['total_cost'])
            entry = {'round': len(rounds) + 1, 'pass': n_pass, 'kind': kind, 'group': cell, 'partners': 0, 'free': n_free,
                     'before': before, 'seconds': round(time.time() - t1, 1)}
            if res is not None and float(res['total_cost']) < before - 1e-6:
                note = _polish_note(mode, cell, sol, res)
                sol = res
                best_values = dict(opt.last_values)
                gain = True
                entry.update({'after': float(sol['total_cost']), 'result': 'improved' + note})
                if cfg.get('dump_dir'):
                    _dump_polish(cfg, pieces, sol, {'stage': mode, 'pass': n_pass, 'round': entry['round'], 'group': cell},
                                 values=best_values)
            else:
                entry.update({'after': before, 'result': 'no gain' if res is not None else 'no design'})
            entry['memory_gb'] = [round(v, 1) for v in process_memory_gb()]
            rounds.append(entry)
            print(f"[Pieces] {kind} polish round {entry['round']}: {entry['before']:.2f} -> {entry['after']:.2f} ({entry['result']}, "
                  f"{entry['seconds']:.0f} s); memory {entry['memory_gb'][0]:.1f} GB now, {entry['memory_gb'][1]:.1f} GB peak", flush=True)
        if not gain:
            print(f"[Pieces] {kind} polish pass {n_pass}: no improvement, the polish ends", flush=True)
            break
        if n_pass < max_passes:
            print(f"[Pieces] {kind} polish pass {n_pass} improved the design: pass {n_pass + 1} of at most {max_passes}", flush=True)
    opt.last_values = best_values
    return sol, rounds


def _dump_polish(cfg, pieces, sol, info, values=None, name='polish'):
    os.makedirs(cfg['dump_dir'], exist_ok=True)
    stamp = time.strftime('%Y-%m-%dT%H:%M:%S')
    if values:
        vpath = os.path.join(cfg['dump_dir'], f'values_{name}.json')
        with open(vpath + '.tmp', 'w') as f:
            json.dump({'time': stamp, 'cost': sol.get('total_cost'), 'values': values}, f)
        os.replace(vpath + '.tmp', vpath)
    placed_x = dict(sol['assignment'])
    placed_r = {s: v['cell'] for s, v in (sol.get('standby') or {}).items()}
    path = os.path.join(cfg['dump_dir'], f'pieces_{name}.json')
    with open(path + '.tmp', 'w') as f:
        json.dump({'step': name, 'pieces': pieces, 'placed_x': placed_x, 'placed_r': placed_r, 'cost': sol.get('total_cost'),
                   'cable_length': sol.get('cable_length'), 'polish': {**info, 'time': stamp}},
                  f, indent=1)
    os.replace(path + '.tmp', path)


def _free_sets(mode, sol, scs, cfg, tried, partitions=None):
    import math
    groups = defaultdict(set)
    if mode == 'gpu':
        gpu = {sc.id for sc in scs if float(getattr(sc, 'tops_req', 0) or 0) > 0}
        if gpu:
            groups['GPU'] = gpu
        items = [(len(v), k, v) for k, v in groups.items() if k not in tried]
        return items
    if mode == 'level':
        for sc in scs:
            groups[f"ASIL {int(getattr(sc, 'asil_req', 0) or 0)}"].add(sc.id)
    else:
        for s, c in sol['assignment'].items():
            groups[c].add(s)
        for s, v in (sol.get('standby') or {}).items():
            groups[v['cell']].add(s)
    max_free = int(cfg.get('polish_max_free', 120))
    split_cells = bool(cfg.get('polish_split_cells', False))
    if mode == 'level' or split_cells:
        by_id = {sc.id: sc for sc in scs}
        for key in [k for k, v in groups.items() if len(v) > max_free]:
            members = groups.pop(key)
            by_sub = defaultdict(set)
            for sc in members:
                sub = (sol['assignment'].get(sc, '?') if mode == 'level'
                       else f"ASIL {int(getattr(by_id.get(sc), 'asil_req', 0) or 0)}")
                by_sub[sub].add(sc)
            n_parts = max(2, math.ceil(len(members) / max_free))
            parts = [set() for _ in range(n_parts)]
            for sub_members in sorted(by_sub.values(), key=len, reverse=True):
                min(parts, key=len).update(sub_members)
            final_parts = []
            for part in parts:
                n_sub = math.ceil(len(part) / max_free)
                ordered = sorted(part)
                final_parts += [set(ordered[j::n_sub]) for j in range(n_sub)] if n_sub > 1 else [part]
            for i, part in enumerate(final_parts, 1):
                groups[f"{key} part {i}/{len(final_parts)}"] = part
            print(f"[Pieces] {mode} polish: {key} ({len(members)} components) above polish_max_free {max_free}: "
                  f"{len(final_parts)} parts of {[len(q) for q in final_parts]} by {'cell' if mode == 'level' else 'level'}", flush=True)
    items = [(len(v), k, v) for k, v in groups.items() if k not in tried and len(v) <= max_free]
    order = str(cfg.get('polish_order', 'fewest') or 'fewest')
    if mode == 'level' and order == 'partitions':
        used = defaultdict(int)
        for key, n in (sol.get('partition_counts') or {}).items():
            used['ASIL ' + key.rsplit('_asil', 1)[1]] += int(n)
        return sorted(items, key=lambda t: (-used.get(' '.join(t[1].split()[:2]), 0), t[0], t[1]))
    if mode == 'level' and order == 'waste':
        by_id = {sc.id: sc for sc in scs}
        caps = {r: float((partitions or {}).get(f'{r}_cap', 0) or 0) for r in ('cpu', 'ram', 'rom')}
        used = defaultdict(int)
        for key, n in (sol.get('partition_counts') or {}).items():
            used['ASIL ' + key.rsplit('_asil', 1)[1]] += int(n)

        def need(members):
            n = 1
            for r, cap in caps.items():
                if cap > 0:
                    n = max(n, math.ceil(sum(float(getattr(by_id[s], f'{r}_req', 0) or 0) for s in members) / cap - 1e-9))
            return n
        return sorted(items, key=lambda t: (-(used.get(t[1], 0) - need(t[2])), t[0], t[1]))
    return sorted(items, key=lambda t: ((-t[0] if order == 'most' else t[0]), t[1]))


def _gpu_module_cap(cfg, sol):
    rho = (sol.get('solve_info') or {}).get('rho')
    key = (cfg.get('config_dir'), cfg.get('architecture_json'), rho)
    if key in _GPU_CAP:
        return _GPU_CAP[key]
    cap = {}
    if not (cfg.get('config_dir') and cfg.get('architecture_json')):
        print("[Pieces] gpu round: no config dir / architecture in the settings, so no module cap", flush=True)
        return cap
    out_dir = cfg.get('dump_dir') or os.path.dirname(os.path.abspath(cfg['architecture_json']))
    import sys
    tools = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools')
    if tools not in sys.path:
        sys.path.insert(0, tools)
    try:
        import design_bound
        from check_solution import SolutionCheck
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, 'gpu_cap_design.json')
        with open(path, 'w') as f:
            json.dump(sol, f, default=str)
        c = SolutionCheck(path, cfg['config_dir'], cfg['architecture_json'])
        c.load()
        c.parse()
        res = design_bound.gpu_module_floor(c, design_bound.design_bound(c), design_bound.bound_settings(c))
        for feat, r_ in res.items():
            cap[feat] = int(sum(r_['mix'].values()))
            print(f"[Pieces] gpu round: the bound's cheapest module set for {feat} is "
                  f"{', '.join(f'{n} {t}' for t, n in sorted(r_['mix'].items()))} ({r_['ub']:.2f}), so at most {cap[feat]} "
                  f"modules in the car{'' if r_['optimal'] else ' (module model at its time limit)'}", flush=True)
    except Exception as e:
        print(f"[Pieces] gpu round: module cap not computed ({type(e).__name__}: {e}); the round runs without a cap", flush=True)
        cap = {}
    _GPU_CAP[key] = cap
    return cap


def _gpu_round_cfg(cfg, sol):
    if str(cfg.get('polish_gpu', 'capped')).lower() != 'capped':
        return {}
    cap = _gpu_module_cap(cfg, sol)
    cells = sorted({k.split('@', 1)[1] for k, mix in (sol.get('hw_unit_types') or {}).items()
                    if '@' in k and k.split('@', 1)[0] in cap and sum((mix or {}).values()) > 0})
    allow = dict(cfg.get('polish_allow_cells') or {})
    if cells and not allow.get('GPU'):
        allow['GPU'] = cells
    return {'polish_time_s': float(cfg.get('polish_gpu_time_s', 3600)),
            'polish_fix_patterns': list(cfg.get('polish_gpu_fix_patterns') or GPU_ROUND_FIX),
            'polish_allow_cells': allow, 'polish_unit_cap': cap}


def _polish_stages(opt, scs, locations, sensors, actuators, cable_types, comm_matrix, *, partitions, hardwares, interfaces,
                   backbone, uncertainty, chains, gurobi_params, cfg, pieces, steps, sol, traffic):
    rounds = []
    stages = list(cfg.get('polish_stages') or [cfg.get('polish_mode', 'cell')])
    max_passes = max(1, int(cfg.get('polish_passes', 1)))
    for n_pass in range(1, max_passes + 1):
        before = float(sol['total_cost'])
        for stage in stages:
            print(f"\n[Pieces] polish pass {n_pass} of at most {max_passes}, stage {stage}", flush=True)
            stage_cfg = {**cfg, 'polish_mode': stage, 'polish_passes': 1, 'polish_parallel': 1}
            if stage == 'gpu':
                stage_cfg.update(_gpu_round_cfg(cfg, sol))
            sol, rs = _polish_cells(opt, scs, locations, sensors, actuators, cable_types, comm_matrix, partitions=partitions,
                                    hardwares=hardwares, interfaces=interfaces, backbone=backbone, uncertainty=uncertainty,
                                    chains=chains, gurobi_params=gurobi_params, pieces=pieces, steps=steps, sol=sol, traffic=traffic,
                                    cfg=stage_cfg)
            for r in rs:
                r.update({'round': len(rounds) + 1, 'pass': n_pass, 'stage': stage})
                rounds.append(r)
        after, min_gain = float(sol['total_cost']), float(cfg.get('polish_min_gain', 0.0) or 0.0)
        if after >= before - 1e-6:
            print(f"[Pieces] polish pass {n_pass}: no stage improved the design, the polish ends", flush=True)
            break
        if min_gain > 0 and before - after < min_gain * before:
            print(f"[Pieces] polish pass {n_pass}: gain {before:.2f} -> {after:.2f} below polish_min_gain {100 * min_gain:.2f} %, "
                  f"the polish ends", flush=True)
            break
        if n_pass < max_passes:
            print(f"[Pieces] polish pass {n_pass} improved the design {before:.2f} -> {float(sol['total_cost']):.2f}: "
                  f"pass {n_pass + 1} of at most {max_passes}", flush=True)
    return sol, rounds


def _polish_note(mode, key, old, new):
    if mode == 'gpu':
        def mods(sol):
            return ', '.join(f"{k.split('@')[1]} {'+'.join(f'{n}x{t}' for t, n in v.items())}"
                             for k, v in sorted((sol.get('hw_unit_types') or {}).items()))
        return f", GPU modules {mods(old)} -> {mods(new)}"
    if mode == 'level':
        a = key.split()[1]
        return (f", partitions {_level_partitions(old, a)} -> {_level_partitions(new, a)} of the level, "
                f"{int(old.get('num_partitions', 0))} -> {int(new.get('num_partitions', 0))} in total")
    used = set(new['assignment'].values()) | {v['cell'] for v in (new.get('standby') or {}).values()}
    return ', cell closed' if key.split(' part ')[0] not in used else ''


def _level_partitions(sol, a):
    return sum(int(n) for k, n in (sol.get('partition_counts') or {}).items() if k.endswith(f'_asil{a}'))


def polish_fn(cfg):
    if cfg.get('polish_stages'):
        return _polish_stages
    if cfg.get('polish_mode', 'cell') in ('cell', 'level'):
        return _polish_cells_parallel if int(cfg.get('polish_parallel', 1)) > 1 else _polish_cells
    return _polish


def _polish(opt, scs, locations, sensors, actuators, cable_types, comm_matrix, *, partitions, hardwares, interfaces,
            backbone, uncertainty, chains, gurobi_params, cfg, pieces, steps, sol, traffic):
    gap_of = {}
    for st in steps:
        if st['result'] not in ('no design', 'split') and st.get('gap') is not None:
            gap_of[st['step'] - 1] = st['gap']
    order = sorted(range(len(pieces)), key=lambda k: -gap_of.get(k, 0.0))
    n_partners = int(cfg.get('polish_partners', 10))
    pair = defaultdict(float)
    for lk in comm_matrix:
        pair[lk['src'], lk['dst']] += float(lk.get('volume', 0.0))
        pair[lk['dst'], lk['src']] += float(lk.get('volume', 0.0))
    nbr = defaultdict(set)
    for a, b in pair:
        nbr[a].add(b)
    best_values = dict(opt.last_values or {})
    placed_x = dict(sol['assignment'])
    placed_r = {s: v['cell'] for s, v in (sol.get('standby') or {}).items()}
    rounds = []
    for r in range(int(cfg['polish_rounds'])):
        g = order[r % len(order)]
        group = set(pieces[g])
        partners = sorted(((sum(pair[s, t] for s in group if (s, t) in pair), t) for t in {t for s in group for t in nbr[s]} - group), reverse=True)
        free = group | {t for _v, t in partners[:n_partners]}
        fixed = {'x': {s: c for s, c in placed_x.items() if s not in free}, 'r': {s: c for s, c in placed_r.items() if s not in free}}
        t1 = time.time()
        print(f"\n[Pieces] polish round {r + 1}/{cfg['polish_rounds']}: group {g + 1} (build gap {100 * gap_of.get(g, 0):.1f} %), "
              f"{len(free)} free ({len(free) - len(group)} partners), {float(cfg.get('polish_time_s', 600)):.0f} s", flush=True)
        res = opt.optimize(scs, locations, sensors, actuators, cable_types, comm_matrix, partitions=partitions, hardwares=hardwares,
                           interfaces=interfaces, time_limit=float(cfg.get('polish_time_s', 600)), mip_gap=float(cfg['piece_gap']),
                           uncertainty=uncertainty, backbone=backbone, phase2_time=float(cfg['piece_phase2_s']),
                           gurobi_params=gurobi_params, chains=chains, start=best_values, fixed=fixed, lns=None)
        before = float(sol['total_cost'])
        entry = {'round': r + 1, 'group': g + 1, 'partners': len(free) - len(group), 'free': len(free), 'before': before,
                 'seconds': round(time.time() - t1, 1)}
        if res is not None and float(res['total_cost']) < before - 1e-6:
            sol = res
            best_values = dict(opt.last_values)
            placed_x = dict(sol['assignment'])
            placed_r = {s: v['cell'] for s, v in (sol.get('standby') or {}).items()}
            entry.update({'after': float(sol['total_cost']), 'result': 'improved'})
        else:
            entry.update({'after': before, 'result': 'no gain' if res is not None else 'no design'})
        rounds.append(entry)
        print(f"[Pieces] polish round {r + 1}: {entry['before']:.2f} -> {entry['after']:.2f} ({entry['result']}, {entry['seconds']:.0f} s)", flush=True)
    return sol, rounds

_DISTANCE_TABLE = None


def set_distance_table(ids, matrix):
    global _DISTANCE_TABLE
    index = {i: k for k, i in enumerate(ids)}
    _DISTANCE_TABLE = (index, matrix)


def get_distance(loc1, loc2):
    if loc1 is None or loc2 is None:
        return 0.0
    if _DISTANCE_TABLE is not None:
        index, matrix = _DISTANCE_TABLE
        a, b = getattr(loc1, 'ref', None), getattr(loc2, 'ref', None)
        if a in index and b in index:
            return matrix[index[a]][index[b]]
    manhattan_dist, _ = loc1.dist(loc2)
    return manhattan_dist

_STEINER_CACHE = {}
STEINER_DP_MAX = 9


def _steiner_flow(D, T):
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
    n, k = len(D), len(T)
    DT = D[np.ix_(T, T)]
    B = np.zeros((k, k))
    best, par, done, order = DT[0].copy(), np.zeros(k, dtype=np.int64), np.zeros(k, dtype=bool), [0]
    done[0] = True
    for _ in range(k - 1):
        x = int(np.where(done, np.inf, best).argmin())
        B[x, order] = B[order, x] = np.maximum(B[par[x], order], best[x])
        order.append(x)
        done[x] = True
        closer = DT[x] < best
        best, par = np.where(closer, DT[x], best), np.where(closer, x, par)
    DnT = D[:, T]
    via = np.maximum(DnT[:, :, None], B[None]).min(axis=1)
    S = np.maximum(via[:, None, :], DnT[None]).min(axis=2)
    S = np.minimum(S, S.T)
    iu, iv = np.triu_indices(n, 1)
    keep = D[iu, iv] <= S[iu, iv] + 1e-9 * np.maximum(1.0, D[iu, iv])
    tail = np.concatenate([iu[keep], iv[keep]])
    head = np.concatenate([iv[keep], iu[keep]])
    m, K, root = len(tail), k - 1, T[0]
    ar, one = np.arange(m), np.ones(m)
    rows, cols, vals, lo, hi = [], [], [], [], []
    for t in range(K):
        rows += [t * n + tail, t * n + head]
        cols += [m * (1 + t) + ar] * 2
        vals += [one, -one]
        b = np.zeros(n)
        b[root], b[T[t + 1]] = 1.0, -1.0
        lo.append(b)
        hi.append(b)
    for t in range(K):
        rows += [K * n + t * m + ar] * 2
        cols += [m * (1 + t) + ar, ar]
        vals += [one, -one]
        lo.append(np.full(m, -np.inf))
        hi.append(np.zeros(m))
    rows.append(K * n + K * m + head)
    cols.append(ar)
    vals.append(one)
    lo.append(np.zeros(n))
    hi.append(np.ones(n))
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(K * n + K * m + n, m * (1 + K)))
    ub = np.ones(m * (1 + K))
    ub[:m][head == root] = 0.0
    res = milp(np.concatenate([D[tail, head], np.zeros(K * m)]), integrality=np.r_[np.ones(m), np.zeros(K * m)],
               bounds=Bounds(0.0, ub), constraints=LinearConstraint(A.tocsr(), np.concatenate(lo), np.concatenate(hi)),
               options={'mip_rel_gap': 0.0})
    if not res.success:
        raise RuntimeError(f"Steiner tree flow model: {res.message}")
    used = {int(a) for a in ar if res.x[a] > 0.5}
    succ = {}
    for a in used:
        succ.setdefault(int(tail[a]), []).append(a)
    arcs, stack = [], [root]
    while stack:
        for a in succ.get(stack.pop(), []):
            arcs.append((int(tail[a]), int(head[a])))
            stack.append(int(head[a]))
    return float(sum(D[a, b] for a, b in arcs)), arcs


def steiner_tree(terminals, points):
    import numpy as np
    ids = tuple(sorted(points))
    key = (ids, tuple(sorted(set(terminals))))
    if key in _STEINER_CACHE:
        return _STEINER_CACHE[key]
    if ('D', ids) not in _STEINER_CACHE:
        D = np.array([[0.0 if a == b else float(get_distance(points[a], points[b])) for b in ids] for a in ids])
        D = np.minimum(D, D.T)
        for k in range(len(ids)):
            D = np.minimum(D, D[:, k:k + 1] + D[k:k + 1, :])
        _STEINER_CACHE[('D', ids)] = D
    D = _STEINER_CACHE[('D', ids)]
    pos = {a: k for k, a in enumerate(ids)}
    T = [pos[t] for t in key[1]]
    k, n = len(T), len(ids)
    if k <= 1:
        res = (0.0, [])
    elif k == 2:
        res = (float(D[T[0], T[1]]), [(ids[T[0]], ids[T[1]])])
    elif k > STEINER_DP_MAX:
        length, arcs = _steiner_flow(D, T)
        res = (length, sorted({tuple(sorted((ids[a], ids[b]))) for a, b in arcs if a != b}))
    else:
        full = (1 << k) - 1
        dp = np.full((1 << k, n), np.inf)
        split = np.zeros((1 << k, n), dtype=np.int64)
        relax = np.full((1 << k, n), -1, dtype=np.int64)
        for i, t in enumerate(T):
            dp[1 << i] = D[t]
        for mask in sorted(range(1, full + 1), key=lambda m_: bin(m_).count('1')):
            if mask & (mask - 1) == 0:
                continue
            best = np.full(n, np.inf)
            bsub = np.zeros(n, dtype=np.int64)
            sub = (mask - 1) & mask
            while sub:
                if sub < (mask ^ sub):
                    val = dp[sub] + dp[mask ^ sub]
                    better = val < best
                    best = np.where(better, val, best)
                    bsub = np.where(better, sub, bsub)
                sub = (sub - 1) & mask
            tot = best[:, None] + D
            u_arg = tot.argmin(axis=0)
            relaxed = tot[u_arg, np.arange(n)]
            use = relaxed < best - 1e-12
            dp[mask] = np.where(use, relaxed, best)
            split[mask] = bsub
            relax[mask] = np.where(use, u_arg, -1)

        def build(mask, v):
            if mask & (mask - 1) == 0:
                t = T[mask.bit_length() - 1]
                return [(t, v)] if t != v else []
            u = int(relax[mask][v])
            if u >= 0:
                return [(u, v)] + build_split(mask, u)
            return build_split(mask, v)

        def build_split(mask, v):
            sub = int(split[mask][v])
            return build(sub, v) + build(mask ^ sub, v)

        v0 = int(dp[full].argmin())
        edges = {tuple(sorted((ids[a], ids[b]))) for a, b in build(full, v0) if a != b}
        res = (float(dp[full][v0]), sorted(edges))
    _STEINER_CACHE[key] = res
    return res


def build_sensor_lookup(sensors):
    return {s.id: s for s in sensors}

def build_actuator_lookup(actuators):
    return {a.id: a for a in actuators}

def build_cost_map(cable_types):
    return {name: ct.cost_per_meter for name, ct in cable_types.items()}


def build_latency_map(cable_types):
    return {name: ct.latency_per_meter for name, ct in cable_types.items()}


def nominal_lane(net, t, getv):
    tr = net['flow_meta'][t]
    if tr['type'] not in ('BUS_SC', 'SC_BUS'):
        return t
    best = None
    for key, t2 in sorted(net['base_variants'][t].items(), key=lambda kv: kv[1]):
        if 'r' in key:
            continue
        src, dst = net['flow_ends'][t2]
        u = next((k for k, var in src.items() if getv(var) > 0.5), None)
        v = next((k for k, var in dst.items() if getv(var) > 0.5), None)
        if u is None or v is None:
            continue
        if u == v:
            return t2
        best = t2 if best is None else best
    return best if best is not None else t


def lane_keys(net, t_base, order, getv):
    tr = net['flow_meta'][t_base]
    keys = order[tr['type']]
    if tr['type'] not in ('BUS_SC', 'SC_BUS'):
        return keys
    variants = net['base_variants'][t_base]

    def rank(k):
        t = variants.get(k)
        if t is None:
            return 2
        src, dst = net['flow_ends'][t]
        u = next((kk for kk, var in src.items() if getv(var) > 0.5), None)
        v = next((kk for kk, var in dst.items() if getv(var) > 0.5), None)
        return 0 if (u is not None and u == v) else 1
    return sorted(keys, key=lambda k: ('r' in k, rank(k)))


def flow_routes(net, getv):
    routes = {}
    for t, (src, dst) in net['flow_ends'].items():
        u = next((k for k, var in src.items() if getv(var) > 0.5), None)
        v = next((k for k, var in dst.items() if getv(var) > 0.5), None)
        if u is None or v is None or u == v:
            routes[t] = None
            continue
        elig = net['flow_elig'][t]
        w_if = elig[0] if len(elig) == 1 else max(elig, key=lambda n: getv(net['w'][t, n]))
        pair = (min(u, v), max(u, v))
        plist = net['paths'][pair + (w_if,)]
        chosen = [p for p in range(len(plist)) if getv(net['sel'][pair + (w_if, p)]) > 0.5]
        routes[t] = (u, v, w_if, chosen[0] if chosen else None)
    return routes


def route_resources(net, u, v, w_if, p):
    if w_if in set((net.get('bus') or {}).get('types', ())):
        return [('line', (w_if, p))]
    pair = (min(u, v), max(u, v))
    path = net['paths'][pair + (w_if,)][p]
    seq = path if (u, v) == pair else path[::-1]
    return [(a, b, w_if) for a, b in zip(seq, seq[1:])]


def flow_is_silent(net, t, hot_val):
    tr = net['flow_meta'][t]
    return any(tr.get(inst_key, 'x') == 'r' and hot_val.get(tr[idx_key], 1.0) < 0.5
               for inst_key, idx_key in (('src_inst', 'src_idx'), ('dst_inst', 'dst_idx')))


def resource_messages(net, routes, getv, carried=None, failed_line=None):
    hot_val = {i: (1.0 if net['hot'][i] is None else getv(net['hot'][i])) for i in net.get('critical', [])}
    msgs = {}
    for t, rt in routes.items():
        if rt is None or rt[3] is None:
            continue
        if (carried is None and flow_is_silent(net, t, hot_val)) or (carried is not None and t not in carried):
            continue
        for res in route_resources(net, *rt):
            msgs.setdefault(res, []).append(('f', t))
    bus = net.get('bus') or {}
    for (b, d), var in bus.get('dev', {}).items():
        if b != failed_line and getv(var) > 0.5:
            msgs.setdefault(('line', b), []).append(('d', d))
    return msgs


def message_kbit(net, msg):
    return net['kbit'][msg[1]] if msg[0] == 'f' else net['bus']['devices'][msg[1]]['kbit']


def resource_rate(net, res):
    return net['cap_of'][res[1][0] if res[0] == 'line' else res[2]]


def message_prio(net, msg):
    pr = net.get('prio')
    if not pr:
        return 0.0
    return pr['f' if msg[0] == 'f' else 'd'].get(msg[1], float('inf'))


def frame_kbit(net, res):
    return net.get('frame_kbit', {}).get(res[1][0] if res[0] == 'line' else res[2], float('inf'))


def queue_ms(net, res, me, msgs):
    pm, wait, block = message_prio(net, me), 0.0, 0.0
    for mm in msgs.get(res, []):
        if mm == me:
            continue
        if message_prio(net, mm) <= pm:
            wait += message_kbit(net, mm)
        else:
            block = max(block, min(message_kbit(net, mm), frame_kbit(net, res)))
    return (wait + block) / resource_rate(net, res)


def robust_ms(net, base):
    lat = net['lat']
    return base * lat['factor'] + lat['sync'] + lat['platform'] if lat['robust'] else base


def crossing_ms(net, t, route, msgs):
    u, v, w_if, p = route
    base = 0.0
    for res in route_resources(net, u, v, w_if, p):
        base += net['kbit'][t] / net['cap_of'][w_if] + net['hop_ms'][w_if] + queue_ms(net, res, ('f', t), msgs)
    return robust_ms(net, base)


def device_line(net, d, getv):
    bus = net.get('bus') or {}
    return next((b for b in bus.get('inst', []) if (b, d) in bus['dev'] and getv(bus['dev'][b, d]) > 0.5), None)


def device_lines(net, d, getv, failed_line=None):
    bus = net.get('bus') or {}
    return [b for b in bus.get('inst', []) if (b, d) in bus['dev'] and getv(bus['dev'][b, d]) > 0.5 and b != failed_line]


def device_lane_ms(net, d, getv, msgs, failed_line=None):
    lines = device_lines(net, d, getv, failed_line)
    return max((device_ms(net, d, b, msgs) for b in lines), default=None)


def device_ms(net, d, line, msgs):
    info = net['bus']['devices'][d]
    base = info['kbit'] / net['cap_of'][info['iface']] + net['hop_ms'][info['iface']]
    if line is not None:
        base += queue_ms(net, ('line', line), ('d', d), msgs)
    return robust_ms(net, base)


def failure_report(net, scs, locations, getv, uncertainty=None):
    routes = flow_routes(net, getv)
    n_locs = len(locations)
    pos_x = {}
    for (i, j), var in net['x_vars'].items():
        if getv(var) > 0.5:
            pos_x[i] = j
    pos_r = {i: next((j for (ii, j), var in net['r'].items() if ii == i and getv(var) > 0.5), None) for i in net['critical']}
    hot_val = {i: (1.0 if net['hot'][i] is None else getv(net['hot'][i])) for i in net['critical']}
    cells_on = {j for j in range(n_locs) if getv(net['y_vars'][j]) > 0.5}
    switches = {j for j in range(n_locs) if net['switch_vars'].get(j) is not None and getv(net['switch_vars'][j]) > 0.5}
    bus_types = set((net.get('bus') or {}).get('types', ()))
    used_links = {(min(u, v), max(u, v), w_if) for rt in routes.values() if rt for u, v, w_if, _p in [rt] if w_if not in bus_types}
    used_lines = {(rt[2], rt[3]) for rt in routes.values() if rt and rt[2] in bus_types and rt[3] is not None}
    classes = set(net['fail_cfg'].get('classes', ['cell', 'switch', 'cable']))
    scenarios = []
    if 'cell' in classes:
        scenarios += [('cell', c) for c in sorted(cells_on | switches)]
    if 'switch' in classes:
        scenarios += [('switch', c) for c in sorted(switches)]
    if 'cable' in classes:
        scenarios += [('cable', k) for k in sorted(used_links)]
        scenarios += [('line', ln) for ln in sorted(used_lines)]
    order = {'SC_SC': [('x', 'x'), ('x', 'r'), ('r', 'x'), ('r', 'r')], 'SENS_SC': [('a', 'x'), ('a', 'r')], 'SC_ACT': [('x', 'a'), ('r', 'a')]}
    n_lanes = max((sum(1 for b in (net.get('bus') or {}).get('inst', []) if b[0] == n) for n in bus_types), default=0)
    order['BUS_SC'] = [(f'a{k}', inst) for inst in ('x', 'r') for k in range(n_lanes)]
    order['SC_BUS'] = [(inst, f'a{k}') for inst in ('x', 'r') for k in range(n_lanes)]

    def ends(t):
        src, dst = net['flow_ends'][t]
        return (next((k for k, var in src.items() if getv(var) > 0.5), None), next((k for k, var in dst.items() if getv(var) > 0.5), None))

    def alive(t, kind, elem):
        u, v = ends(t)
        if u is None or v is None or (kind == 'cell' and elem in (u, v)):
            return False
        tr_t = net['flow_meta'][t]
        if kind == 'line' and tr_t['type'] in ('BUS_SC', 'SC_BUS') and tr_t.get('line') == elem:
            return False
        if u == v:
            return True
        rt = routes.get(t)
        if rt is None or rt[3] is None:
            return False
        _u, _v, w_if, p = rt
        path = net['paths'][(min(u, v), max(u, v), w_if)][p]
        if kind in ('cell', 'switch') and elem in path[1:-1]:
            return False
        if kind == 'switch' and w_if == 'ETH' and elem in (u, v):
            return False
        if kind == 'cable' and w_if not in bus_types and elem[2] == w_if and any({pa, pb} == {elem[0], elem[1]} for pa, pb in zip(path, path[1:])):
            return False
        if kind == 'line' and (w_if, p) == elem:
            return False
        return True

    crit = set(net['critical'])
    non_crit = [i for i in range(len(scs)) if i not in crit]
    inputs, act_outputs = {}, {}
    for t_base in net['base_variants']:
        tr = net['flow_meta'][t_base]
        if tr['type'] in ('SC_ACT', 'SC_BUS'):
            act_outputs.setdefault(tr['src_idx'], []).append(t_base)
        else:
            inputs.setdefault(tr['dst_idx'], []).append(t_base)
    dev_lines = {i: [set(device_lines(net, d, getv)) for d, info in (net.get('bus') or {}).get('devices', {}).items() if i in info['users']]
                 for i in non_crit}

    def lost_functions(kind, elem):
        dead = lambda t_base: not any(alive(t, kind, elem) for t in net['base_variants'][t_base].values())
        lost = {i for i in non_crit if (kind == 'cell' and pos_x.get(i) == elem) or (kind == 'line' and {elem} in dev_lines[i])
                or any(dead(tb) for tb in act_outputs.get(i, []))}
        grown = True
        while grown:
            grown = False
            for i in non_crit:
                if i not in lost and any((net['flow_meta'][tb]['type'] == 'SC_SC' and net['flow_meta'][tb]['src_idx'] in lost) or dead(tb)
                                         for tb in inputs.get(i, [])):
                    lost.add(i)
                    grown = True
        return sorted(scs[i].id for i in lost)

    report = []
    for kind, elem in scenarios:
        carried = {t for t in net['flow_meta'] if not flow_is_silent(net, t, hot_val) and alive(t, kind, elem)}
        for t_base, variants in net['base_variants'].items():
            pick = next((variants[k] for k in lane_keys(net, t_base, order, getv) if k in variants and alive(variants[k], kind, elem)), None)
            if pick is not None:
                carried.add(pick)
        msgs = resource_messages(net, routes, getv, carried=carried, failed_line=(elem if kind == 'line' else None))
        entry = {'kind': kind, 'element': (f"{locations[elem[0]].id}-{locations[elem[1]].id} {elem[2]}" if kind == 'cable'
                                           else f"bus line {elem[0]}{elem[1] + 1}" if kind == 'line' else locations[elem].id)}
        if kind == 'cell':
            act = [i for i in net['critical'] if pos_x.get(i) == elem and pos_r.get(i) is not None and hot_val[i] < 0.5]
            entry['activated_cold_standbys'] = [scs[i].id for i in act]
            load = {}
            for i in act:
                load[locations[pos_r[i]].id] = load.get(locations[pos_r[i]].id, 0.0) + scs[i].cpu_req * sw_multiplier(scs[i], uncertainty, 'cpu')
            entry['added_cpu_load'] = {k: round(v) for k, v in load.items()}
        chains = {}
        for ch in net.get('chains', []):
            total, broken = 0.0, False
            for t_base in ch['flows']:
                picked = None
                for key in lane_keys(net, t_base, order, getv):
                    t = net['base_variants'][t_base].get(key)
                    if t is not None and alive(t, kind, elem):
                        picked = t
                        break
                if picked is None:
                    broken = True
                    break
                u, v = ends(picked)
                if u != v:
                    total += crossing_ms(net, picked, routes[picked], msgs)
            for d in ch.get('dev_ends', []):
                ms = device_lane_ms(net, d, getv, msgs, failed_line=(elem if kind == 'line' else None))
                if ms is None:
                    broken = True
                    continue
                total += ms
            status = 'broken (non-critical member lost)' if broken else ('ok' if total <= ch['degraded'] + 1e-6 else 'over bound')
            chains[ch['id']] = {'latency_ms': None if broken else round(total, 3), 'degraded_bound_ms': ch['degraded'],
                                'ok': status != 'over bound', 'status': status}
        entry['chains'] = chains
        worst = max((c for c in chains.values() if c['latency_ms'] is not None), key=lambda c: c['latency_ms'] / c['degraded_bound_ms'], default=None)
        entry['worst_chain_ratio'] = round(worst['latency_ms'] / worst['degraded_bound_ms'], 3) if worst else None
        entry['lost_functions'] = lost_functions(kind, elem)
        report.append(entry)
    return report


def lost_function_summary(scenarios, n_non_critical):
    if not scenarios:
        return None
    counts = [len(e.get('lost_functions', [])) for e in scenarios]
    worst = max(range(len(counts)), key=lambda k: counts[k])
    return {'non_critical': n_non_critical, 'scenarios': len(counts), 'worst': counts[worst],
            'worst_scenario': f"{scenarios[worst]['kind']} {scenarios[worst]['element']}", 'mean': round(sum(counts) / len(counts), 3)}


def chain_latencies(net, routes, getv):
    msgs = resource_messages(net, routes, getv)
    out = {}
    for ch in net.get('chains', []):
        total, crossings = 0.0, []
        for t in ch['flows']:
            t = nominal_lane(net, t, getv)
            r = routes.get(t)
            if r is None or r[3] is None:
                continue
            u, v, w_if, p = r
            ms = crossing_ms(net, t, r, msgs)
            total += ms
            crossings.append({'flow': t, 'from': net['site_ids'][u], 'to': net['site_ids'][v], 'iface': w_if,
                              'hops': len(route_resources(net, *r)), 'ms': round(ms, 3)})
        dev = sum(device_lane_ms(net, d, getv, msgs) or 0.0 for d in ch.get('dev_ends', []))
        total += dev
        out[ch['id']] = {'latency_ms': round(total, 3), 'budget_ms': ch['budget'], 'ok': bool(total <= ch['budget'] + 1e-6),
                         'device_bus_ms': round(dev, 3), 'crossings': crossings}
    return out


def margin_of(cfg, res):
    free = cfg.get('free_capacity') or {}
    f = float(free.get(res, free.get('default', 0.0)) or 0.0)
    return 1.0 / (1.0 - f) - 1.0 if f < 1.0 else 0.0


def sw_multiplier(sc, uncertainty, res='cpu'):
    unc = uncertainty or {}
    m = 1.0
    domain = str(getattr(sc, 'domain', '') or '').strip()
    if unc.get('sw_margin'):
        cfg = unc.get('params', {}).get('sw_margin', {})
        if 'free_capacity' in cfg:
            m *= 1.0 + rho_level(unc) * margin_of(cfg, res)
        elif domain.upper() in {str(d).strip().upper() for d in cfg.get('domains', [])}:
            margins = {int(k): float(v) for k, v in cfg.get('asil_margins', {}).items()}
            m_asil = margins.get(int(getattr(sc, 'asil_req', 0)), float(cfg['default_asil_margin']))
            m_ai = float(cfg['ai_margin']) if (cfg['ai_hw_feature'] in (getattr(sc, 'hw_required', None) or [])) else 0.0
            m *= 1.0 + rho_level(unc) * (m_asil + m_ai)
    if unc.get('growth'):
        cfg = unc.get('params', {}).get('growth', {})
        rcfg = unc.get('params', {}).get('rho', {})
        rate = float(rcfg.get('growth_rate_ref', 0.20)) * float(cfg.get('profiles', {}).get(domain, cfg.get('default_profile', 0.0)))
        m *= (1.0 + rate) ** (float(rcfg.get('growth_years', 5)) * rho_level(unc))
    return m


def rho_level(uncertainty):
    return float((uncertainty or {}).get('rho', 1.0))


def _fail_op_base(sc, fail_cfg):
    flag = getattr(sc, 'fail_operational', None)
    if flag is not None:
        return bool(flag)
    return int(sc.asil_req) >= int((fail_cfg or {}).get('critical_asil_min', 3))


def is_fail_operational(sc, fail_cfg):
    ids = (fail_cfg or {}).get('standby_ids')
    if ids is not None:
        return sc.id in ids
    return _fail_op_base(sc, fail_cfg)


def standby_units(scs, chains, fail_cfg):
    fo = {sc.id: sc for sc in scs if _fail_op_base(sc, fail_cfg)}
    units, owner = [], {}
    for ch in (chains or []):
        mem = [s for s in ch.get('swcs', []) if s in fo]
        if not mem:
            continue
        ftti = float(ch['ftti_ms']) if ch.get('ftti_ms') is not None else float('inf')
        hit = sorted({owner[s] for s in mem if s in owner})
        if hit:
            k = hit[0]
            for k2 in hit[1:]:
                units[k]['m'] |= units[k2]['m']
                units[k]['ftti'] = min(units[k]['ftti'], units[k2]['ftti'])
                for s in units[k2]['m']:
                    owner[s] = k
                units[k2] = None
            units[k]['m'] |= set(mem)
            units[k]['ftti'] = min(units[k]['ftti'], ftti)
        else:
            k = len(units)
            units.append({'m': set(mem), 'ftti': ftti})
        for s in mem:
            owner[s] = k
    units = [u for u in units if u is not None]
    for s in fo:
        if s not in owner:
            units.append({'m': {s}, 'ftti': float('inf')})
    units.sort(key=lambda u: (-max(int(fo[s].asil_req) for s in u['m']), u['ftti'], min(u['m'])))
    return [sorted(u['m']) for u in units]


def gradual_standby_ids(scs, chains, rho, fail_cfg):
    units = standby_units(scs, chains, fail_cfg)
    total = sum(len(u) for u in units)
    target = float(rho) * total - 1e-9
    ids = set()
    for u in units:
        if len(ids) >= target:
            break
        ids.update(u)
    return ids


def with_gradual_standbys(uncertainty, scs, chains):
    unc = dict(uncertainty or {})
    params = dict(unc.get('params') or {})
    fcfg = dict(params.get('failures') or {})
    fcfg.pop('standby_ids', None)
    if unc.get('failures') and fcfg.get('gradual'):
        ids = gradual_standby_ids(scs, chains, rho_level(unc), fcfg)
        n_all = sum(1 for sc in scs if _fail_op_base(sc, fcfg))
        fcfg['standby_ids'] = sorted(ids)
        print(f"  Failures, gradual rule: {len(ids)} of {n_all} fail-operational components get a standby at rho "
              f"{rho_level(unc):.2f} (units by ASIL, FTTI, id)")
    params['failures'] = fcfg
    unc['params'] = params
    return unc


def site_derating(location, uncertainty):
    if not (uncertainty or {}).get('derating'):
        return 1.0
    h = float(getattr(location, 'health_factor', 1.0))
    return max(0.0, 1.0 - rho_level(uncertainty) * (1.0 - h))


def _bin_caps(bin_cap, n_bins):
    return [dict(c) for c in bin_cap] if isinstance(bin_cap, (list, tuple)) else [dict(bin_cap) for _ in range(n_bins)]


def _first_fit_decreasing(members, demand, bin_cap, n_bins=None):
    caps = _bin_caps(bin_cap, n_bins)
    active = [r for r in (caps[0] if caps else {}) if any(c[r] > 0 for c in caps)]
    top = {r: max(c[r] for c in caps) for r in active}

    def largest(i):
        return max((demand[i][r] / top[r] for r in active), default=0.0)

    load = [{r: 0.0 for r in active} for _ in caps]
    slots = {}
    for i in sorted(members, key=largest, reverse=True):
        for b, c in enumerate(caps):
            if all(load[b][r] + demand[i][r] <= c[r] * (1.0 + 1e-9) + 1e-9 for r in active):
                for r in active:
                    load[b][r] += demand[i][r]
                slots[i] = b
                break
        else:
            return None
    return slots


def _exact_packing(members, demand, bin_cap, n_bins=None):
    import gurobipy as gp
    from gurobipy import GRB
    caps = _bin_caps(bin_cap, n_bins)
    if not caps:
        return None
    active = [r for r in caps[0] if any(c[r] > 0 for c in caps)]
    m = gp.Model("partition_packing")
    m.setParam('OutputFlag', 0)
    nb = len(caps)
    u = {(i, b): m.addVar(vtype=GRB.BINARY) for i in members for b in range(nb)}
    for i in members:
        m.addConstr(gp.quicksum(u[i, b] for b in range(nb)) == 1)
    for b in range(nb):
        for r in active:
            m.addConstr(gp.quicksum(demand[i][r] * u[i, b] for i in members) <= caps[b][r])
    m.optimize()
    if m.SolCount == 0:
        return None
    return {i: next(b for b in range(nb) if u[i, b].X > 0.5) for i in members}


def _type_bins(types, counts, h, res):
    order = sorted((t for t in range(len(types)) for _ in range(counts[t])),
                   key=lambda t: (-types[t]['cpu_cap'], -types[t]['ram_cap'], types[t]['id']))
    return order, [{r: types[t][f'{r}_cap'] * (h if r in ('cpu', 'tops') else 1.0) for r in res} for t in order]


def _packs(keys, demand, bin_cap):
    if not keys:
        return True
    if not bin_cap:
        return False
    return _first_fit_decreasing(keys, demand, bin_cap) is not None or _exact_packing(keys, demand, bin_cap) is not None


def minimal_unpackable(keys, demand, bin_cap):
    if not bin_cap or _packs(keys, demand, bin_cap):
        return list(keys)
    top = {r: max(c[r] for c in bin_cap) for r in bin_cap[0]}

    def size(k):
        return max((demand[k][r] / top[r] for r in top if top[r] > 0), default=0.0)

    core = list(keys)
    for k in sorted(keys, key=size):
        if len(core) <= 1:
            break
        trial = [m for m in core if m != k]
        if not _packs(trial, demand, bin_cap):
            core = trial
    return core


def useful_types(keys, demand, types, h, res):
    out = []
    for t, ty in enumerate(types):
        cap = {r: ty[f'{r}_cap'] * (h if r in ('cpu', 'tops') else 1.0) for r in res}
        if any(all(demand[k][r] <= cap[r] * (1.0 + 1e-9) + 1e-9 for r in res) for k in keys):
            out.append(t)
    return out


def failing_mixes(keys, demand, types, counts, h, res, max_blocks=None, useful=None):
    fails = [tuple(counts)]
    for t in (useful if useful is not None else range(len(types))):
        c = list(counts)
        c[t] += 1
        if max_blocks is not None and sum(c) > int(max_blocks):
            continue
        _order, bins = _type_bins(types, c, h, res)
        if not _packs(keys, demand, bins):
            fails.append(tuple(c))
    return fails


def pack_partitions(x, P, scs, locations, partitions, uncertainty=None, standby=None, Q=None, r=None, hot=None, critical=None):
    types = partitions.get('types') if Q else None
    res = ('cpu', 'ram', 'rom', 'tops') if types and any(ty.get('tops_cap', 0) > 0 for ty in types) else ('cpu', 'ram', 'rom')
    caps = {r: float(partitions.get(f'{r}_cap', 0) or 0) for r in res}
    groups = {}
    for (i, j), var in x.items():
        if var.X > 0.5:
            groups.setdefault((j, scs[i].asil_req), []).append(((i, 'x'), var))
    for (i, j), var in (standby or {}).items():
        if var.X > 0.5:
            groups.setdefault((j, scs[i].asil_req), []).append(((i, 'r'), var))
    woken = {}
    if r and critical:
        pos_x = {i: j for (i, j), var in x.items() if var.X > 0.5}
        for (i, j), var in r.items():
            h_i = (hot or {}).get(i)
            if var.X > 0.5 and h_i is not None and h_i.X < 0.5 and i in pos_x and pos_x[i] != j:
                c = pos_x[i]
                groups.setdefault((j, scs[i].asil_req), []).append(((i, 'c'), var))
                woken.setdefault((j, scs[i].asil_req), {}).setdefault(c, []).append(((i, 'f', c), x[i, c] + var - 1 - h_i))

    def _bins(j, a):
        n_bins = int(round(P[j, a].X))
        h = site_derating(locations[j], uncertainty)
        if types:
            counts = tuple(int(round(Q[j, a, t].X)) for t in range(len(types)))
            order, bin_cap = _type_bins(types, counts, h, res)
        else:
            counts, order = None, None
            bin_cap = [{r_: caps[r_] * (h if r_ == 'cpu' else 1.0) for r_ in caps}] * n_bins
        return n_bins, h, counts, order, bin_cap

    def _demand(keys):
        return {k: {r_: (0.0 if k[1] == 'c' and r_ != 'rom' else
                         float(getattr(scs[k[0]], f'{r_}_req', 0) or 0) * sw_multiplier(scs[k[0]], uncertainty, r_)) for r_ in res}
                for k in keys}

    partition_map, failed, part_type = {}, [], {}
    for (j, a), members in sorted(groups.items()):
        n_bins, h, counts, order, bin_cap = _bins(j, a)
        keys = [k for k, _v in members]
        demand = _demand(keys)
        slots = _first_fit_decreasing(keys, demand, bin_cap) if bin_cap else None
        if slots is None:
            slots = _exact_packing(keys, demand, bin_cap) if bin_cap else None
        if slots is None:
            failed.append((j, a, list(members), n_bins, counts, demand, h, res))
            slots = {k: 0 for k in keys}
        else:
            for c, extra in sorted(woken.get((j, a), {}).items()):
                woke = {k[0] for k, _v in extra}
                stay = [m for m in members if not (m[0][1] == 'c' and m[0][0] in woke)]
                keys_c = [k for k, _v in stay] + [k for k, _v in extra]
                demand_c = _demand(keys_c)
                if not _packs(keys_c, demand_c, bin_cap):
                    failed.append((j, a, stay + list(extra), n_bins, counts, demand_c, h, res))
        for (i, inst), b in slots.items():
            partition_map[scs[i].id + ('@standby' if inst in ('r', 'c') else '')] = f"{locations[j].id}_asil{a}_p{b}"
        if order is not None:
            for b, t in enumerate(order):
                part_type[f"{locations[j].id}_asil{a}_p{b}"] = types[t]['id']
    for (j, a), by_c in sorted(woken.items()):
        if (j, a) in groups:
            continue
        n_bins, h, counts, order, bin_cap = _bins(j, a)
        for c, extra in sorted(by_c.items()):
            keys_c = [k for k, _v in extra]
            demand_c = _demand(keys_c)
            if not _packs(keys_c, demand_c, bin_cap):
                failed.append((j, a, list(extra), n_bins, counts, demand_c, h, res))
    return partition_map, failed, part_type


def calculate_sensor_actuator_cable_costs(x, scs, locations, sensors, actuators, cable_types, attach_s=None, attach_a=None, bus=None, standby_r=None):
    cable_length = 0.0
    cable_cost = 0.0
    cost_map = {name: ct.cost_per_meter for name, ct in cable_types.items()}
    sensor_lookup = build_sensor_lookup(sensors)
    actuator_lookup = build_actuator_lookup(actuators)

    bus_types = set((bus or {}).get('types', ()))
    dev_f = float((bus or {}).get('dev_factor', 1.0) or 1.0)

    hosts = list((x or {}).items()) + list((standby_r or {}).items())
    for (i, j), var in hosts:
        if var.X > 0.5:
            sc = scs[i]
            for dev in [sensor_lookup.get(s_id) for s_id in (sc.sensors or [])] + [actuator_lookup.get(a_id) for a_id in (sc.actuators or [])]:
                if dev is None or not dev.location or dev.interface == 'ETH' or dev.interface in bus_types:
                    continue
                dist = get_distance(dev.location, locations[j].location)
                cable_length += dist
                cable_cost += dev_f * dist * cost_map.get(dev.interface, 0.0)

    for att, devs in ((attach_s, sensors), (attach_a, actuators)):
        for (di, j), av in (att or {}).items():
            if av.X > 0.5 and getattr(devs[di], 'location', None):
                dist = get_distance(devs[di].location, locations[j].location)
                cable_length += dist
                cable_cost += dev_f * dist * cost_map.get(getattr(devs[di], 'interface', 'ETH'), 0.0)

    for b, lv in (bus or {}).get('len', {}).items():
        length = float(lv.X)
        if length > 1e-9:
            cable_length += length
            cable_cost += dev_f * length * cost_map.get(b[0], 0.0)

    return cable_length, cable_cost


def calculate_partition_cost(P, partitions):
    num_partitions_opened = sum(int(round(var.X)) for var in P.values())
    return num_partitions_opened * partitions.get('cost', 0)


def calculate_hardware_cost(hw_use, hardwares, locations):
    hw_opened = []
    for (j, h), var in hw_use.items():
        hw_opened += [f"{h}@{locations[j].id}"] * int(round(var.X))

    hw_cost = sum(hardwares.get(hw_name.split('@')[0], 0) for hw_name in hw_opened)
    return hw_cost, hw_opened


def calculate_interface_cost(if_use, interfaces, locations):
    if_opened = []
    if_cost = 0.0
    for (j, i_name), var in if_use.items():
        if var.X > 0.1:
            count = int(round(var.X))
            if_opened.append(f"{count}x{i_name}@{locations[j].id}")
            if_cost += count * interfaces[i_name].port_cost

    return if_cost, if_opened


def calculate_communication_cost(comm, locations, cable_types):
    comm_cost = 0.0
    for (j1, j2, iface), var in comm.items():
        if var.X > 0.5:
            num_links = int(round(var.X))
            dist = get_distance(locations[j1].location, locations[j2].location)
            cost_per_link = dist * cable_types[iface].cost_per_meter
            comm_cost += num_links * cost_per_link

    return comm_cost


def extract_communication_links(comm, locations, min_link_value=0.5):
    links = []
    for (j1, j2, iface), var in comm.items():
        if var.X > min_link_value:
            links.append({
                'src_loc': locations[j1].id,
                'dst_loc': locations[j2].id,
                'iface': iface,
                'count': int(round(var.X)),
            })
    return links


def extract_solution(x, P, y, hw_use, if_use, comm, scs, locations, sensors, actuators,
                     cable_types, partitions, hardwares, interfaces, comm_matrix=None,
                     traffic_flows=None, net=None, attach_s=None, attach_a=None,
                     bus=None, partition_map=None, standby=None, uncertainty=None, part_type=None):
        assignment_map = {}
        partition_map = dict(partition_map or {})
        num_locations_used = 0

        for (i, j), var in x.items():
            if var.X > 0.5:
                assignment_map[scs[i].id] = locations[j].id
                partition_map.setdefault(scs[i].id, f"{locations[j].id}_asil{scs[i].asil_req}_p0")
        standby_map = {}
        if standby:
            r_vars, hot_vars = standby
            for (i, j), var in r_vars.items():
                if var.X > 0.5:
                    kind = 'hot' if (hot_vars.get(i) is None or hot_vars[i].X > 0.5) else 'cold'
                    standby_map[scs[i].id] = {'cell': locations[j].id, 'type': kind,
                                              'partition': partition_map.get(scs[i].id + '@standby', f"{locations[j].id}_asil{scs[i].asil_req}_p0")}
            print(f"\n[DEBUG] Standby instances: {len(standby_map)} ({sum(1 for s in standby_map.values() if s['type'] == 'hot')} hot, "
                  f"{sum(1 for s in standby_map.values() if s['type'] == 'cold')} cold)")
        partition_map = {k: v for k, v in partition_map.items() if not k.endswith('@standby')}

        print(f"\n[DEBUG] SC Placement Analysis:")
        print(f"  Total SCs assigned: {len(assignment_map)}")
        if comm_matrix:
            print(f"  Communication links required: {len(comm_matrix)}")
            for link in comm_matrix:
                src_id = link.get('src')
                dst_id = link.get('dst')
                vol = link.get('volume', 0)
                src_loc = assignment_map.get(src_id, 'NOT_ASSIGNED')
                dst_loc = assignment_map.get(dst_id, 'NOT_ASSIGNED')
                same_loc = (src_loc == dst_loc)
                print(f"    {src_id} ({src_loc}) -> {dst_id} ({dst_loc}) vol={vol} {'[SAME_LOC]' if same_loc else '[DIFFERENT_LOC]'}")

        num_partitions_opened = sum(int(round(var.X)) for var in P.values())
        site_fixed_cost = float(partitions.get('site_fixed_cost', 0) or 0)
        site_cost = site_fixed_cost * sum(1 for var in y.values() if var.X > 0.5)
        switch_node_cost = float(partitions.get('switch_node_cost', 0) or 0)
        if switch_node_cost > 0:
            site_cost += switch_node_cost * sum(1 for j, var in y.items() if var.X < 0.5 and hw_use.get((j, 'SWITCH')) is not None
                                                and hw_use[j, 'SWITCH'].X > 0.5)

        hw_cost, hw_opened = calculate_hardware_cost(hw_use, hardwares, locations)
        hw_unit_types = None
        hw_t = (net or {}).get('hw_t') or {}
        if hw_t:
            capd = (partitions or {}).get('hw_capacity', {}) or {}
            hw_unit_types = {}
            for (j, h, t), var in hw_t.items():
                n = int(round(var.X))
                if n > 0:
                    ty = capd[h]['types'][t]
                    hw_unit_types.setdefault(f"{h}@{locations[j].id}", {})[ty['id']] = n
                    hw_cost += n * float(ty['cost'])

        switches_selected = [hw for hw in hw_opened if hw.startswith("SWITCH")]
        all_hw_available = list(hardwares.keys()) if hardwares else []
        if switches_selected:
            print(f"[DEBUG] SWITCH selected at: {switches_selected}")
        elif "SWITCH" in all_hw_available:
            print(f"[DEBUG] WARNING: SWITCH available but NOT selected in optimal solution")

        if_cost, if_opened = calculate_interface_cost(if_use, interfaces, locations)

        locs_used = set()
        for (i, j), var in x.items():
            if var.X > 0.5:
                locs_used.add(j)
        num_locations_used = len(locs_used)

        cable_length, cable_cost = calculate_sensor_actuator_cable_costs(
            x, scs, locations, sensors, actuators, cable_types,
            attach_s=attach_s, attach_a=attach_a, bus=bus, standby_r=(standby[0] if standby else None)
        )

        partition_cost = calculate_partition_cost(P, partitions)
        partition_type_counts = None
        if (partitions or {}).get('types') and net and net.get('Q'):
            types = partitions['types']
            partition_type_counts = {}
            for (j, a, t), var in net['Q'].items():
                if var.X > 0.5:
                    partition_type_counts.setdefault(f"{locations[j].id}_asil{a}", {})[types[t]['id']] = int(round(var.X))
            partition_cost = sum(types[t]['cost'] * int(round(var.X)) for (j, a, t), var in net['Q'].items())
        comm_cost = calculate_communication_cost(comm, locations, cable_types)
        comm_links = extract_communication_links(comm, locations)

        backbone_length = sum(
            int(round(var.X)) * get_distance(locations[j1].location, locations[j2].location)
            for (j1, j2, _iface), var in comm.items()
            if var.X > 0.5
        )
        peripheral_cable_length = cable_length
        cable_length = peripheral_cable_length + backbone_length

        if attach_s:
            attached = []
            for (si, j), av in attach_s.items():
                if av.X > 0.5:
                    attached.append((sensors[si].id, locations[j].id))
            print("\n[DEBUG] ETH Sensor Attachments:")
            if not attached:
                print("  None")
            else:
                print(f"  Total: {len(attached)}")
                for sid, lid in attached[:20]:
                    print(f"  {sid} -> {lid}")
                if len(attached) > 20:
                    print(f"  ... ({len(attached) - 20} more)")

        if attach_a:
            attached = []
            for (ai, j), av in attach_a.items():
                if av.X > 0.5:
                    attached.append((actuators[ai].id, locations[j].id))
            print("\n[DEBUG] ETH Actuator Attachments:")
            if not attached:
                print("  None")
            else:
                print(f"  Total: {len(attached)}")
                for aid, lid in attached[:20]:
                    print(f"  {aid} -> {lid}")
                if len(attached) > 20:
                    print(f"  ... ({len(attached) - 20} more)")

        print(f"\n[DEBUG] Selected Backbone Links (comm):")
        if not comm_links:
            print("  No backbone links opened")
        else:
            def _lk_sort_key(lk):
                a, b = (lk['src_loc'], lk['dst_loc']) if lk['src_loc'] <= lk['dst_loc'] else (lk['dst_loc'], lk['src_loc'])
                return (lk.get('iface', ''), a, b)

            for lk in sorted(comm_links, key=_lk_sort_key):
                src_id = lk['src_loc']
                dst_id = lk['dst_loc']
                iface = lk.get('iface', '?')
                count = lk.get('count', 0)
                src_idx = next((ii for ii, loc in enumerate(locations) if loc.id == src_id), None)
                dst_idx = next((ii for ii, loc in enumerate(locations) if loc.id == dst_id), None)
                if src_idx is not None and dst_idx is not None:
                    dist = get_distance(locations[src_idx].location, locations[dst_idx].location)
                    cpm = cable_types.get(iface).cost_per_meter if cable_types.get(iface) else 0.0
                    print(f"  {src_id} <-> {dst_id} iface={iface} count={count} dist={dist:.2f} cost/link={dist * cpm:.2f}")
                else:
                    print(f"  {src_id} <-> {dst_id} iface={iface} count={count}")

        comm_link_peak_load = {}
        backbone_routes = []
        chain_latency = {}
        failure_scenarios = []
        lost_summary = None
        print(f"\n[DEBUG] Backbone routing (site pairs with traffic):")
        if not net:
            print("  Routing unavailable (net not provided to extract_solution)")
        else:
            paths, sel = net['paths'], net['sel']
            arc_load = {}
            relayed = 0
            pair_dir_vol = {}
            hot_val = {i: (1.0 if net['hot'][i] is None else net['hot'][i].X) for i in net.get('critical', [])}
            for t, (src, dst) in net['flow_ends'].items():
                u = next((k for k, var in src.items() if var.X > 0.5), None)
                v = next((k for k, var in dst.items() if var.X > 0.5), None)
                if u is None or v is None or u == v:
                    continue
                elig = net['flow_elig'][t]
                w_if = elig[0] if len(elig) == 1 else max(elig, key=lambda n: net['w'][t, n].X)
                pair = (min(u, v), max(u, v))
                tr = net['flow_meta'][t]
                silent = any(tr.get(inst_key, 'x') == 'r' and hot_val.get(tr[idx_key], 1.0) < 0.5
                             for inst_key, idx_key in (('src_inst', 'src_idx'), ('dst_inst', 'dst_idx')))
                vols = pair_dir_vol.setdefault(pair + (w_if,), [0.0, 0.0])
                if not silent:
                    vols[0 if (u, v) == pair else 1] += net['volumes'][t]
            bus_types = set((net.get('bus') or {}).get('types', ()))
            line_traffic = {}
            for (u, v, w_if), plist in paths.items():
                chosen = [p for p in range(len(plist)) if sel[u, v, w_if, p].X > 0.5]
                if not chosen:
                    continue
                p = chosen[0]
                path = plist[p]
                if (u, v, w_if) not in pair_dir_vol:
                    continue
                vol_fwd, vol_rev = pair_dir_vol[(u, v, w_if)]
                ids = [locations[j].id for j in path]
                entry = {'src_loc': ids[0], 'dst_loc': ids[-1], 'iface': w_if, 'path': ids,
                         'volume_fwd': round(vol_fwd, 3), 'volume_rev': round(vol_rev, 3)}
                if w_if in bus_types:
                    entry['line'] = f"{w_if}{p + 1}"
                    line_traffic[(w_if, p)] = line_traffic.get((w_if, p), 0.0) + vol_fwd + vol_rev
                else:
                    for vol, seq in ((vol_fwd, path), (vol_rev, path[::-1])):
                        for a, b in zip(seq, seq[1:]):
                            arc_load[(a, b, w_if)] = arc_load.get((a, b, w_if), 0.0) + vol
                relayed += int(len(path) > 2)
                backbone_routes.append(entry)
                print(f"  {ids[0]} <-> {ids[-1]} [{w_if}{' line ' + entry['line'] if 'line' in entry else ''}]: {' -> '.join(ids)}  "
                      f"volume {vol_fwd:.1f} / {vol_rev:.1f}" + ("  (relayed)" if len(path) > 2 else ""))
            print(f"  Summary: {len(backbone_routes)} site pairs with traffic, {relayed} relayed")
            for (a, b, w_if), load in arc_load.items():
                ia, ib = locations[a].id, locations[b].id
                key = f"{min(ia, ib)}|{max(ia, ib)}|{w_if}"
                comm_link_peak_load[key] = max(comm_link_peak_load.get(key, 0.0), float(load))
            if net.get('critical'):
                failure_scenarios = failure_report(net, scs, locations, lambda var: var.X, uncertainty)
                over = [e for e in failure_scenarios if any(c['status'] == 'over bound' for c in e['chains'].values())]
                broken = [e for e in failure_scenarios if any(c['status'].startswith('broken') for c in e['chains'].values())]
                print(f"\n[DEBUG] Failure scenarios: {len(failure_scenarios)} checked, {len(over)} with a chain over its degraded bound, "
                      f"{len(broken)} with a chain broken by a lost non-critical member; worst chain ratio "
                      f"{max((e['worst_chain_ratio'] or 0) for e in failure_scenarios):.2f}")
                lost_summary = lost_function_summary(failure_scenarios, len(scs) - len(net['critical']))
                if lost_summary:
                    print(f"[DEBUG] Lost functions: of {lost_summary['non_critical']} non-critical components at most "
                          f"{lost_summary['worst']} lost ({lost_summary['worst_scenario']}), mean {lost_summary['mean']} per failure")
            if net.get('chains'):
                chain_latency = chain_latencies(net, flow_routes(net, lambda var: var.X), lambda var: var.X)
                print(f"\n[DEBUG] Chain network latency (ms):")
                for cid, c in chain_latency.items():
                    print(f"  {cid:42s} {c['latency_ms']:8.3f} / budget {c['budget_ms']:7.1f}  crossings {len(c['crossings'])}  {'ok' if c['ok'] else 'VIOLATED'}")

        total_cost = partition_cost + site_cost + hw_cost + if_cost + cable_cost + comm_cost

        partition_counts = {f"{locations[j].id}_asil{a}": int(round(var.X)) for (j, a), var in P.items() if var.X > 0.5}
        flow_route_list = []
        if net:
            routes = flow_routes(net, lambda var: var.X)
            for t, tr in net['flow_meta'].items():
                if tr['type'] == 'SC_SC':
                    src, dst = scs[tr['src_idx']].id, scs[tr['dst_idx']].id
                elif tr['type'] == 'SENS_SC':
                    src, dst = sensors[tr['sensor_idx']].id, scs[tr['dst_idx']].id
                elif tr['type'] == 'BUS_SC':
                    src, dst = tr['dev'], scs[tr['dst_idx']].id
                elif tr['type'] == 'SC_BUS':
                    src, dst = scs[tr['src_idx']].id, tr['dev']
                else:
                    src, dst = scs[tr['src_idx']].id, actuators[tr['act_idx']].id
                entry = {'type': tr['type'], 'src': src, 'dst': dst, 'line': None,
                         'src_inst': tr.get('src_inst', 'a' if tr['type'] == 'SENS_SC' else 'x'),
                         'dst_inst': tr.get('dst_inst', 'a' if tr['type'] == 'SC_ACT' else 'x'),
                         'iface': None, 'path': None,
                         'lane_line': f"{tr['line'][0]}{tr['line'][1] + 1}" if tr['type'] in ('BUS_SC', 'SC_BUS') else None}
                rt = routes.get(t)
                if rt is not None:
                    u, v, w_if, p_idx = rt
                    entry['iface'] = w_if
                    if p_idx is not None:
                        if w_if in set((net.get('bus') or {}).get('types', ())):
                            entry['line'] = f"{w_if}{p_idx + 1}"
                        path = net['paths'][(min(u, v), max(u, v), w_if)][p_idx]
                        path = path if path[0] == u else path[::-1]
                        entry['path'] = [locations[k].id for k in path]
                flow_route_list.append(entry)
        bus_report = []
        line_limit = []
        for n in (bus or {}).get('types', []):
            lines_n = [b for b in bus['inst'] if b[0] == n]
            if lines_n and all(bus['act'][b].X > 0.5 for b in lines_n):
                line_limit.append(n)
                print(f"[WARN] all {len(lines_n)} {n} bus line(s) in use: more lines might give a cheaper design "
                      f"(configs/buses.json: backbone.bus_lines)")
        for b in (bus or {}).get('inst', []):
            if bus['act'][b].X < 0.5:
                continue
            units = [locations[j].id for j in range(len(locations)) if bus['unit'][b, j].X > 0.5]
            devs = [d for d in bus['devices'] if (b, d) in bus['dev'] and bus['dev'][b, d].X > 0.5]
            st_len, tree = steiner_tree(units + devs, bus['points'])
            bus_report.append({'id': f"{b[0]}{b[1] + 1}", 'type': b[0], 'units': units, 'devices': devs,
                               'asil_group': 'HIGH' if bus['high'][b].X > 0.5 else 'LOW',
                               'length_m': float(bus['len'][b].X), 'steiner_m': st_len, 'tree': [list(e) for e in tree],
                               'device_load': sum(bus['devices'][d]['volume'] for d in devs),
                               'traffic_load': round(line_traffic.get(b, 0.0), 3) if net else 0.0})

        eth_sensor_attachments = {}
        if attach_s:
            for (si, j), av in attach_s.items():
                if av.X > 0.5:
                    eth_sensor_attachments[sensors[si].id] = locations[j].id

        bus_gateways = {}
        for (d, b, j), gv in ((net.get('bus') or {}).get('gw') or {}).items() if net else []:
            if gv.X > 0.5:
                bus_gateways.setdefault(d, {})[f"{b[0]}{b[1] + 1}"] = locations[j].id

        eth_actuator_attachments = {}
        if attach_a:
            for (ai, j), av in attach_a.items():
                if av.X > 0.5:
                    eth_actuator_attachments[actuators[ai].id] = locations[j].id

        solution = {
            'assignment': assignment_map,
            'partitions': partition_map,
            'hardware_cost': hw_cost,
            'hw_cost': hw_cost,
            'interface_cost': if_cost,
            'cable_cost': cable_cost,
            'comm_cost': comm_cost,
            'comm_links': comm_links,
            'comm_link_peak_load': comm_link_peak_load,
            'backbone_routes': backbone_routes,
            'chain_latency': chain_latency,
            'standby': standby_map,
            'failure_scenarios': failure_scenarios,
            'lost_functions': lost_summary,
            'eth_sensor_attachments': eth_sensor_attachments,
            'eth_actuator_attachments': eth_actuator_attachments,
            'bus_gateways': bus_gateways,
            'buses': bus_report,
            'bus_line_limit_reached': line_limit,
            'cable_length': cable_length,
            'peripheral_cable_length': peripheral_cable_length,
            'backbone_length': backbone_length,
            'total_cost': total_cost,
            'num_locations_used': num_locations_used,
            'num_candidate_sites': len(locations),
            'candidate_sites': [loc.id for loc in locations],
            'partition_counts': partition_counts,
            'partition_types': dict(part_type or {}) if partition_type_counts is not None else None,
            'partition_type_counts': partition_type_counts,
            'flow_routes': flow_route_list,
            'num_ecus_used': num_locations_used,
            'num_partitions': num_partitions_opened,
            'hw_features': hw_opened,
            'hw_unit_types': hw_unit_types,
            'interfaces': if_opened,
            'partition_cost': partition_cost,
            'site_cost': site_cost,
            'objective': 'Total Cost',
            'status': 'OPTIMAL'
        }
        return solution

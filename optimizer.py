import gc
import itertools
import os
import random
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from benders import routing_lp_cut, failure_lp_cut
from optimizer_utils import (get_distance, build_sensor_lookup, build_actuator_lookup, nominal_lane, lane_keys,
                             build_cost_map, build_latency_map, extract_solution,
                             sw_multiplier, site_derating, is_fail_operational, with_gradual_standbys, pack_partitions, flow_routes, chain_latencies, steiner_tree,
                             resource_messages, message_kbit, resource_rate, crossing_ms, robust_ms, device_line, device_ms,
                             message_prio, frame_kbit, queue_ms, device_lines)

UNCERTAINTY_BLOCKS = ('sw_margin', 'derating', 'redundancy', 'ai_contention', 'robust_latency', 'growth', 'failures')

MAX_PACKING_CUTS = 5


_GW_DEBUG = os.environ.get('DSE_GW_DEBUG', '') == '1'
_CB_VERIFY = os.environ.get('DSE_CB_VERIFY') == '1'
_CHAIN_DEBUG = os.environ.get('DSE_CHAIN_DEBUG') == '1'


def job_memory_gb():
    v = os.environ.get('SLURM_MEM_PER_NODE')
    if v:
        try:
            return float(v) * 1048576 / 1e9
        except ValueError:
            pass
    try:
        path = open('/proc/self/cgroup').read().strip().split('::', 1)[1].rstrip('/')
        while True:
            try:
                lim = open(f'/sys/fs/cgroup{path}/memory.max').read().strip()
                if lim != 'max':
                    return float(lim) / 1e9
            except OSError:
                pass
            if not path:
                return None
            path = path.rsplit('/', 1)[0]
    except (OSError, IndexError, ValueError):
        return None


def process_memory_gb():
    now = peak = 0.0
    try:
        for line in open('/proc/self/status'):
            if line.startswith('VmRSS:'):
                now = float(line.split()[1]) * 1024 / 1e9
            elif line.startswith('VmHWM:'):
                peak = float(line.split()[1]) * 1024 / 1e9
    except OSError:
        pass
    return now, peak


class AssignmentOptimizer:
    def __init__(self):
        print("Optimizer initialised")
        self.last_values = None
        self._session = None
        self._resumable = False
        self._round_caps = []
        self.memory = {'soft_share': 0.75, 'hard_share': 0.9}

    def calculate_cable_expressions(self, x, scs, locations, sensors, actuators, cable_types, comm_matrix, feasible_locs_per_sc, comm=None, attach_s=None, attach_a=None, bus=None, standby=None):
        cost_map = build_cost_map(cable_types)
        sensor_lookup = build_sensor_lookup(sensors)
        actuator_lookup = build_actuator_lookup(actuators)
        bus_types = set((bus or {}).get('types', ()))
        dev_f = float((bus or {}).get('dev_factor', 1.0) or 1.0)
        cost_terms, dist_terms = [], []

        for i, sc in enumerate(scs):
            devs = [sensor_lookup.get(s_id) for s_id in (sc.sensors or [])] + [actuator_lookup.get(a_id) for a_id in (sc.actuators or [])]
            devs = [d for d in devs if d is not None and d.location and d.interface != 'ETH' and d.interface not in bus_types]
            if not devs:
                continue
            for j in feasible_locs_per_sc[i]:
                dist = sum(get_distance(d.location, locations[j].location) for d in devs)
                cost = dev_f * sum(get_distance(d.location, locations[j].location) * cost_map.get(d.interface, 0.0) for d in devs)
                if dist > 0:
                    cost_terms.append((x[i, j], cost))
                    dist_terms.append((x[i, j], dist))
                    if standby and (i, j) in standby:
                        cost_terms.append((standby[i, j], cost))
                        dist_terms.append((standby[i, j], dist))

        for att, devs in ((attach_s, sensors), (attach_a, actuators)):
            for (di, j), av in (att or {}).items():
                if getattr(devs[di], 'location', None):
                    dist = get_distance(devs[di].location, locations[j].location)
                    cost_terms.append((av, dev_f * dist * cost_map.get(getattr(devs[di], 'interface', 'ETH'), 0.0)))
                    dist_terms.append((av, dist))

        for b, lv in (bus or {}).get('len', {}).items():
            cost_terms.append((lv, dev_f * cost_map.get(b[0], 0.0)))
            dist_terms.append((lv, 1.0))

        for (j1, j2, iface), comm_var in (comm or {}).items():
            dist = get_distance(locations[j1].location, locations[j2].location)
            cost_terms.append((comm_var, dist * cable_types[iface].cost_per_meter))
            dist_terms.append((comm_var, dist))

        cable_cost_expr = gp.quicksum(v * c for v, c in cost_terms) if cost_terms else 0
        cable_distance_expr = gp.quicksum(v * d for v, d in dist_terms) if dist_terms else 0
        return cable_cost_expr, cable_distance_expr, 0

    def _create_model_and_variables(self, n_sc, n_locs, scs, locations, unique_asils, all_hw, sensors, actuators, all_interfaces, infeasible_ij=None, comm_matrix=None, partitions=None, backbone=None, cable_types=None, uncertainty=None, chains=None):
        model = gp.Model("ECU_Assignment")
        model.setParam('OutputFlag', 1)
        model.setParam('MIPFocus', 2)
        model.setParam("Seed", 0)
        model.update()

        partitions = partitions or {}
        _mpa, _smp = partitions.get('max_per_asil_per_site', 3), partitions.get('site_max_partitions')
        max_p_per_asil = int(_mpa) if _mpa is not None else (int(_smp) if _smp is not None else 10 ** 6)

        infeasible_ij = set(infeasible_ij or set())
        max_len = {n: float(getattr(ct, 'max_length_m', float('inf')) or float('inf')) for n, ct in (cable_types or {}).items()}
        shared_types = {n for n in ('CAN', 'LIN', 'FLEXRAY') if n in (cable_types or {})}
        _s_look, _a_look = build_sensor_lookup(sensors), build_actuator_lookup(actuators)
        n_reach = 0
        for i, sc in enumerate(scs):
            devs = [_s_look.get(d) for d in (getattr(sc, 'sensors', None) or [])] + [_a_look.get(d) for d in (getattr(sc, 'actuators', None) or [])]
            devs = [d for d in devs if d is not None and d.location and d.interface != 'ETH' and d.interface not in shared_types]
            for d in devs:
                for j in range(n_locs):
                    if get_distance(d.location, locations[j].location) > max_len.get(d.interface, float('inf')) + 1e-9 and (i, j) not in infeasible_ij:
                        infeasible_ij.add((i, j))
                        n_reach += 1
        if n_reach:
            print(f"  Reach: {n_reach} placements excluded (device cable longer than its type's max_length_m)")
        n_tpl = 0
        for i, sc in enumerate(scs):
            ok = getattr(sc, 'allowed_cells', None)
            if ok:
                for j in range(n_locs):
                    if locations[j].id not in ok and (i, j) not in infeasible_ij:
                        infeasible_ij.add((i, j))
                        n_tpl += 1
        if n_tpl:
            print(f"  Template: {n_tpl} placements excluded (cells outside the component's group)")
        feasible_locs_per_sc = {i: [] for i in range(n_sc)}
        feasible_scs_per_loc = {j: [] for j in range(n_locs)}
        x = {}
        for i in range(n_sc):
            for j in range(n_locs):
                if (i, j) in infeasible_ij:
                    continue
                feasible_locs_per_sc[i].append(j)
                feasible_scs_per_loc[j].append(i)
                x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{scs[i].id}_{locations[j].id}")

        for i in range(n_sc):
            if not feasible_locs_per_sc[i]:
                raise ValueError(f"SC {scs[i].id} has no feasible locations after latency precompute")

        P = {}
        Q = {}
        types = partitions.get('types') or None
        for j in range(n_locs):
            for a in unique_asils:
                n_sc_a = sum(1 for i in feasible_scs_per_loc[j] if scs[i].asil_req == a)
                ub = min(max_p_per_asil, n_sc_a)
                P[j, a] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=ub, name=f"P_{locations[j].id}_asil{a}")
                if types:
                    for t, ty in enumerate(types):
                        Q[j, a, t] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=ub, name=f"Q_{locations[j].id}_asil{a}_{ty['id']}")
                    model.addConstr(P[j, a] == gp.quicksum(Q[j, a, t] for t in range(len(types))),
                                    name=f"partition_types_{locations[j].id}_asil{a}")

        y = {j: model.addVar(vtype=GRB.BINARY, name=f"y_{locations[j].id}") for j in range(n_locs)}

        fail_on = bool((uncertainty or {}).get('failures'))
        fail_cfg = (uncertainty or {}).get('params', {}).get('failures', {}) if fail_on else {}
        r, hot, hr, critical = {}, {}, {}, []
        if fail_on:
            asil_min = int(fail_cfg.get('critical_asil_min', 3))
            recovery = float(fail_cfg.get('recovery_ms', 50.0))
            critical = [i for i in range(n_sc) if is_fail_operational(scs[i], fail_cfg)]
            ftti = {}
            for ch in (chains or []):
                for sid in ch.get('swcs', []):
                    ftti[sid] = min(ftti.get(sid, float('inf')), float(ch.get('ftti_ms', float('inf'))))
            for i in critical:
                cold_ok = ftti.get(scs[i].id, float('inf')) >= recovery
                hot[i] = model.addVar(vtype=GRB.BINARY, name=f"hot_{scs[i].id}") if cold_ok else None
                for j in feasible_locs_per_sc[i]:
                    r[i, j] = model.addVar(vtype=GRB.BINARY, name=f"r_{scs[i].id}_{locations[j].id}")
                    hr[i, j] = (model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"hr_{scs[i].id}_{locations[j].id}")
                                if cold_ok else r[i, j])
            print(f"  Failures: {len(critical)} critical SCs (fail_operational flag, else ASIL >= {asil_min}), cold standby admissible for "
                  f"{sum(1 for i in critical if hot[i] is not None)} of them (recovery {recovery:.0f} ms)")

        hw_cap = (partitions or {}).get('hw_capacity', {}) or {}
        hw_use = {}
        for j in range(n_locs):
            for h in all_hw:
                if h in hw_cap:
                    hw_use[j, h] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=int(hw_cap[h].get('max_units', 1)), name=f"hw_{locations[j].id}_{h}")
                else:
                    hw_use[j, h] = model.addVar(vtype=GRB.BINARY, name=f"hw_{locations[j].id}_{h}")
        hw_t = {}
        for j in range(n_locs):
            for h in all_hw:
                for t, ty in enumerate((hw_cap.get(h) or {}).get('types') or []):
                    hw_t[j, h, t] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=int(hw_cap[h].get('max_units', 1)),
                                                 name=f"hwt_{locations[j].id}_{h}_{ty['id']}")
        for (j, h) in hw_use:
            ts = [v for (jj, hh, _t), v in hw_t.items() if jj == j and hh == h]
            if ts:
                model.addConstr(hw_use[j, h] == gp.quicksum(ts), name=f"hw_types_{locations[j].id}_{h}")
        for h in all_hw:
            cap_total = (hw_cap.get(h) or {}).get('max_units_total')
            if cap_total is not None:
                model.addConstr(gp.quicksum(hw_use[j, h] for j in range(n_locs)) <= int(cap_total), name=f"hw_total_{h}")
                print(f"  Hardware cap: at most {int(cap_total)} {h} unit(s) in the whole car")

        if_use = {}
        for j in range(n_locs):
            for i_name in all_interfaces:
                if_use[j, i_name] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"if_{locations[j].id}_{i_name}")

        sensor_id_to_idx = {s.id: si for si, s in enumerate(sensors)}
        actuator_id_to_idx = {a.id: ai for ai, a in enumerate(actuators)}

        eth_sensor_indices = [si for si, s in enumerate(sensors) if getattr(s, 'interface', None) == 'ETH']
        eth_actuator_indices = [ai for ai, a in enumerate(actuators) if getattr(a, 'interface', None) == 'ETH']

        attach_s = {}
        for si in eth_sensor_indices:
            for j in range(n_locs):
                if sensors[si].location and get_distance(sensors[si].location, locations[j].location) > max_len.get('ETH', float('inf')) + 1e-9:
                    continue
                attach_s[si, j] = model.addVar(vtype=GRB.BINARY, name=f"attachS_{sensors[si].id}_{locations[j].id}")

        attach_a = {}
        for ai in eth_actuator_indices:
            for j in range(n_locs):
                if actuators[ai].location and get_distance(actuators[ai].location, locations[j].location) > max_len.get('ETH', float('inf')) + 1e-9:
                    continue
                attach_a[ai, j] = model.addVar(vtype=GRB.BINARY, name=f"attachA_{actuators[ai].id}_{locations[j].id}")

        bus_types = [n for n in ('CAN', 'LIN', 'FLEXRAY') if n in all_interfaces]
        bus_cfg = dict((backbone or {}).get('bus_lines', {}))
        bus = {'types': [], 'inst': [], 'act': {}, 'len': {}, 'high': {}, 'unit': {}, 'dev': {}, 'devices': {},
               'points': {**{loc.id: loc.location for loc in locations}, **{d.id: d.location for d in list(sensors) + list(actuators)}}}
        for kind, devs in (('sensor', sensors), ('actuator', actuators)):
            for dev in devs:
                if getattr(dev, 'interface', None) not in bus_types:
                    continue
                users = [i for i, sc in enumerate(scs) if dev.id in (sc.sensors or []) + (sc.actuators or [])]
                bus['devices'][dev.id] = {'iface': dev.interface, 'kind': kind, 'users': users,
                                          'volume': float(getattr(dev, 'volume', 0.0) or 0.0),
                                          'kbit': float(getattr(dev, 'volume', 0.0) or 0.0) * 1000.0 / float(getattr(dev, 'rate_hz', 10.0) or 10.0),
                                          'group': 'HIGH' if any(int(scs[i].asil_req) >= 3 for i in users) else 'LOW'}
        bus['all_types'] = list(bus_types)
        for w_if in bus_types:
            devs = [d for d, info in bus['devices'].items() if info['iface'] == w_if]
            n_lines = int(bus_cfg.get(w_if, max(1, len(devs))))
            if n_lines > 0:
                bus['types'].append(w_if)
            for k in range(n_lines):
                b = (w_if, k)
                bus['inst'].append(b)
                bus['act'][b] = model.addVar(vtype=GRB.BINARY, name=f"bus_{w_if}{k + 1}")
                bus['len'][b] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=max_len.get(w_if, float('inf')), name=f"busLen_{w_if}{k + 1}")
                bus.setdefault('len_ub', {})[b] = max_len.get(w_if, float('inf'))
                bus['high'][b] = model.addVar(vtype=GRB.BINARY, name=f"busHigh_{w_if}{k + 1}")
                for j in range(n_locs):
                    bus['unit'][b, j] = model.addVar(vtype=GRB.BINARY, name=f"busUnit_{w_if}{k + 1}_{locations[j].id}")
                for d in devs:
                    bus['dev'][b, d] = model.addVar(vtype=GRB.BINARY, name=f"busDev_{w_if}{k + 1}_{d}")
        bus['gateway'] = bool((backbone or {}).get('bus_gateway', False))
        bus['dev_factor'] = float((backbone or {}).get('device_cable_factor', 1.0) or 1.0)
        bus['gw'] = {}
        if bus['gateway']:
            for d, info in bus['devices'].items():
                if not info['users']:
                    continue
                for b in bus['inst']:
                    if b[0] != info['iface']:
                        continue
                    for j in range(n_locs):
                        if (b, j) in bus['unit']:
                            bus['gw'][d, b, j] = model.addVar(vtype=GRB.BINARY, name=f"busGw_{b[0]}{b[1] + 1}_{d}_{locations[j].id}")
            print(f"  Bus gateway: on, {len(bus['gw'])} gateway variables")
        if bus['inst']:
            print(f"  Bus lines: " + ", ".join(f"{n} {sum(1 for b in bus['inst'] if b[0] == n)} line(s) for "
                                               f"{sum(1 for d in bus['devices'].values() if d['iface'] == n)} device(s)" for n in bus['types']))

        bb_ifaces = [n for n in all_interfaces if n not in bus['all_types'] or n in bus['types']]
        cap_of = {n: float(getattr((cable_types or {}).get(n), 'capacity', float('inf'))) for n in bb_ifaces}
        comm = {}
        for j1 in range(n_locs):
            for j2 in range(j1 + 1, n_locs):
                for w_if in bb_ifaces:
                    if w_if in bus['types']:
                        continue
                    comm[j1, j2, w_if] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"comm_{locations[j1].id}_{locations[j2].id}_{w_if}")

        traffic_flows = []
        t_idx = 0

        sc_id_map = {sc.id: i for i, sc in enumerate(scs)}
        if comm_matrix:
            for link in comm_matrix:
                src_id = link.get('src')
                dst_id = link.get('dst')
                vol = link.get('volume', 0)
                if vol > 0 and src_id in sc_id_map and dst_id in sc_id_map:
                    traffic_flows.append({
                        'id': t_idx,
                        'type': 'SC_SC',
                        'src_idx': sc_id_map[src_id],
                        'dst_idx': sc_id_map[dst_id],
                        'volume': vol,
                        'kbit': float(link.get('kbit', 0.0)),
                    })
                    t_idx += 1

        sensor_lookup = build_sensor_lookup(sensors)
        for dst_sc_idx, sc in enumerate(scs):
            for s_id in (getattr(sc, 'sensors', None) or []):
                s_obj = sensor_lookup.get(s_id)
                if not s_obj or getattr(s_obj, 'interface', None) != 'ETH':
                    continue
                vol = getattr(s_obj, 'volume', 0) or 0
                if vol <= 0:
                    continue
                s_idx = sensor_id_to_idx.get(s_id)
                if s_idx is None:
                    continue
                traffic_flows.append({
                    'id': t_idx,
                    'type': 'SENS_SC',
                    'sensor_idx': s_idx,
                    'dst_idx': dst_sc_idx,
                    'volume': vol,
                    'kbit': vol * 1000.0 / float(getattr(s_obj, 'rate_hz', 10.0) or 10.0),
                })
                t_idx += 1

        actuator_lookup = build_actuator_lookup(actuators)
        for src_sc_idx, sc in enumerate(scs):
            for a_id in (getattr(sc, 'actuators', None) or []):
                a_obj = actuator_lookup.get(a_id)
                if not a_obj or getattr(a_obj, 'interface', None) != 'ETH':
                    continue
                vol = getattr(a_obj, 'volume', 0) or 0
                if vol <= 0:
                    continue
                a_idx = actuator_id_to_idx.get(a_id)
                if a_idx is None:
                    continue
                traffic_flows.append({
                    'id': t_idx,
                    'type': 'SC_ACT',
                    'src_idx': src_sc_idx,
                    'act_idx': a_idx,
                    'volume': vol,
                    'kbit': vol * 1000.0 / float(getattr(a_obj, 'rate_hz', 10.0) or 10.0),
                })
                t_idx += 1

        if bus['gateway']:
            for d, info in bus['devices'].items():
                if not info['users'] or float(info['volume']) <= 0:
                    continue
                lines = [b for b in bus['inst'] if b[0] == info['iface']]
                for i in info['users']:
                    base_id = None
                    for k, b in enumerate(lines):
                        tr = {'id': t_idx, 'type': 'BUS_SC' if info['kind'] == 'sensor' else 'SC_BUS', 'dev': d, 'line': b, 'lane': k,
                              'volume': float(info['volume']), 'kbit': float(info['kbit'])}
                        if info['kind'] == 'sensor':
                            tr.update({'dst_idx': i, 'src_inst': f'a{k}', 'dst_inst': 'x'})
                        else:
                            tr.update({'src_idx': i, 'src_inst': 'x', 'dst_inst': f'a{k}'})
                        if base_id is None:
                            base_id = t_idx
                        else:
                            tr['base'] = base_id
                        traffic_flows.append(tr)
                        t_idx += 1
            print(f"  Bus gateway flows: {sum(1 for tr in traffic_flows if tr['type'] in ('BUS_SC', 'SC_BUS'))} lane flows of "
                  f"{sum(1 for tr in traffic_flows if tr['type'] in ('BUS_SC', 'SC_BUS') and 'base' not in tr)} device-user pairs")

        if critical:
            crit = set(critical)
            for tr in list(traffic_flows):
                variants = []
                if tr['type'] == 'SC_SC':
                    s, d = tr['src_idx'], tr['dst_idx']
                    if s in crit:
                        variants.append(('r', 'x'))
                    if d in crit:
                        variants.append(('x', 'r'))
                    if s in crit and d in crit:
                        variants.append(('r', 'r'))
                elif tr['type'] == 'SENS_SC' and tr['dst_idx'] in crit:
                    variants.append(('a', 'r'))
                elif tr['type'] == 'SC_ACT' and tr['src_idx'] in crit:
                    variants.append(('r', 'a'))
                elif tr['type'] == 'BUS_SC' and tr['dst_idx'] in crit:
                    variants.append((tr['src_inst'], 'r'))
                elif tr['type'] == 'SC_BUS' and tr['src_idx'] in crit:
                    variants.append(('r', tr['dst_inst']))
                for si, di in variants:
                    traffic_flows.append({**tr, 'id': t_idx, 'src_inst': si, 'dst_inst': di, 'base': tr.get('base', tr['id'])})
                    t_idx += 1
            print(f"  Standby flows: {sum(1 for tr in traffic_flows if 'base' in tr)} added to {sum(1 for tr in traffic_flows if 'base' not in tr)} base flows")

        import hashlib
        seen_keys = {}
        for tr in traffic_flows:
            typ = tr['type']
            if typ == 'SC_SC':
                ends = f"{scs[tr['src_idx']].id}>{scs[tr['dst_idx']].id}"
            elif typ == 'SENS_SC':
                ends = f"S:{sensors[tr['sensor_idx']].id}>{scs[tr['dst_idx']].id}"
            elif typ == 'SC_ACT':
                ends = f"{scs[tr['src_idx']].id}>A:{actuators[tr['act_idx']].id}"
            elif typ == 'BUS_SC':
                ends = f"B:{tr['dev']}@{tr['line'][0]}{tr['line'][1]}>{scs[tr['dst_idx']].id}"
            else:
                ends = f"{scs[tr['src_idx']].id}>B:{tr['dev']}@{tr['line'][0]}{tr['line'][1]}"
            key = f"{typ}|{ends}|{tr.get('src_inst', 'x')}{tr.get('dst_inst', 'x')}"
            k = seen_keys.get(key, 0)
            seen_keys[key] = k + 1
            tr['key'] = hashlib.sha1(f"{key}#{k}".encode()).hexdigest()[:12]

        k_paths = max(1, int((backbone or {}).get('candidate_paths', 3)))
        w, flow_elig = {}, {}
        for tr in traffic_flows:
            vol = float(tr['volume'])
            elig = [n for n in bb_ifaces if vol <= cap_of[n]] or [max(bb_ifaces, key=lambda n: cap_of[n])]
            flow_elig[tr['id']] = elig
            if len(elig) > 1:
                for n in elig:
                    w[tr['id'], n] = model.addVar(vtype=GRB.BINARY, name=f"w_{tr['key']}_{n}")
        pair_vol, flow_ends = {}, {}
        for tr in traffic_flows:
            si, di = tr.get('src_inst', 'x'), tr.get('dst_inst', 'x')
            if tr['type'] == 'SC_SC':
                src = {u: (r if si == 'r' else x)[tr['src_idx'], u] for u in feasible_locs_per_sc[tr['src_idx']]}
                dst = {v: (r if di == 'r' else x)[tr['dst_idx'], v] for v in feasible_locs_per_sc[tr['dst_idx']]}
            elif tr['type'] == 'SENS_SC':
                src = {u: attach_s[tr['sensor_idx'], u] for u in range(n_locs) if (tr['sensor_idx'], u) in attach_s}
                dst = {v: (r if di == 'r' else x)[tr['dst_idx'], v] for v in feasible_locs_per_sc[tr['dst_idx']]}
            elif tr['type'] == 'BUS_SC':
                src = {u: bus['gw'][tr['dev'], tr['line'], u] for u in range(n_locs) if (tr['dev'], tr['line'], u) in bus['gw']}
                dst = {v: (r if di == 'r' else x)[tr['dst_idx'], v] for v in feasible_locs_per_sc[tr['dst_idx']]}
            elif tr['type'] == 'SC_BUS':
                src = {u: (r if si == 'r' else x)[tr['src_idx'], u] for u in feasible_locs_per_sc[tr['src_idx']]}
                dst = {v: bus['gw'][tr['dev'], tr['line'], v] for v in range(n_locs) if (tr['dev'], tr['line'], v) in bus['gw']}
            else:
                src = {u: (r if si == 'r' else x)[tr['src_idx'], u] for u in feasible_locs_per_sc[tr['src_idx']]}
                dst = {v: attach_a[tr['act_idx'], v] for v in range(n_locs) if (tr['act_idx'], v) in attach_a}
            flow_ends[tr['id']] = (src, dst)
            for u in src:
                for v in dst:
                    if u != v:
                        pair_vol.setdefault((u, v), []).append(float(tr['volume']))

        site_dist = {(u, v): get_distance(locations[u].location, locations[v].location)
                     for u in range(n_locs) for v in range(n_locs) if u != v}
        paths, sel = {}, {}
        for (u, v) in sorted({(min(a, b), max(a, b)) for (a, b) in pair_vol}):
            relays = sorted((r for r in range(n_locs) if r not in (u, v)), key=lambda r: site_dist[u, r] + site_dist[r, v])
            for w_if in bb_ifaces:
                if w_if in bus['types']:
                    paths[u, v, w_if] = [[u, v] for b in bus['inst'] if b[0] == w_if]
                else:
                    reach = max_len.get(w_if, float('inf')) + 1e-9
                    direct = [[u, v]] if site_dist[u, v] <= reach else []
                    relay = [[u, r, v] for r in relays if site_dist[u, r] <= reach and site_dist[r, v] <= reach][:k_paths - 1] if w_if == 'ETH' else []
                    paths[u, v, w_if] = direct + relay
                for p in range(len(paths[u, v, w_if])):
                    sel[u, v, w_if, p] = model.addVar(vtype=GRB.BINARY, name=f"sel_{locations[u].id}_{locations[v].id}_{w_if}_p{p}")
        net = {'Q': Q, 'types': types, 'hw_t': hw_t,
               'paths': paths, 'sel': sel, 'flow_ends': flow_ends, 'bus': bus,
               'bb_ifaces': bb_ifaces, 'cap_of': cap_of, 'w': w, 'flow_elig': flow_elig,
               'r': r, 'hot': hot, 'hr': hr, 'critical': critical, 'fail_cfg': fail_cfg,
               'x_vars': x, 'y_vars': y, 'switch_vars': {j: hw_use.get((j, 'SWITCH')) for j in range(n_locs)},
               'flow_meta': {tr['id']: tr for tr in traffic_flows},
               'k_paths': k_paths, 'volumes': {tr['id']: float(tr['volume']) for tr in traffic_flows},
               'kbit': {tr['id']: float(tr.get('kbit', 0.0)) for tr in traffic_flows},
               'site_ids': [loc.id for loc in locations],
               'hop_ms': {n: float(getattr((cable_types or {}).get(n), 'hop_latency_ms', 0.0)) for n in bb_ifaces},
               'path_link_cut': bool((backbone or {}).get('path_link_cut', False)),
               'upfront_paths': bool((backbone or {}).get('upfront_paths', False)),
               'bus_tree_bounds': str((backbone or {}).get('bus_tree_bounds', 'full') or 'full'),
               'cell_cut_capacity': bool((backbone or {}).get('cell_cut_capacity', False)),
               'benders': bool((backbone or {}).get('benders_cuts', False)),
               'benders_fail': bool((backbone or {}).get('benders_failure_cuts', False)),
               'bus_umax': float((backbone or {}).get('bus_max_utilization', 1.0))}

        var_stats = {
            'x': len(x),
            'P': len(P) + len(Q),
            'y': len(y),
            'hw_use': len(hw_use),
            'if_use': len(if_use),
            'attach_s': len(attach_s),
            'attach_a': len(attach_a),
            'bus': len(bus['act']) + len(bus['len']) + len(bus['high']) + len(bus['unit']) + len(bus['dev']),
            'comm': len(comm),
            'traffic_flows': len(traffic_flows),
            'sel': len(sel),
            'w': len(w),
            'r': len(r),
            'hot': sum(1 for v in hot.values() if v is not None),
            'pairs_with_paths': len(paths),
        }
        var_stats['total'] = (
            var_stats['x']
            + var_stats['P']
            + var_stats['y']
            + var_stats['hw_use']
            + var_stats['if_use']
            + var_stats['attach_s']
            + var_stats['attach_a']
            + var_stats['bus']
            + var_stats['comm']
            + var_stats['sel']
            + var_stats['w']
            + var_stats['r']
            + var_stats['hot']
        )

        return model, x, P, y, hw_use, if_use, attach_s, attach_a, bus, comm, traffic_flows, net, feasible_locs_per_sc, feasible_scs_per_loc, var_stats

    def _add_constraints(self, model, x, P, y, hw_use, if_use, attach_s, attach_a, bus, comm, traffic_flows, net, scs, locations, sensors, actuators,
                              cable_types, comm_matrix, partitions, unique_asils,
                              feasible_locs_per_sc, feasible_scs_per_loc, enable_comm_bw_constraints=True,
                              comm_bw_big_m=None, uncertainty=None):
        n_sc = len(scs)
        n_locs = len(locations)
        all_interfaces = list(cable_types.keys())
        all_hw = sorted(list(set(h for (j, h) in hw_use.keys())))

        print(f"\nAdding constraints...")

        cstats = {
            'c1a_unique_site': 0,
            'c1b_x_le_y': 0,
            'c1c_partitions_le_scs': 0,
            'c1d_site_caps': 0,
            'c1e_x_le_P': 0,
            'c1g_standby': 0,
            'c1h_cold_capacity': 0,
            'c2_capacity': 0,
            'c3_hw_required': 0,
            'c4_if_required': 0,
            'c5_redundancy': 0,
            'c6_ai_contention': 0,
            'c9_if_activation': 0,
            'c9_attach': 0,
            'c11_bus': 0,
            'c10_pair_traffic': 0,
            'c10_path_choice': 0,
            'c10_relay_switch': 0,
        }

        def x_expr(i, j):
            return x.get((i, j), 0)

        unc = uncertainty or {}
        unc_params = unc.get('params', {})
        print(f"  Uncertainty blocks: " + ", ".join(f"{k}={bool(unc.get(k))}" for k in UNCERTAINTY_BLOCKS))
        if unc.get('growth'):
            print(f"  Robustness level rho = {float(unc.get('rho', 1.0)):.2f} (growth profiles {unc.get('params', {}).get('growth', {}).get('profiles')})")

        def get_sw_multiplier(sc, res='cpu'):
            return sw_multiplier(sc, unc, res)

        types = net.get('types')
        Q = net.get('Q') or {}
        res_m = ('cpu', 'ram', 'rom', 'tops') if types and any(ty.get('tops_cap', 0) > 0 for ty in types) else ('cpu', 'ram', 'rom')

        def der(res, H):
            return H if res in ('cpu', 'tops') else 1.0

        def cap_expr(j, a, res):
            if types:
                return gp.quicksum(Q[j, a, t] * ty[f'{res}_cap'] for t, ty in enumerate(types) if ty[f'{res}_cap'] > 0)
            return P[j, a] * partitions.get(f'{res}_cap', float('inf'))

        def req(i, res):
            return float(getattr(scs[i], f'{res}_req', 0) or 0) * get_sw_multiplier(scs[i], res)

        def hosts(i, j):
            if not types:
                return P[j, scs[i].asil_req]
            H = site_derating(locations[j], unc)
            fit = [t for t, ty in enumerate(types) if all(req(i, r) <= ty[f'{r}_cap'] * der(r, H) + 1e-9 for r in res_m)]
            if len(fit) == len(types):
                return P[j, scs[i].asil_req]
            return gp.quicksum(Q[j, scs[i].asil_req, t] for t in fit)

        total_traffic_volume = sum(float(tr['volume']) for tr in traffic_flows)
        comm_ub = {n: max(2, int(np.ceil(total_traffic_volume / cap))) if cap < float('inf') else 2 for n, cap in net['cap_of'].items()}
        max_len_bb = {n: float(getattr(ct, 'max_length_m', float('inf')) or float('inf')) for n, ct in (cable_types or {}).items()}
        n_out = 0
        for (_j1, _j2, w_if), comm_var in comm.items():
            within = get_distance(locations[_j1].location, locations[_j2].location) <= max_len_bb.get(w_if, float('inf')) + 1e-9
            comm_var.UB = comm_ub[w_if] if within else 0
            n_out += 0 if within else 1
        if n_out:
            print(f"  Reach: {n_out} backbone cable options out of reach (max_length_m)")
        net['comm_ub'] = comm_ub
        print(f"  Backbone cable upper bounds per type: {comm_ub} (total traffic {total_traffic_volume:.1f})")

        for i in range(n_sc):
            model.addConstr(gp.quicksum(x[i, j] for j in feasible_locs_per_sc[i]) == 1, name=f"unique_site_{scs[i].id}")
            cstats['c1a_unique_site'] += 1

        for j in range(n_locs):
            for i in feasible_scs_per_loc[j]:
                model.addConstr(x[i, j] <= y[j], name=f"x_le_y_{scs[i].id}_{locations[j].id}")
                cstats['c1b_x_le_y'] += 1
            model.addConstr(y[j] <= gp.quicksum(x[i, j] for i in feasible_scs_per_loc[j])
                            + gp.quicksum(net['r'][i, j] for i in net['critical'] if (i, j) in net['r']), name=f"y_le_sum_x_{locations[j].id}")
            cstats['c1b_x_le_y'] += 1

        for j in range(n_locs):
            model.addConstr(gp.quicksum(P[j, a] for a in unique_asils) >= y[j], name=f"site_has_partition_{locations[j].id}")
            cstats['c1b_x_le_y'] += 1

        for j in range(n_locs):
            for a in unique_asils:
                model.addConstr(
                    P[j, a] <= gp.quicksum(x[i, j] for i in feasible_scs_per_loc[j] if scs[i].asil_req == a)
                    + gp.quicksum(net['r'][i, j] for i in net['critical'] if (i, j) in net['r'] and scs[i].asil_req == a),
                    name=f"partitions_le_scs_{locations[j].id}_asil{a}"
                )
                cstats['c1c_partitions_le_scs'] += 1

        for (i, j), xv in x.items():
            model.addConstr(xv <= hosts(i, j), name=f"x_le_P_{scs[i].id}_{locations[j].id}")
            cstats['c1e_x_le_P'] += 1

        def sb_var(i, j, res):
            return net['r'][i, j] if res == 'rom' else net['hr'][i, j]

        site_max_p = partitions.get('site_max_partitions')
        for j in range(n_locs):
            p_sum = gp.quicksum(P[j, a] for a in unique_asils)
            if site_max_p is not None:
                model.addConstr(p_sum <= float(site_max_p) * y[j], name=f"site_max_partitions_{locations[j].id}")
                cstats['c1d_site_caps'] += 1
            H_j = site_derating(locations[j], unc)
            for res in ('cpu', 'ram', 'rom'):
                site_cap = partitions.get(f'site_{res}_cap')
                if site_cap is None:
                    continue
                demand = gp.quicksum(
                    x[i, j] * getattr(scs[i], f'{res}_req') * get_sw_multiplier(scs[i], res) for i in feasible_scs_per_loc[j]
                ) + gp.quicksum(
                    sb_var(i, j, res) * getattr(scs[i], f'{res}_req') * get_sw_multiplier(scs[i], res)
                    for i in feasible_scs_per_loc[j] if (i, j) in net['hr']
                )
                model.addConstr(demand <= float(site_cap) * (H_j if res == 'cpu' else 1.0) * y[j], name=f"site_{res}_cap_{locations[j].id}")
                cstats['c1d_site_caps'] += 1

        for j in range(n_locs):
            H_j = site_derating(locations[j], unc)
            for a in unique_asils:
                scs_ja = [i for i in feasible_scs_per_loc[j] if scs[i].asil_req == a]
                for res in res_m:
                    users = [i for i in scs_ja if req(i, res) > 0] if res == 'tops' else scs_ja
                    if not users:
                        continue
                    demand = gp.quicksum(
                        x[i, j] * getattr(scs[i], f'{res}_req') * get_sw_multiplier(scs[i], res) for i in users
                    ) + gp.quicksum(
                        sb_var(i, j, res) * getattr(scs[i], f'{res}_req') * get_sw_multiplier(scs[i], res)
                        for i in users if (i, j) in net['hr']
                    )
                    model.addConstr(
                        demand <= cap_expr(j, a, res) * der(res, H_j),
                        name=f"{res}_capacity_{locations[j].id}_asil{a}"
                    )
                    cstats['c2_capacity'] += 1
                big_cap = ({r: max(ty[f'{r}_cap'] for ty in types) for r in res_m} if types
                           else {r: float(partitions.get(f'{r}_cap', float('inf'))) for r in res_m})

                def _big(i):
                    return any(req(i, res) > 0.5 * big_cap[res] * der(res, H_j) for res in res_m if big_cap[res] > 0)
                if types:
                    for s_t, ty_s in enumerate(types):
                        under = {t for t, ty in enumerate(types) if all(ty[f'{r}_cap'] <= ty_s[f'{r}_cap'] for r in res_m)}
                        others = [t for t in range(len(types)) if t not in under]
                        if not others:
                            continue
                        nofit = [i for i in scs_ja if any(req(i, r) > ty_s[f'{r}_cap'] * der(r, H_j) + 1e-9 for r in res_m)]
                        if not nofit:
                            continue
                        for res in res_m:
                            users = [i for i in nofit if req(i, res) > 0]
                            if not users:
                                continue
                            demand = gp.quicksum(x[i, j] * req(i, res) for i in users) + gp.quicksum(
                                sb_var(i, j, res) * req(i, res) for i in users if (i, j) in net['hr'])
                            model.addConstr(demand <= gp.quicksum(Q[j, a, t] * types[t][f'{res}_cap'] for t in others if types[t][f'{res}_cap'] > 0) * der(res, H_j),
                                            name=f"size_class_{res}_{locations[j].id}_asil{a}_{ty_s['id']}")
                            cstats['c2_capacity'] += 1
                big = [i for i in scs_ja if _big(i)]
                if big:
                    big_rom = [i for i in big if big_cap['rom'] > 0 and req(i, 'rom') > 0.5 * big_cap['rom'] * der('rom', H_j)]
                    model.addConstr(P[j, a] >= gp.quicksum(x[i, j] for i in big)
                                    + gp.quicksum(net['hr'][i, j] for i in big if (i, j) in net['hr'] and i not in big_rom)
                                    + gp.quicksum(net['r'][i, j] for i in big_rom if (i, j) in net['hr']),
                                    name=f"big_items_{locations[j].id}_asil{a}")
                    cstats['c2_capacity'] += 1
                if types:
                    for res in res_m:
                        fits_of = {}
                        for i in scs_ja:
                            fit = [t for t, ty in enumerate(types) if all(req(i, r) <= ty[f'{r}_cap'] * der(r, H_j) + 1e-9 for r in res_m)]
                            if fit and all(req(i, res) > 0.5 * types[t][f'{res}_cap'] * der(res, H_j) for t in fit):
                                fits_of[i] = fit
                        if fits_of:
                            union = sorted({t for f in fits_of.values() for t in f})
                            model.addConstr(gp.quicksum(Q[j, a, t] for t in union) >= gp.quicksum(x[i, j] for i in fits_of)
                                            + gp.quicksum(sb_var(i, j, res) for i in fits_of if (i, j) in net['hr']),
                                            name=f"big_items_{res}_{locations[j].id}_asil{a}")
                            cstats['c2_capacity'] += 1

        for i in net['critical']:
            model.addConstr(gp.quicksum(net['r'][i, j] for j in feasible_locs_per_sc[i]) == 1, name=f"standby_{scs[i].id}")
            for j in feasible_locs_per_sc[i]:
                rv = net['r'][i, j]
                model.addConstr(rv + x[i, j] <= 1, name=f"standby_other_cell_{scs[i].id}_{locations[j].id}")
                model.addConstr(rv <= y[j], name=f"standby_site_{scs[i].id}_{locations[j].id}")
                model.addConstr(rv <= hosts(i, j), name=f"standby_partition_{scs[i].id}_{locations[j].id}")
                for h in (scs[i].hw_required or []):
                    model.addConstr(rv <= hw_use[j, h], name=f"standby_hw_{scs[i].id}_{h}_{locations[j].id}")
                if net['hot'][i] is not None:
                    hv = net['hr'][i, j]
                    model.addConstr(hv <= rv, name=f"hr_le_r_{scs[i].id}_{locations[j].id}")
                    model.addConstr(hv <= net['hot'][i], name=f"hr_le_hot_{scs[i].id}_{locations[j].id}")
                    model.addConstr(hv >= rv + net['hot'][i] - 1, name=f"hr_ge_{scs[i].id}_{locations[j].id}")
                cstats['c1g_standby'] += 1

        cold = [i for i in net['critical'] if net['hot'][i] is not None]
        act = {}
        net['act'] = act
        for i in cold:
            for c in feasible_locs_per_sc[i]:
                for j in feasible_locs_per_sc[i]:
                    if j != c and (i, j) in net['r']:
                        act[i, c, j] = model.addVar(lb=0.0, ub=1.0, name=f"act_{scs[i].id}_{locations[c].id}_{locations[j].id}")
                        model.addConstr(act[i, c, j] >= x[i, c] + net['r'][i, j] - 1 - net['hot'][i],
                                        name=f"act_{scs[i].id}_{locations[c].id}_{locations[j].id}")
                        cstats['c1h_cold_capacity'] += 1
        if act:
            for j in range(n_locs):
                H_j = site_derating(locations[j], unc)
                for res in res_m:
                    if res == 'rom':
                        continue
                    dem = {i: getattr(scs[i], f'{res}_req') * get_sw_multiplier(scs[i], res) for i in feasible_scs_per_loc[j]}
                    if res == 'tops' and not any(dem[i] > 0 for i in cold if i in dem):
                        continue
                    hosted = {}
                    for a in unique_asils:
                        hosted[a] = model.addVar(lb=0.0, name=f"hosted_{res}_{locations[j].id}_asil{a}")
                        model.addConstr(hosted[a] == gp.quicksum(dem[i] * x[i, j] for i in feasible_scs_per_loc[j] if scs[i].asil_req == a)
                                        + gp.quicksum(dem[i] * net['hr'][i, j] for i in feasible_scs_per_loc[j]
                                                      if (i, j) in net['hr'] and scs[i].asil_req == a),
                                        name=f"hosted_{res}_{locations[j].id}_asil{a}")
                    part_cap = partitions.get(f'{res}_cap', float('inf'))
                    site_cap = partitions.get(f'site_{res}_cap')
                    for c in range(n_locs):
                        woken = [(i, act[i, c, j]) for i in cold if (i, c, j) in act]
                        if not woken:
                            continue
                        for a in unique_asils:
                            terms = [dem[i] * v for i, v in woken if scs[i].asil_req == a and (dem[i] > 0 or not types)]
                            if terms and (types or part_cap < float('inf')):
                                model.addConstr(hosted[a] + gp.quicksum(terms) <= cap_expr(j, a, res) * der(res, H_j),
                                                name=f"cold_{res}_capacity_{locations[c].id}_{locations[j].id}_asil{a}")
                                cstats['c1h_cold_capacity'] += 1
                        if site_cap is not None:
                            model.addConstr(gp.quicksum(hosted.values()) + gp.quicksum(dem[i] * v for i, v in woken)
                                            <= float(site_cap) * (H_j if res == 'cpu' else 1.0) * y[j],
                                            name=f"cold_site_{res}_cap_{locations[c].id}_{locations[j].id}")
                            cstats['c1h_cold_capacity'] += 1

        def _slot_classes(here, h, capd, H_j):
            types_h = capd.get('types') or []
            if not types_h:
                return {}
            small = min(float(ty['tops']) for ty in types_h) * H_j
            classes = {}
            for i in here:
                d = float(scs[i].tops_req) * get_sw_multiplier(scs[i], 'tops')
                if d > small + 1e-9:
                    classes.setdefault(round(d, 6), []).append(i)
            return classes

        def tops_of(j, h, capd):
            if capd.get('types'):
                return gp.quicksum(ty['tops'] * net['hw_t'][j, h, t] for t, ty in enumerate(capd['types']) if (j, h, t) in net['hw_t'])
            return float(capd.get('tops', 0) or 0) * hw_use[j, h]

        hw_cap_all = (partitions or {}).get('hw_capacity', {}) or {}
        if act and hw_cap_all:
            for h, capd in hw_cap_all.items():
                tops_unit = float(capd.get('tops', 0) or 0)
                if tops_unit <= 0:
                    continue
                for j in range(n_locs):
                    if (j, h) not in hw_use:
                        continue
                    here = [i for i in feasible_scs_per_loc[j] if h in (scs[i].hw_required or []) and float(getattr(scs[i], 'tops_req', 0) or 0) > 0]
                    if not here:
                        continue
                    H_j = site_derating(locations[j], unc)
                    base = gp.quicksum(x[i, j] * scs[i].tops_req * get_sw_multiplier(scs[i], 'tops') for i in here) + gp.quicksum(
                        net['hr'][i, j] * scs[i].tops_req * get_sw_multiplier(scs[i], 'tops') for i in here if (i, j) in net['hr'])
                    for c in range(n_locs):
                        woken = [(i, act[i, c, j]) for i in here if (i, c, j) in act]
                        if woken:
                            model.addConstr(base + gp.quicksum(scs[i].tops_req * get_sw_multiplier(scs[i], 'tops') * v for i, v in woken)
                                            <= H_j * tops_of(j, h, capd), name=f"cold_tops_{locations[c].id}_{locations[j].id}_{h}")
                            cstats['c1h_cold_capacity'] += 1
                            woken_of = dict(woken)
                            for d_cls, members in _slot_classes(here, h, capd, H_j).items():
                                if not any(i in woken_of for i in members):
                                    continue
                                slots = gp.quicksum(int((float(ty['tops']) * H_j) // d_cls) * net['hw_t'][j, h, t]
                                                    for t, ty in enumerate(capd['types']) if (j, h, t) in net['hw_t'] and (float(ty['tops']) * H_j) // d_cls >= 1)
                                model.addConstr(gp.quicksum(x[i, j] for i in members) + gp.quicksum(net['hr'][i, j] for i in members if (i, j) in net['hr'])
                                                + gp.quicksum(woken_of[i] for i in members if i in woken_of) <= slots,
                                                name=f"cold_tops_slots_{locations[c].id}_{locations[j].id}_{h}_{int(d_cls)}")
                                cstats['c1h_cold_capacity'] += 1

        for i in range(n_sc):
            for h in (scs[i].hw_required or []):
                for j in feasible_locs_per_sc[i]:
                    model.addConstr(
                        x_expr(i, j) <= hw_use[j, h],
                        name=f"hw_required_{scs[i].id}_{h}_{locations[j].id}"
                    )
                    cstats['c3_hw_required'] += 1

        s_lookup = build_sensor_lookup(sensors)
        a_lookup = build_actuator_lookup(actuators)
        for j in range(n_locs):
            for i_name in all_interfaces:
                comm_port_demand = gp.LinExpr()
                for (j1, j2, iface), comm_var in comm.items():
                    if (j1 == j) and (iface == i_name):
                        comm_port_demand.add(comm_var, 1)
                    elif (j2 == j) and (iface == i_name):
                        comm_port_demand.add(comm_var, 1)

                if i_name in bus['types']:
                    model.addConstr(
                        if_use[j, i_name] >= gp.quicksum(bus['unit'][b, j] for b in bus['inst'] if b[0] == i_name) + comm_port_demand,
                        name=f"bus_ports_{locations[j].id}_{i_name}"
                    )

                else:
                    sc_port_demand = gp.LinExpr()

                    for i in feasible_scs_per_loc[j]:
                        sc = scs[i]
                        if i_name == 'ETH' and (attach_s is not None or attach_a is not None):
                            num_sensors_if = 0
                            num_actuators_if = 0
                        else:
                            num_sensors_if = sum(
                                1
                                for s_id in (sc.sensors or [])
                                if s_lookup.get(s_id)
                                and s_lookup.get(s_id).interface == i_name
                            )
                            num_actuators_if = sum(
                                1
                                for a_id in (sc.actuators or [])
                                if a_lookup.get(a_id)
                                and a_lookup.get(a_id).interface == i_name
                            )

                        if num_sensors_if + num_actuators_if > 0:
                            sc_port_demand.add(x_expr(i, j), num_sensors_if + num_actuators_if)
                            if (i, j) in net['r']:
                                sc_port_demand.add(net['r'][i, j], num_sensors_if + num_actuators_if)

                    if i_name == 'ETH':
                        for (si, jj), av in (attach_s or {}).items():
                            if jj == j:
                                sc_port_demand.add(av, 1.0)
                        for (ai, jj), av in (attach_a or {}).items():
                            if jj == j:
                                sc_port_demand.add(av, 1.0)

                    if i_name == 'ETH' and 'SWITCH' in all_hw:
                        SWITCH_ETH_PORTS = 16
                        model.addConstr(
                            if_use[j, i_name] + SWITCH_ETH_PORTS * hw_use[j, 'SWITCH'] >= sc_port_demand + comm_port_demand,
                            name=f"port_count_{locations[j].id}_{i_name}_switch_covers"
                        )
                    else:
                        model.addConstr(
                            if_use[j, i_name] >= sc_port_demand + comm_port_demand,
                            name=f"port_count_{locations[j].id}_{i_name}"
                        )

                cstats['c4_if_required'] += 1

        if unc.get('redundancy'):
            sc_id_to_idx = {sc.id: i for i, sc in enumerate(scs)}
            for i, sc in enumerate(scs):
                if hasattr(sc, 'redundant_with') and sc.redundant_with:
                    partner_id = sc.redundant_with
                    if partner_id in sc_id_to_idx:
                        partner_idx = sc_id_to_idx[partner_id]
                        if i < partner_idx:
                            common_js = set(feasible_locs_per_sc[i]).intersection(set(feasible_locs_per_sc[partner_idx]))
                            for j in common_js:
                                model.addConstr(
                                    x_expr(i, j)
                                    + x_expr(partner_idx, j)
                                    <= 1,
                                    name=f"redundancy_{sc.id}_{partner_id}_{locations[j].id}"
                                )
                                cstats['c5_redundancy'] += 1

        hw_cap6 = (partitions or {}).get('hw_capacity', {}) or {}
        for h, capd in hw_cap6.items():
            tops_unit = float(capd.get('tops', 0) or 0)
            users = {i for i in range(n_sc) if h in (scs[i].hw_required or []) and float(getattr(scs[i], 'tops_req', 0) or 0) > 0}
            if tops_unit <= 0 or not users:
                continue
            for j in range(n_locs):
                here = [i for i in feasible_scs_per_loc[j] if i in users]
                if not here or (j, h) not in hw_use:
                    continue
                H_j = site_derating(locations[j], unc)
                dem = gp.quicksum(x[i, j] * scs[i].tops_req * get_sw_multiplier(scs[i], 'tops') for i in here) + gp.quicksum(
                    net['hr'][i, j] * scs[i].tops_req * get_sw_multiplier(scs[i], 'tops') for i in here if (i, j) in net['hr'])
                model.addConstr(dem <= H_j * tops_of(j, h, capd), name=f"tops_capacity_{locations[j].id}_{h}")
                cstats['c6_ai_contention'] += 1
                for d_cls, members in _slot_classes(here, h, capd, H_j).items():
                    slots = gp.quicksum(int((float(ty['tops']) * H_j) // d_cls) * net['hw_t'][j, h, t]
                                        for t, ty in enumerate(capd['types']) if (j, h, t) in net['hw_t'] and (float(ty['tops']) * H_j) // d_cls >= 1)
                    model.addConstr(gp.quicksum(x[i, j] for i in members) + gp.quicksum(net['hr'][i, j] for i in members if (i, j) in net['hr']) <= slots,
                                    name=f"tops_slots_{locations[j].id}_{h}_{int(d_cls)}")
                    cstats['c6_ai_contention'] += 1

        all_interfaces = list(cable_types.keys())

        if 'SWITCH' in all_hw:
            loc_active = {j: y[j] + hw_use[j, 'SWITCH'] for j in range(n_locs)}
        else:
            loc_active = {j: 1.0 * y[j] for j in range(n_locs)}

        eth_cap_obj = cable_types.get('ETH') if cable_types else None
        eth_cap = getattr(eth_cap_obj, 'capacity', None) if eth_cap_obj is not None else None
        try:
            eth_cap = float(eth_cap) if eth_cap is not None else None
        except Exception:
            eth_cap = None
        if attach_s:
            for (si, j), av in attach_s.items():
                model.addConstr(av <= loc_active[j], name=f"attachS_active_{sensors[si].id}_{locations[j].id}")
                cstats['c9_attach'] += 1

                vol = float(getattr(sensors[si], 'volume', 0.0) or 0.0)
                if eth_cap is not None and eth_cap > 0 and eth_cap != float('inf') and vol > 0:
                    model.addConstr(
                        vol * av <= eth_cap,
                        name=f"attachS_bw_{sensors[si].id}_{locations[j].id}"
                    )
                    cstats['c9_attach'] += 1


            for si in {k[0] for k in attach_s.keys()}:
                model.addConstr(
                    gp.quicksum(attach_s[si, j] for j in range(n_locs) if (si, j) in attach_s) == 1,
                    name=f"attachS_one_{sensors[si].id}"
                )
                cstats['c9_attach'] += 1

        if attach_a:
            for (ai, j), av in attach_a.items():
                model.addConstr(av <= loc_active[j], name=f"attachA_active_{actuators[ai].id}_{locations[j].id}")
                cstats['c9_attach'] += 1

                vol = float(getattr(actuators[ai], 'volume', 0.0) or 0.0)
                if eth_cap is not None and eth_cap > 0 and eth_cap != float('inf') and vol > 0:
                    model.addConstr(
                        vol * av <= eth_cap,
                        name=f"attachA_bw_{actuators[ai].id}_{locations[j].id}"
                    )
                    cstats['c9_attach'] += 1


            for ai in {k[0] for k in attach_a.keys()}:
                model.addConstr(
                    gp.quicksum(attach_a[ai, j] for j in range(n_locs) if (ai, j) in attach_a) == 1,
                    name=f"attachA_one_{actuators[ai].id}"
                )
                cstats['c9_attach'] += 1

        crit_set = set(net['critical'])
        for d, info in bus['devices'].items():
            lines = [b for b in bus['inst'] if b[0] == info['iface']]
            info['lanes'] = 2 if any(i in crit_set for i in info['users']) else 1
            total = gp.quicksum(bus['dev'][b, d] for b in lines)
            model.addConstr(total == info['lanes'] if info['users'] else total <= 1, name=f"bus_device_{d}")
            cstats['c11_bus'] += 1
            for b in lines:
                dv = bus['dev'][b, d]
                model.addConstr(dv <= bus['act'][b], name=f"bus_dev_act_{d}_{b[0]}{b[1] + 1}")
                if info['group'] == 'HIGH':
                    model.addConstr(dv <= bus['high'][b], name=f"bus_dev_high_{d}_{b[0]}{b[1] + 1}")
                else:
                    model.addConstr(dv <= 1 - bus['high'][b], name=f"bus_dev_low_{d}_{b[0]}{b[1] + 1}")
                cstats['c11_bus'] += 2
                for i in ([] if bus['gateway'] else info['users']):
                    for j in feasible_locs_per_sc[i]:
                        model.addConstr(bus['unit'][b, j] >= dv + x[i, j] - 1, name=f"bus_driver_{d}_{scs[i].id}_{b[0]}{b[1] + 1}_{locations[j].id}")
                        cstats['c11_bus'] += 1
                        if (i, j) in net['r']:
                            model.addConstr(bus['unit'][b, j] >= dv + net['r'][i, j] - 1, name=f"bus_standby_{d}_{scs[i].id}_{b[0]}{b[1] + 1}_{locations[j].id}")
                            cstats['c11_bus'] += 1
        for (d, b, j), gv in bus['gw'].items():
            model.addConstr(gv <= bus['unit'][b, j], name=f"bus_gw_member_{d}_{b[0]}{b[1] + 1}_{locations[j].id}")
            model.addConstr(gv <= bus['dev'][b, d], name=f"bus_gw_lane_{d}_{b[0]}{b[1] + 1}_{locations[j].id}")
            cstats['c11_bus'] += 2
        for d in bus['devices']:
            for b in bus['inst']:
                gvs = [bus['gw'][d, b, j] for j in range(n_locs) if (d, b, j) in bus['gw']]
                if gvs:
                    model.addConstr(gp.quicksum(gvs) == bus['dev'][b, d], name=f"bus_gw_one_{d}_{b[0]}{b[1] + 1}")
                    cstats['c11_bus'] += 1
        for d, info in bus['devices'].items():
            if info.get('lanes', 1) < 2:
                continue
            for j in range(n_locs):
                gvs = [bus['gw'][d, b, j] for b in bus['inst'] if (d, b, j) in bus['gw']]
                if len(gvs) > 1:
                    model.addConstr(gp.quicksum(gvs) <= 1, name=f"bus_gw_apart_{d}_{locations[j].id}")
                    cstats['c11_bus'] += 1
        for b in bus['inst']:
            name = f"{b[0]}{b[1] + 1}"
            devs = [d for d in bus['devices'] if (b, d) in bus['dev']]
            for j in range(n_locs):
                model.addConstr(bus['unit'][b, j] <= y[j], name=f"bus_unit_exists_{name}_{locations[j].id}")
                model.addConstr(bus['unit'][b, j] <= bus['act'][b], name=f"bus_unit_act_{name}_{locations[j].id}")
                cstats['c11_bus'] += 2
            line_sel = [sv for (u, v, w_if, p), sv in net['sel'].items() if w_if == b[0] and p == b[1]]
            model.addConstr(bus['act'][b] <= gp.quicksum(bus['dev'][b, d] for d in devs) + gp.quicksum(line_sel), name=f"bus_has_use_{name}")
            cap = float(getattr(cable_types.get(b[0]), 'capacity', float('inf'))) * net['bus_umax']
            if cap < float('inf'):
                model.addConstr(gp.quicksum(bus['devices'][d]['volume'] * bus['dev'][b, d] for d in devs) <= cap, name=f"bus_capacity_{name}")
                cstats['c11_bus'] += 1
            members = [(locations[j].id, bus['unit'][b, j]) for j in range(n_locs)] + [(d, bus['dev'][b, d]) for d in devs]
            nn = {e: min((steiner_tree([e, f], bus['points'])[0] for f, _v in members if f != e), default=0.0) for e, _v in members}
            model.addConstr(bus['len'][b] >= 0.5 * gp.quicksum(nn[e] * v for e, v in members), name=f"bus_len_tour_{name}")
            cstats['c11_bus'] += 1
            for k1 in range(len(members)):
                for k2 in range(k1 + 1, len(members)):
                    (ida, va), (idb, vb) = members[k1], members[k2]
                    dist = steiner_tree([ida, idb], bus['points'])[0]
                    if dist > 0:
                        model.addConstr(bus['len'][b] >= dist * (va + vb - 1), name=f"bus_len_pair_{name}_{k1}_{k2}")
                        cstats['c11_bus'] += 1
            nxt = (b[0], b[1] + 1)
            if nxt in bus['act']:
                model.addConstr(bus['act'][b] >= bus['act'][nxt], name=f"bus_order_{name}")
                cstats['c11_bus'] += 1
            allowed, opened = [], 0
            for d in sorted(devs):
                opened += bus['devices'][d].get('lanes', 1)
                if b[1] >= opened:
                    model.addConstr(bus['dev'][b, d] == 0, name=f"bus_symmetry_{name}_{d}")
                    cstats['c11_bus'] += 1
                else:
                    allowed.append(d)
            def tree_bound(members, label):
                st_len = steiner_tree([e for e, _v in members], bus['points'])[0]
                if st_len > 0:
                    model.addConstr(bus['len'][b] >= st_len * (gp.quicksum(v for _e, v in members) - (len(members) - 1)), name=f"bus_tree_{name}_{label}")
                    cstats['c11_bus'] += 1
            units = [(locations[j].id, bus['unit'][b, j]) for j in range(n_locs)]
            tb_mode = str(net.get('bus_tree_bounds', 'full'))
            if tb_mode != 'pairs':
                for size in range(1, min(4, len(allowed)) + 1):
                    for group in itertools.combinations(allowed, size):
                        gm = [(d, bus['dev'][b, d]) for d in group]
                        if size >= 2:
                            tree_bound(gm, '_'.join(group))
                        for u1 in range(n_locs):
                            tree_bound(gm + [units[u1]], f"{'_'.join(group)}_{units[u1][0]}")
                            if tb_mode == 'full':
                                for u2 in range(u1 + 1, n_locs):
                                    tree_bound(gm + [units[u1], units[u2]], f"{'_'.join(group)}_{units[u1][0]}_{units[u2][0]}")
            if n_locs <= 12 and tb_mode == 'full':
                for trio in itertools.combinations(range(n_locs), 3):
                    tree_bound([units[k] for k in trio], '_'.join(units[k][0] for k in trio))

        for (j1, j2, iface), comm_var in comm.items():
            M = comm_ub[iface]
            model.addConstr(
                comm_var <= M * loc_active[j1],
                name=f"comm_active_j1_{locations[j1].id}_{locations[j2].id}_{iface}"
            )
            cstats['c9_if_activation'] += 1
            model.addConstr(
                comm_var <= M * loc_active[j2],
                name=f"comm_active_j2_{locations[j1].id}_{locations[j2].id}_{iface}"
            )
            cstats['c9_if_activation'] += 1

            if iface == 'ETH' and 'SWITCH' in all_hw:
                model.addConstr(
                    comm_var <= M * (if_use[j1, iface] + hw_use[j1, 'SWITCH']),
                    name=f"if_act_j1_{locations[j1].id}_{locations[j2].id}_{iface}_or_switch"
                )
                cstats['c9_if_activation'] += 1
                model.addConstr(
                    comm_var <= M * (if_use[j2, iface] + hw_use[j2, 'SWITCH']),
                    name=f"if_act_j2_{locations[j1].id}_{locations[j2].id}_{iface}_or_switch"
                )
                cstats['c9_if_activation'] += 1
            else:
                model.addConstr(
                    comm_var <= M * if_use[j1, iface],
                    name=f"if_act_j1_{locations[j1].id}_{locations[j2].id}_{iface}"
                )
                cstats['c9_if_activation'] += 1
                model.addConstr(
                    comm_var <= M * if_use[j2, iface],
                    name=f"if_act_j2_{locations[j1].id}_{locations[j2].id}_{iface}"
                )
                cstats['c9_if_activation'] += 1

        if 'SWITCH' in all_hw:
            for j in range(n_locs):
                part_sum_j = y[j]
                incident_comm_j = gp.LinExpr()
                for (j1, j2, _iface), comm_var in comm.items():
                    if j1 == j or j2 == j:
                        incident_comm_j.add(comm_var, 1.0)

                model.addConstr(
                    incident_comm_j >= 2 * hw_use[j, 'SWITCH'] - 2 * part_sum_j,
                    name=f"switch_only_degree_ge2_{locations[j].id}"
                )


        paths, sel = net['paths'], net['sel']
        has_switch = 'SWITCH' in all_hw

        for t, elig in net['flow_elig'].items():
            if len(elig) > 1:
                model.addConstr(gp.quicksum(net['w'][t, n] for n in elig) == 1, name=f"flow_iface_{t}")
                cstats['c10_pair_traffic'] += 1

        for (u, v, w_if), plist in paths.items():
            model.addConstr(gp.quicksum(sel[u, v, w_if, p] for p in range(len(plist))) <= 1, name=f"one_path_{locations[u].id}_{locations[v].id}_{w_if}")
            cstats['c10_path_choice'] += 1
            if w_if in bus['types']:
                d_uv = get_distance(locations[u].location, locations[v].location)
                for p in range(len(plist)):
                    for j in (u, v):
                        model.addConstr(sel[u, v, w_if, p] <= bus['unit'][(w_if, p), j], name=f"line_member_{locations[u].id}_{locations[v].id}_{w_if}{p + 1}_{locations[j].id}")
                        cstats['c10_path_choice'] += 1
                    model.addConstr(bus['len'][(w_if, p)] >= d_uv * sel[u, v, w_if, p], name=f"line_length_{locations[u].id}_{locations[v].id}_{w_if}{p + 1}")
                    cstats['c10_path_choice'] += 1
            elif net.get('path_link_cut'):
                for p, path in enumerate(plist):
                    for a, b in zip(path, path[1:]):
                        key = (a, b, w_if) if a < b else (b, a, w_if)
                        model.addConstr(comm[key] >= sel[u, v, w_if, p], name=f"path_cable_{locations[u].id}_{locations[v].id}_{w_if}_p{p}_{locations[a].id}_{locations[b].id}")
                        cstats['c10_path_choice'] += 1
            for p, path in enumerate(plist):
                for r in path[1:-1]:
                    if has_switch:
                        model.addConstr(sel[u, v, w_if, p] <= hw_use[r, 'SWITCH'], name=f"relay_switch_{locations[u].id}_{locations[v].id}_p{p}_{locations[r].id}")
                    else:
                        model.addConstr(sel[u, v, w_if, p] == 0, name=f"relay_impossible_{locations[u].id}_{locations[v].id}_p{p}")
                    cstats['c10_relay_switch'] += 1

        if net.get('upfront_paths'):
            cstats.setdefault('c10_upfront_paths', 0)
            for t, (src, dst) in net['flow_ends'].items():
                for u, su in src.items():
                    for v, dv in dst.items():
                        if u == v:
                            continue
                        pair = (min(u, v), max(u, v))
                        terms = [sel[pair + (w_if, q)] for w_if in net['flow_elig'][t] if pair + (w_if,) in paths
                                 for q in range(len(paths[pair + (w_if,)]))]
                        if terms:
                            model.addConstr(gp.quicksum(terms) >= su + dv - 1, name=f"upp_{cstats['c10_upfront_paths']}")
                            cstats['c10_upfront_paths'] += 1
        if net.get('cell_cut_capacity'):
            cstats.setdefault('c10_cell_cut', 0)
            for u in range(n_locs):
                cap = [net['cap_of'][w_if] * var for (a, b2, w_if), var in comm.items() if u in (a, b2) and net['cap_of'][w_if] < float('inf')]
                cap += [net['cap_of'][b[0]] * net['bus_umax'] * bus['unit'][b, u] for b in bus['inst'] if (b, u) in bus['unit']
                        and net['cap_of'].get(b[0], float('inf')) < float('inf')]
                out_t, in_t = [], []
                for t, (src, dst) in net['flow_ends'].items():
                    tr = net['flow_meta'][t]
                    if tr.get('src_inst', 'x') == 'r' or tr.get('dst_inst', 'x') == 'r':
                        continue
                    vol = net['volumes'][t]
                    if vol <= 0:
                        continue
                    if u in src:
                        o = model.addVar(lb=0.0, ub=1.0, name=f"cut_out_{u}_{len(out_t)}")
                        model.addConstr(o >= src[u] - dst.get(u, 0), name=f"cut_out_def_{u}_{len(out_t)}")
                        out_t.append(vol * o)
                    if u in dst:
                        i_ = model.addVar(lb=0.0, ub=1.0, name=f"cut_in_{u}_{len(in_t)}")
                        model.addConstr(i_ >= dst[u] - src.get(u, 0), name=f"cut_in_def_{u}_{len(in_t)}")
                        in_t.append(vol * i_)
                if out_t:
                    model.addConstr(gp.quicksum(out_t) <= gp.quicksum(cap), name=f"cell_cut_out_{locations[u].id}")
                    cstats['c10_cell_cut'] += 1
                if in_t:
                    model.addConstr(gp.quicksum(in_t) <= gp.quicksum(cap), name=f"cell_cut_in_{locations[u].id}")
                    cstats['c10_cell_cut'] += 1

        cstats['total'] = sum(v for k, v in cstats.items() if k != 'total')
        return model, cstats

    def _build_objective_function(self, model, x, P, y, hw_use, if_use, comm, scs, locations, sensors, actuators,
                                  cable_types, comm_matrix, partitions, hardwares, interfaces,
                                  n_locs, unique_asils, all_hw, all_interfaces,
                                  feasible_locs_per_sc, attach_s=None, attach_a=None, bus=None,
                                  minimize_cable_length=False, cable_length_limit=None, cost_limit=None, standby=None, part_types=None,
                                  hw_types=None):
        print(f"Building objective function...")

        Q, types = part_types or ({}, None)
        if types:
            partition_cost_expr = gp.quicksum(types[t]['cost'] * v for (_j, _a, t), v in Q.items())
        else:
            partition_cost_expr = gp.quicksum(partitions.get('cost', 0) * P[j, a] for j in range(n_locs) for a in unique_asils)

        site_fixed_cost = float(partitions.get('site_fixed_cost', 0) or 0)
        site_cost_expr = gp.quicksum(site_fixed_cost * y[j] for j in range(n_locs))
        switch_node_cost = float(partitions.get('switch_node_cost', 0) or 0)
        if switch_node_cost > 0 and 'SWITCH' in all_hw:
            for j in range(n_locs):
                node = model.addVar(lb=0.0, ub=1.0, name=f"switch_node_{locations[j].id}")
                model.addConstr(node >= hw_use[j, 'SWITCH'] - y[j], name=f"switch_node_def_{locations[j].id}")
                model.addConstr(node <= hw_use[j, 'SWITCH'], name=f"switch_node_hw_{locations[j].id}")
                model.addConstr(node <= 1 - y[j], name=f"switch_node_unit_{locations[j].id}")
                site_cost_expr += switch_node_cost * node

        hw_cost_expr = gp.quicksum(
            hardwares.get(h, 0) * hw_use[j, h]
            for j in range(n_locs)
            for h in all_hw
        )
        hw_t, hw_capd = hw_types or ({}, {})
        if hw_t:
            hw_cost_expr += gp.quicksum(hw_capd[h]['types'][t]['cost'] * v for (j, h, t), v in hw_t.items())

        if_cost_expr = gp.quicksum(
            interfaces[i_name].port_cost * if_use[j, i_name]
            for j in range(n_locs)
            for i_name in all_interfaces
        )

        cable_cost_expr, cable_distance_expr, _ = self.calculate_cable_expressions(
            x, scs, locations, sensors, actuators, cable_types, comm_matrix,
            feasible_locs_per_sc, comm=comm, attach_s=attach_s, attach_a=attach_a, bus=bus, standby=standby
        )

        total_cost = partition_cost_expr + site_cost_expr + hw_cost_expr + if_cost_expr + cable_cost_expr

        if cable_length_limit is not None:
            model.addConstr(
                cable_distance_expr <= float(cable_length_limit),
                name="eps_cable_length_limit"
            )

        if cost_limit is not None:
            model.addConstr(
                total_cost <= float(cost_limit),
                name="eps_cost_limit"
            )

        cost_obj = ('total_cost', gp.LinExpr() + total_cost)
        cable_obj = ('cable_length', gp.LinExpr() + cable_distance_expr)
        objectives = [cable_obj, cost_obj] if minimize_cable_length else [cost_obj, cable_obj]
        print(f"Objective: {objectives[0][0]}, then {objectives[1][0]} (lexicographic)")

        return model, objectives

    def optimize(self, scs, locations, sensors, actuators, cable_types, comm_matrix, partitions=None, hardwares=None, interfaces=None, enable_comm_bw_constraints=True, comm_bw_big_m=10000, minimize_cable_length=False, cable_length_limit=None, cost_limit=None, time_limit=None, mip_gap=None, lp_path=None, uncertainty=None, backbone=None, phase2_time=600, gurobi_params=None, chains=None, start=None, lns=None, fixed=None, phase2_cap=None, reuse=False, resume=False, relax_network=False):
        print("=" * 80)
        print("Gurobi OPTIMIZATION")
        print("=" * 80)

        n_sc = len(scs)
        n_locs = len(locations)

        unique_asils = sorted(list(set(sc.asil_req for sc in scs)))
        all_hw = list(hardwares.keys()) if hardwares else []
        all_interfaces = list(interfaces.keys()) if interfaces else list(cable_types.keys())
        tops_feature = partitions.get('tops_feature') if partitions.get('types') else None
        if tops_feature:
            n_tf = 0
            for sc in scs:
                if tops_feature in (sc.hw_required or []):
                    sc.hw_required = [h for h in sc.hw_required if h != tops_feature]
                    n_tf += 1
            if n_tf:
                print(f"  Partition types: the {tops_feature} requirement of {n_tf} SCs is served by their block's TOPS")
            types_ = partitions['types']
            print("  Partition types: " + "; ".join(f"{t['id']} cpu {t['cpu_cap']:g} ram {t['ram_cap']:g} rom {t['rom_cap']:g} "
                                                     f"tops {t['tops_cap']:g} cost {t['cost']:g}" for t in types_))

        print(f"\nProblem size:")
        print(f"  SCs: {n_sc}, Locations: {n_locs}")
        print(f"  Unique ASIL levels: {unique_asils}")
        print(f"  HW features available: {all_hw}")
        print(f"  Interfaces available: {all_interfaces}")
        print(f"  Partition config: cost={partitions.get('cost', 0)}, cpu_cap={partitions.get('cpu_cap', 0)}, ram_cap={partitions.get('ram_cap', 0)}, rom_cap={partitions.get('rom_cap', 0)}")
        print(f"  Site config: max partitions per ASIL={partitions.get('max_per_asil_per_site', 3)}, site max partitions={partitions.get('site_max_partitions')}, "
              f"site caps cpu/ram/rom={partitions.get('site_cpu_cap')}/{partitions.get('site_ram_cap')}/{partitions.get('site_rom_cap')}, fixed cost={partitions.get('site_fixed_cost', 0)}")

        infeasible_ij = set()

        key = (tuple(sc.id for sc in scs), tuple(loc.id for loc in locations), bool((backbone or {}).get('bus_gateway', False)))
        session = self._session if (reuse and self._session is not None and self._session['key'] == key) else None
        resuming = bool(resume) and session is not None and bool(self._resumable)
        self._resumable = False
        if resume and not resuming:
            print("  [resume] no resumable solve of this instance: solved afresh")
        if session is None:
            if self._session is not None:
                old = self._session['parts'][0]
                self._session = None
                old.dispose()
                gc.collect()
                print("  [memory] the model of the previous instance was freed before the translation of this one", flush=True)
            model, x, P, y, hw_use, if_use, attach_s, attach_a, bus, comm, traffic_flows, net, feasible_locs_per_sc, feasible_scs_per_loc, var_stats = self._create_model_and_variables(
                n_sc, n_locs, scs, locations, unique_asils, all_hw, sensors, actuators, all_interfaces, infeasible_ij=infeasible_ij, comm_matrix=comm_matrix,
                partitions=partitions, backbone=backbone, cable_types=cable_types, uncertainty=uncertainty, chains=chains
            )

            model, cstats = self._add_constraints(
                model, x, P, y, hw_use, if_use, attach_s, attach_a, bus, comm, traffic_flows, net, scs, locations, sensors, actuators,
                cable_types, comm_matrix, partitions, unique_asils,
                feasible_locs_per_sc, feasible_scs_per_loc,
                enable_comm_bw_constraints=enable_comm_bw_constraints,
                comm_bw_big_m=comm_bw_big_m,
                uncertainty=uncertainty
            )

            print("\n[DEBUG] Variable breakdown:")
            print(f"  total vars: {var_stats['total']}")
            print(f"  x: {var_stats['x']}")
            print(f"  P: {var_stats['P']}")
            print(f"  y: {var_stats['y']}")
            print(f"  hw_use: {var_stats['hw_use']}")
            print(f"  if_use: {var_stats['if_use']}")
            print(f"  attach_s: {var_stats.get('attach_s', 0)}")
            print(f"  attach_a: {var_stats.get('attach_a', 0)}")
            print(f"  bus lines: {var_stats.get('bus', 0)}")
            print(f"  comm: {var_stats['comm']} ({(100.0 * var_stats['comm'] / max(1, var_stats['total'])):.1f}%)")
            print(f"  traffic_flows: {var_stats['traffic_flows']}")
            print(f"  sel: {var_stats['sel']}  (site pairs with candidate paths: {var_stats['pairs_with_paths']})")

            print("\n[DEBUG] Constraint breakdown:")
            for k, v in cstats.items():
                print(f"  {k}: {v}")

            model, objectives = self._build_objective_function(
                model, x, P, y, hw_use, if_use, comm, scs, locations, sensors, actuators,
                cable_types, comm_matrix, partitions, hardwares, interfaces,
                n_locs, unique_asils, all_hw, all_interfaces,
                feasible_locs_per_sc, attach_s=attach_s, attach_a=attach_a, bus=bus,
                minimize_cable_length=minimize_cable_length, cable_length_limit=cable_length_limit,
                cost_limit=cost_limit, standby=net['r'], part_types=(net['Q'], net['types']),
                hw_types=(net['hw_t'], (partitions or {}).get('hw_capacity', {}) or {})
            )

            unc_lat = (uncertainty or {}).get('params', {}).get('robust_latency', {})
            net['lat'] = {'robust': bool((uncertainty or {}).get('robust_latency')),
                          'factor': 1.0,
                          'sync': float(unc_lat.get('sync_ms', 0.0)),
                          'platform': float(unc_lat.get('platform_mean_ms', unc_lat.get('platform_ms', 0.0)))
                          + float((uncertainty or {}).get('rho', 1.0)) * (float(unc_lat.get('platform_p99_ms', unc_lat.get('platform_mean_ms', unc_lat.get('platform_ms', 0.0))))
                                                              - float(unc_lat.get('platform_mean_ms', unc_lat.get('platform_ms', 0.0))))}
            sc_index = {sc.id: i for i, sc in enumerate(scs)}
            base_flows = [tr for tr in traffic_flows if 'base' not in tr]
            flow_of_pair = {(tr['src_idx'], tr['dst_idx']): tr['id'] for tr in base_flows if tr['type'] == 'SC_SC'}
            flow_into = {tr['dst_idx']: tr['id'] for tr in base_flows if tr['type'] in ('SENS_SC', 'BUS_SC')}
            flow_from = {tr['src_idx']: tr['id'] for tr in base_flows if tr['type'] in ('SC_ACT', 'SC_BUS')}
            dev_end = {}
            for d, info in net['bus']['devices'].items():
                t_if = cable_types[info['iface']]
                ms = robust_ms(net, info['kbit'] / float(t_if.capacity) + float(getattr(t_if, 'hop_latency_ms', 0.0)))
                for i in info['users']:
                    if ms > dev_end.get((i, info['kind']), (None, -1.0))[1]:
                        dev_end[i, info['kind']] = (d, ms)
            net['chains'] = []
            for ch in (chains or []):
                idx = [sc_index[s] for s in ch['swcs'] if s in sc_index]
                if not idx:
                    continue
                flows = [flow_into[idx[0]]] if idx and idx[0] in flow_into else []
                flows += [flow_of_pair[a, b] for a, b in zip(idx, idx[1:]) if (a, b) in flow_of_pair]
                flows += [flow_from[idx[-1]]] if idx and idx[-1] in flow_from else []
                ends = [dev_end[k] for k in ((idx[0], 'sensor'), (idx[-1], 'actuator')) if idx and k in dev_end]
                const = sum(ms for _d, ms in ends)
                net['chains'].append({'id': ch['id'], 'budget': float(ch['net_budget_ms']), 'const': const, 'dev_ends': [d for d, _ms in ends],
                                      'degraded': float(ch.get('degraded_bound_ms', float(ch['net_budget_ms']))), 'flows': flows})
                if const > float(ch['net_budget_ms']):
                    print(f"  WARNING: chain {ch['id']} cannot meet its budget: its bus crossings alone take {const:.2f} ms")
            if net['chains']:
                print(f"  Chains: {len(net['chains'])} (network budgets {min(c['budget'] for c in net['chains']):.1f} - "
                      f"{max(c['budget'] for c in net['chains']):.1f} ms, robust latency {net['lat']['robust']})")

            net['frame_kbit'] = {n: float(getattr((cable_types or {}).get(n), 'frame_kbit', float('inf'))) for n in net['bb_ifaces']}
            if (backbone or {}).get('priorities'):
                pf, pd = {}, {}
                for ch in net['chains']:
                    for t in ch['flows']:
                        pf[t] = min(pf.get(t, float('inf')), ch['budget'])
                    for d in ch['dev_ends']:
                        pd[d] = min(pd.get(d, float('inf')), ch['budget'])
                for t, tr in net['flow_meta'].items():
                    if tr.get('base') in pf:
                        pf[t] = pf[tr['base']]
                net['prio'] = {'f': pf, 'd': pd}
                print(f"  Priorities: {len(set(pf.values()) | set(pd.values()))} chain classes for {len(pf)} flows and "
                      f"{len(pd)} devices, all other traffic last")

            def min_crossing(t):
                return min(robust_ms(net, net['kbit'][t] / net['cap_of'][n] + net['hop_ms'][n]) for n in net['flow_elig'][t])

            cross, cross_c, n_nom, n_fail = {}, {}, 0, 0
            for ch in net['chains']:
                terms = []
                for t in ch['flows']:
                    if t not in cross:
                        src, dst = net['flow_ends'][t]
                        cross[t] = model.addVar(lb=0.0, ub=1.0, name=f"cross_{net['flow_meta'][t]['key']}")
                        tr_c = net['flow_meta'][t]
                        if tr_c['type'] in ('BUS_SC', 'SC_BUS'):
                            user = dst if tr_c['type'] == 'BUS_SC' else src
                            lines_c = [b for b in bus['inst'] if b[0] == bus['devices'][tr_c['dev']]['iface']]
                            for j, var in user.items():
                                gws = [bus['gw'][tr_c['dev'], b, j] for b in lines_c if (tr_c['dev'], b, j) in bus['gw']]
                                model.addConstr(cross[t] >= var - gp.quicksum(gws), name=f"cross_{t}_{j}")
                        else:
                            for j, var in src.items():
                                model.addConstr(cross[t] >= var - (dst[j] if j in dst else 0), name=f"cross_{t}_{j}")
                    terms.append(min_crossing(t) * cross[t])
                model.addConstr(gp.quicksum(terms) <= ch['budget'] - ch['const'], name=f"chain_lb_{ch['id']}")
                n_nom += 1
            crit = set(net['critical'])
            for ch in net['chains']:
                meta = [net['flow_meta'][t] for t in ch['flows']]
                if not meta or any(tr['type'] != 'SC_SC' or tr['src_idx'] not in crit or tr['dst_idx'] not in crit for tr in meta):
                    continue
                for c in range(n_locs):
                    terms = []
                    for t, tr in zip(ch['flows'], meta):
                        a, b = tr['src_idx'], tr['dst_idx']
                        if (a, c) not in x and (b, c) not in x:
                            continue
                        if (t, c) not in cross_c:
                            z = cross_c[t, c] = model.addVar(lb=0.0, ub=1.0, name=f"cross_{net['flow_meta'][t]['key']}_fail_{locations[c].id}")
                            for p, q in ((a, b), (b, a)):
                                if (p, c) not in x:
                                    continue
                                for j in feasible_locs_per_sc[p]:
                                    if (p, j) in net['r']:
                                        model.addConstr(z >= x[p, c] + 2 * net['r'][p, j] - x.get((q, j), 0) - net['r'].get((q, j), 0) - 2,
                                                        name=f"cross_{t}_fail_{locations[c].id}_{scs[p].id}_{locations[j].id}")
                        terms.append(min_crossing(t) * cross_c[t, c])
                    if terms:
                        model.addConstr(gp.quicksum(terms) <= ch['degraded'] - ch['const'], name=f"chain_lb_{ch['id']}_fail_{locations[c].id}")
                        n_fail += 1
            if net['chains']:
                print(f"  Chain latency bounds in the model: {n_nom} nominal, {n_fail} after a cell failure")
            model.update()
            print("\n[DEBUG] Gurobi model size (final):")
            print(f"  NumVars: {model.NumVars}")
            print(f"  NumConstrs: {model.NumConstrs}")
            bounds = [(var, var.LB, var.UB) for var in model.getVars()]
            self._session = {'key': key, 'bounds': bounds, 'parts': (model, x, P, y, hw_use, if_use, attach_s, attach_a, bus, comm,
                                                                     traffic_flows, net, feasible_locs_per_sc, feasible_scs_per_loc,
                                                                     objectives)} if reuse else None
        else:
            (model, x, P, y, hw_use, if_use, attach_s, attach_a, bus, comm, traffic_flows, net, feasible_locs_per_sc,
             feasible_scs_per_loc, objectives) = session['parts']
            if not resuming:
                for var, lb, ub in session['bounds']:
                    var.LB, var.UB = lb, ub
                model.update()
            print(f"\n  Model {'resumed: the search continues where the previous attempt stopped' if resuming else 'reused'} "
                  f"from the previous solve: {model.NumVars} variables, {model.NumConstrs} constraints; the translation is skipped")
        if session is None:
            self._round_caps = []
        elif not resuming and self._round_caps:
            model.remove(self._round_caps)
            self._round_caps = []
            model.update()


        if fixed and not resuming:
            sc_index = {sc.id: i for i, sc in enumerate(scs)}
            loc_index = {loc.id: j for j, loc in enumerate(locations)}
            n_fix = {'x': 0, 'r': 0}
            for kind, family in (('x', x), ('r', net['r'])):
                for sid, cell in (fixed.get(kind) or {}).items():
                    i, jj = sc_index.get(sid), loc_index.get(cell)
                    if i is None or jj is None or not any((i, j) in family for j in feasible_locs_per_sc[i]):
                        continue
                    for j in feasible_locs_per_sc[i]:
                        if (i, j) in family:
                            family[i, j].LB = family[i, j].UB = 1.0 if j == jj else 0.0
                    n_fix[kind] += 1
            n_allow = 0
            for sid, cells in (fixed.get('allow') or {}).items():
                i = sc_index.get(sid)
                if i is None:
                    continue
                ok = {loc_index[c] for c in cells if c in loc_index}
                for j in feasible_locs_per_sc[i]:
                    if j not in ok:
                        if (i, j) in x:
                            x[i, j].UB = 0.0
                        if (i, j) in net['r']:
                            net['r'][i, j].UB = 0.0
                n_allow += 1
            n_vars = 0
            if fixed.get('vars'):
                for v in model.getVars():
                    val = fixed['vars'].get(v.VarName)
                    if val is not None and v.VType != 'C':
                        v.LB = v.UB = float(round(val))
                        n_vars += 1
            for h, n_cap in (fixed.get('unit_cap') or {}).items():
                terms = [hw_use[j, h] for j in range(n_locs) if (j, h) in hw_use]
                if terms:
                    self._round_caps.append(model.addConstr(gp.quicksum(terms) <= int(n_cap), name=f"round_cap_{h}"))
                    print(f"  Unit cap of this solve: at most {int(n_cap)} {h} unit(s) in the whole car")
            model.update()
            print(f"  Fixed placements: {n_fix['x']} components, {n_fix['r']} standbys; restricted to their own cells: {n_allow}"
                  + (f"; {n_vars} variables fixed by name" if n_vars else ''))

        if start and not resuming:
            matched = 0
            for v in model.getVars():
                val = start.get(v.VarName)
                if val is not None:
                    v.Start = val
                    matched += 1
            print(f"  Start solution: {matched} of {model.NumVars} variables taken from the previous solve")
            if bus.get('gw') and not any(v.VarName in start for v in bus['gw'].values()):
                model.update()
                sv = lambda var: var.Start if var.Start < 1e100 else 0.0
                pos_s = {i: j for (i, j), var in x.items() if sv(var) > 0.5}
                pos_sr = {i: j for (i, j), var in (net.get('r') or {}).items() if sv(var) > 0.5}
                n_gw = 0
                for d, info in bus['devices'].items():
                    lines_d = [b for b in bus['inst'] if (b, d) in bus['dev'] and sv(bus['dev'][b, d]) > 0.5]
                    hosts = [pos_s[i] for i in info['users'] if i in pos_s]
                    stby = [pos_sr[i] for i in info['users'] if i in pos_sr]
                    taken = set()
                    for b in bus['inst']:
                        if (b, d) not in bus['dev']:
                            continue
                        members = [j for j in range(n_locs) if (d, b, j) in bus['gw'] and sv(bus['unit'][b, j]) > 0.5]
                        new = None
                        if b in lines_d:
                            k = lines_d.index(b)
                            prefer = [j for j in (hosts if k == 0 else stby + hosts) if j in members and j not in taken]
                            new = prefer[0] if prefer else next((j for j in members if j not in taken), members[0] if members else None)
                        for j in range(n_locs):
                            if (d, b, j) in bus['gw']:
                                bus['gw'][d, b, j].Start = 1.0 if j == new else 0.0
                        if new is not None:
                            taken.add(new)
                            n_gw += 1
                print(f"  Start solution: {n_gw} bus gateways set from the start's placements (the start had none)")

        if time_limit is not None:
            try:
                tl = float(time_limit)
                if tl > 0:
                    model.setParam('TimeLimit', tl)
                    print(f"  Time limit per solve: {tl:.1f} s")
            except Exception:
                print(f"  [WARN] Ignoring invalid time_limit value: {time_limit}")
        if mip_gap is not None:
            model.setParam('MIPGap', float(mip_gap))
            print(f"  MIPGap: {float(mip_gap)}")
        job_gb = job_memory_gb()
        mem_cfg = dict(self.memory or {})
        if job_gb and (mem_cfg.get('soft_share') or mem_cfg.get('hard_share')):
            rss_now, _rss_peak = process_memory_gb()
            outside = max(0.0, rss_now - float(model.MemUsed))
            limits = {}
            for name, share in (('SoftMemLimit', mem_cfg.get('soft_share')), ('MemLimit', mem_cfg.get('hard_share'))):
                if share:
                    limits[name] = max(1.0, float(share) * job_gb - outside)
                    model.setParam(name, limits[name])
            print(f"  [memory] job {job_gb:.0f} GB, process {rss_now:.1f} GB (Gurobi {float(model.MemUsed):.1f} GB): "
                  + ", ".join(f"{k} {v:.0f} GB" for k, v in limits.items()), flush=True)
        for item in (gurobi_params or '').split(','):
            if '=' in item:
                k, v = item.split('=', 1)
                model.setParam(k.strip(), float(v) if '.' in v else int(v))
                print(f"  Gurobi parameter {k.strip()} = {v.strip()}")
        print(f"\nSolving...")
        (primary_name, primary_expr), (secondary_name, secondary_expr) = objectives
        if lp_path:
            model.setObjective(primary_expr, GRB.MINIMIZE)
            model.write(lp_path)
            print(f"  Model written to: {lp_path}")

        tl = float(time_limit) if (time_limit is not None and float(time_limit) > 0) else None

        if relax_network:
            if tl is not None:
                model.setParam('TimeLimit', tl)
            model.setObjective(primary_expr, GRB.MINIMIZE)
            print(f"\n[Relaxation] placement-only bound: {model.NumVars} variables, {model.NumConstrs} constraints, no lazy rules", flush=True)
            model.optimize()
            has_sol = model.SolCount > 0
            info = {'status': self._phase_info(model, primary_name)['status'], 'bound': float(model.ObjBound) if has_sol or model.status == GRB.TIME_LIMIT else None,
                    'objective': float(model.ObjVal) if has_sol else None, 'mip_gap': float(model.MIPGap) if has_sol else None,
                    'runtime_s': float(model.Runtime), 'node_count': float(model.NodeCount)}
            if has_sol:
                info['partitions'] = {f"{locations[j].id}_asil{a}": int(round(v.X)) for (j, a), v in P.items() if v.X > 0.5}
                info['units'] = [locations[j].id for j, v in y.items() if v.X > 0.5]
                info['hardware'] = [f"{h}@{locations[j].id}" for (j, h), v in hw_use.items() if v.X > 0.5]
                info['n_partitions'] = int(round(sum(v.X for v in P.values())))
                if net.get('types'):
                    info['partition_types'] = {}
                    for (j, a, t), v in net['Q'].items():
                        if v.X > 0.5:
                            info['partition_types'].setdefault(f"{locations[j].id}_asil{a}", {})[net['types'][t]['id']] = int(round(v.X))
                info['assignment'] = {scs[i].id: locations[j].id for (i, j), v in x.items() if v.X > 0.5}
            print(f"[Relaxation] {info['status']}: bound {info['bound']}, incumbent {info['objective']}, gap {info['mip_gap']}, {info['runtime_s']:.0f} s", flush=True)
            self.last_status = 'RELAXATION'
            self._session = None
            model.dispose()
            gc.collect()
            return info

        model.setParam('LazyConstraints', 1)
        watched = []
        for t, (src, dst) in net['flow_ends'].items():
            watched += list(src.values()) + list(dst.values())
        watched += list(net['sel'].values()) + list(net['w'].values())
        watched += list(x.values()) + list(y.values()) + list(hw_use.values()) + list(P.values()) + list((net.get('Q') or {}).values()) \
            + list((net.get('hw_t') or {}).values())
        watched += list(net['r'].values()) + [v for v in net['hot'].values() if v is not None] + list(net['hr'].values())
        watched += list(attach_s.values()) + list(attach_a.values()) + list(bus['gw'].values())
        watched += list(bus['act'].values()) + list(bus['len'].values()) + list(bus['unit'].values()) + list(bus['dev'].values())
        seen_ids, cb_vars = set(), []
        for var in watched:
            if id(var) not in seen_ids:
                seen_ids.add(id(var)); cb_vars.append(var)
        comm_keys = list(comm.keys())
        comm_vars = [comm[k] for k in comm_keys]
        cap_of = net['cap_of']
        model._lazy_cuts = 0
        model._fail_cuts = 0
        base_variants, flows_of = {}, {i: [] for i in net['critical']}
        for tr in traffic_flows:
            si = tr.get('src_inst', 'a' if tr['type'] == 'SENS_SC' else 'x')
            di = tr.get('dst_inst', 'a' if tr['type'] == 'SC_ACT' else 'x')
            base_variants.setdefault(tr.get('base', tr['id']), {})[(si, di)] = tr['id']
            if 'base' not in tr:
                for role, key in (('src', 'src_idx'), ('dst', 'dst_idx')):
                    if tr.get(key) in flows_of:
                        flows_of[tr[key]].append((tr['id'], role))
        net['base_variants'], net['flows_of'] = base_variants, flows_of

        model._repair = None
        model._repairs_used = 0
        model._cb_stats = {'time_s': 0.0, 'candidates': 0, 'rejected': 0, 'repair_tries': 0, 'repair_valid': 0, 'repair_fail': {},
                           'rejected_by': {}, 'cuts_by': {}}
        model._cb_last_print = 0.0
        has_switch_hw = 'SWITCH' in all_hw
        if session is None:
            model._st_cuts, model._st_added = {}, set()

        def keep_steiner_cuts():
            for key, (members, st_len) in model._st_cuts.items():
                if key not in model._st_added:
                    b = key[0]
                    model.addConstr(bus['len'][b] >= st_len * (gp.quicksum(members) - (len(members) - 1)),
                                    name=f"steiner_{b[0]}{b[1] + 1}_{len(model._st_added)}")
                    model._st_added.add(key)

        if start and not resuming:
            for b in bus['inst']:
                if start.get(bus['act'][b].VarName, 0.0) < 0.5:
                    continue
                mem = [(locations[j].id, bus['unit'][b, j]) for j in range(n_locs) if start.get(bus['unit'][b, j].VarName, 0.0) > 0.5]
                mem += [(d, bus['dev'][b, d]) for d in bus['devices'] if (b, d) in bus['dev'] and start.get(bus['dev'][b, d].VarName, 0.0) > 0.5]
                if len(mem) > 1:
                    model._st_cuts[(b, tuple(sorted(e for e, _v in mem)))] = ([v for _e, v in mem], steiner_tree([e for e, _v in mem], bus['points'])[0])
            keep_steiner_cuts()
        cont_ids = {id(v) for v in cb_vars if v.VType == GRB.CONTINUOUS}

        def lazy_cb(m, where):
            if where == GRB.Callback.MIPNODE:
                if m._repair is not None and m.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
                    rv, rvals = zip(*m._repair.items())
                    m.cbSetSolution(list(rv), list(rvals))
                    obj = m.cbUseSolution()
                    m._repair = None
                    if obj < GRB.INFINITY:
                        m._repairs_used += 1
                return
            if where != GRB.Callback.MIPSOL:
                return
            cuts_before = m._lazy_cuts
            _t0 = time.perf_counter()
            m._cb_stats['candidates'] += 1
            vals = dict(zip((id(v) for v in cb_vars), m.cbGetSolution(cb_vars)))
            cv = dict(zip(comm_keys, m.cbGetSolution(comm_vars)))
            getv = lambda var: vals[id(var)]
            routes = flow_routes(net, getv)

            _ends, _silent, _fcfg = {}, {}, {}

            def ends_of_t(t):
                r = _ends.get(t)
                if r is None:
                    src, dst = net['flow_ends'][t]
                    r = _ends[t] = (next((k for k, var in src.items() if getv(var) > 0.5), None),
                                    next((k for k, var in dst.items() if getv(var) > 0.5), None))
                return r
            hot_val = {i: (1.0 if net['hot'][i] is None else getv(net['hot'][i])) for i in net['critical']}

            def _silent_scan(t):
                tr = net['flow_meta'][t]
                for inst, key in ((tr.get('src_inst', 'x'), 'src_idx'), (tr.get('dst_inst', 'x'), 'dst_idx')):
                    if inst == 'r' and hot_val.get(tr[key], 1.0) < 0.5:
                        return True
                return False

            def silent(t):
                s = _silent.get(t)
                if s is None:
                    s = _silent[t] = _silent_scan(t)
                return s

            def indicator(t, u, v, w_if, p):
                src, dst = net['flow_ends'][t]
                pair = (min(u, v), max(u, v))
                expr = src[u] + dst[v] + net['sel'][pair + (w_if, p)] - 2
                if len(net['flow_elig'][t]) > 1:
                    expr = expr + net['w'][t, w_if] - 1
                return expr

            def traffic_indicator(t, u, v, w_if, p):
                expr = indicator(t, u, v, w_if, p)
                tr = net['flow_meta'][t]
                for inst, key in ((tr.get('src_inst', 'x'), 'src_idx'), (tr.get('dst_inst', 'x'), 'dst_idx')):
                    if inst == 'r' and net['hot'].get(tr[key]) is not None:
                        expr = expr + net['hot'][tr[key]] - 1
                return expr

            def tally(name, before):
                d = m._lazy_cuts - before
                if d > 0:
                    m._cb_stats['cuts_by'][name] = m._cb_stats['cuts_by'].get(name, 0) + d
                    m._cb_stats['rejected_by'][name] = m._cb_stats['rejected_by'].get(name, 0) + 1
                return m._lazy_cuts

            _c = m._lazy_cuts
            bus_members = {}
            for b in bus['inst']:
                if getv(bus['act'][b]) < 0.5:
                    continue
                mem = [(locations[j].id, bus['unit'][b, j]) for j in range(n_locs) if getv(bus['unit'][b, j]) > 0.5]
                mem += [(d, bus['dev'][b, d]) for d in bus['devices'] if (b, d) in bus['dev'] and getv(bus['dev'][b, d]) > 0.5]
                st_len = steiner_tree([e for e, _v in mem], bus['points'])[0]
                bus_members[b] = st_len
                if getv(bus['len'][b]) < st_len - 1e-6:
                    m.cbLazy(bus['len'][b] >= st_len * (gp.quicksum(v for _e, v in mem) - (len(mem) - 1)))
                    m._lazy_cuts += 1
                    m._st_cuts[(b, tuple(sorted(e for e, _v in mem)))] = ([v for _e, v in mem], st_len)

            def resources(u, v, w_if, p):
                if w_if in bus['types']:
                    return [('line', (w_if, p))]
                pair = (min(u, v), max(u, v))
                path = net['paths'][pair + (w_if,)][p]
                seq = path if (u, v) == pair else path[::-1]
                return [(a, b2, w_if) for a, b2 in zip(seq, seq[1:])]

            def capacity_left(res, counts):
                if res[0] == 'line':
                    ln = res[1]
                    dev_load = sum(bus['devices'][d]['volume'] for d in bus['devices'] if (ln, d) in bus['dev'] and getv(bus['dev'][ln, d]) > 0.5)
                    return cap_of[ln[0]] * net['bus_umax'] - dev_load
                a, b2, w_if = res
                return cap_of[w_if] * counts[(a, b2, w_if) if a < b2 else (b2, a, w_if)]

            def capacity_cut(res, terms):
                if res[0] == 'line':
                    ln = res[1]
                    devs = [(bus['dev'][ln, d], bus['devices'][d]['volume']) for d in bus['devices'] if (ln, d) in bus['dev']]
                    m.cbLazy(cap_of[ln[0]] * net['bus_umax'] >= gp.quicksum(vol * dv for dv, vol in devs) + gp.quicksum(vol * ind for ind, vol in terms))
                else:
                    a, b2, w_if = res
                    key = (a, b2, w_if) if a < b2 else (b2, a, w_if)
                    m.cbLazy(cap_of[w_if] * comm[key] >= gp.quicksum(vol * ind for ind, vol in terms))
                m._lazy_cuts += 1

            _c = tally('bus length', _c)
            load, contrib = {}, {}
            for t, rt in routes.items():
                if rt is None:
                    continue
                u, v, w_if, p = rt
                pair = (min(u, v), max(u, v))
                plist = net['paths'][pair + (w_if,)]
                if p is None:
                    src, dst = net['flow_ends'][t]
                    rhs = src[u] + dst[v] - 1
                    if len(net['flow_elig'][t]) > 1:
                        rhs = rhs + net['w'][t, w_if] - 1
                    m.cbLazy(gp.quicksum(net['sel'][pair + (w_if, q)] for q in range(len(plist))) >= rhs)
                    m._lazy_cuts += 1
                    continue
                if silent(t):
                    continue
                vol = net['volumes'][t]
                for res in resources(u, v, w_if, p):
                    load[res] = load.get(res, 0.0) + vol
                    contrib.setdefault(res, []).append((traffic_indicator(t, u, v, w_if, p), vol))
            for res, ld in load.items():
                if ld > capacity_left(res, cv) + 1e-6:
                    capacity_cut(res, contrib[res])

            _c = tally('paths and capacity', _c)
            if net.get('benders'):
                ends_c = {t: ends_of_t(t) for t in net['flow_meta'] if not silent(t)}
                act = {}
                for t, (u, v) in ends_c.items():
                    if u is not None and v is not None:
                        src, dst = net['flow_ends'][t]
                        e = src[u] + dst[v] - 1
                        tr = net['flow_meta'][t]
                        for inst, key in ((tr.get('src_inst', 'x'), 'src_idx'), (tr.get('dst_inst', 'x'), 'dst_idx')):
                            if inst == 'r' and net['hot'].get(tr[key]) is not None:
                                e = e + net['hot'][tr[key]] - 1
                        act[t] = e
                cab = {key: cv[key] for key in comm_keys}
                sw_cells = {j for j in range(n_locs) if has_switch_hw and getv(hw_use[j, 'SWITCH']) > 0.5}
                res_b = routing_lp_cut(net, ends_c, cab, sw_cells, comm, hw_use, act)
                if res_b is not None:
                    m.cbLazy(res_b[0] <= 0)
                    m._lazy_cuts += 1
                    m._cb_stats['benders_value'] = m._cb_stats.get('benders_value', 0.0) + res_b[1]
            _c = tally('benders', _c)
            if net.get('benders_fail') and net['critical']:
                ends_all = {t: ends_of_t(t) for t in net['flow_meta']}
                cab = {key: cv[key] for key in comm_keys}
                sw_cells = {j for j in range(n_locs) if has_switch_hw and getv(hw_use[j, 'SWITCH']) > 0.5}
                px = {i: next((j for j in feasible_locs_per_sc[i] if getv(x[i, j]) > 0.5), None) for i in range(n_sc)}
                pr = {i: next((j for j in feasible_locs_per_sc[i] if getv(net['r'][i, j]) > 0.5), None) for i in net['critical']}
                for c_f in range(n_locs):
                    if not (getv(y[c_f]) > 0.5 or (has_switch_hw and getv(hw_use[c_f, 'SWITCH']) > 0.5)):
                        continue
                    res_f = failure_lp_cut(net, c_f, ends_all, cab, sw_cells, comm, hw_use, x, net['r'], px, pr)
                    if res_f is not None:
                        m.cbLazy(res_f[0] <= 0)
                        m._lazy_cuts += 1
            _c = tally('benders failure', _c)
            nom_msgs = resource_messages(net, routes, getv)
            qf = net['lat']['factor'] if net['lat']['robust'] else 1.0

            def msg_ind(msg, res, term_of_flow):
                return term_of_flow(msg[1]) if msg[0] == 'f' else bus['dev'][res[1], msg[1]]

            def queue_terms(me, ind_me, res, msgs, term_of_flow):
                pm, rate, out, block = message_prio(net, me), resource_rate(net, res), [], None
                for msg in msgs.get(res, []):
                    if msg == me:
                        continue
                    if message_prio(net, msg) <= pm:
                        out.append(qf * message_kbit(net, msg) / rate * (ind_me + msg_ind(msg, res, term_of_flow) - 1))
                    else:
                        kb = min(message_kbit(net, msg), frame_kbit(net, res))
                        if block is None or kb > block[0]:
                            block = (kb, msg)
                if block is not None:
                    out.append(qf * block[0] / rate * (ind_me + msg_ind(block[1], res, term_of_flow) - 1))
                return out

            def chain_terms(ch, picks, msgs, term_of_flow, add_terms, routes_now, failed_line=None):
                total, terms = ch['const'], []
                for t, ind_t in picks:
                    rt = routes_now[t]
                    total += crossing_ms(net, t, rt, msgs)
                    if not add_terms:
                        continue
                    u, v, w_if, p = rt
                    own = robust_ms(net, sum(net['kbit'][t] / net['cap_of'][w_if] + net['hop_ms'][w_if] for _r in resources(u, v, w_if, p)))
                    terms.append(own * ind_t)
                    for res in resources(u, v, w_if, p):
                        terms += queue_terms(('f', t), ind_t, res, msgs, term_of_flow)
                for d in ch['dev_ends']:
                    lanes = device_lines(net, d, getv, failed_line)
                    if not lanes:
                        continue
                    line = max(lanes, key=lambda b: queue_ms(net, ('line', b), ('d', d), msgs))
                    res = ('line', line)
                    total += qf * queue_ms(net, res, ('d', d), msgs)
                    if add_terms:
                        terms += queue_terms(('d', d), bus['dev'][line, d], res, msgs, term_of_flow)
                return total, terms

            def nominal_term(k):
                rt = routes[k]
                return traffic_indicator(k, rt[0], rt[1], rt[2], rt[3])

            def active_lane(t):
                return nominal_lane(net, t, getv)

            for ch in net['chains']:
                picks = []
                for t in ch['flows']:
                    t = active_lane(t)
                    rt = routes.get(t)
                    if rt is None or rt[3] is None:
                        continue
                    picks.append((t, indicator(t, *rt)))
                total, _ = chain_terms(ch, picks, nom_msgs, nominal_term, False, routes)
                if total > ch['budget'] + 1e-6:
                    if _CHAIN_DEBUG:
                        print(f"[chain debug] {ch.get('id')}: {total:.3f} ms > budget {ch['budget']:.3f} (const {ch['const']:.3f}); "
                              + "; ".join(f"flow {t} {routes[t][2]} {locations[routes[t][0]].id if isinstance(routes[t][0], int) else routes[t][0]}->"
                                          f"{locations[routes[t][1]].id if isinstance(routes[t][1], int) else routes[t][1]} "
                                          f"{crossing_ms(net, t, routes[t], nom_msgs):.3f} ms" for t, _i in picks)
                              + f"; devices {ch['dev_ends']}", flush=True)
                    _t, terms = chain_terms(ch, picks, nom_msgs, nominal_term, True, routes)
                    m.cbLazy(gp.quicksum(terms) <= ch['budget'] - ch['const'])
                    m._lazy_cuts += 1

            _c = tally('chain latency', _c)
            placement_ok = True
            pos_x = {i: next((j for j in feasible_locs_per_sc[i] if getv(x[i, j]) > 0.5), None) for i in range(n_sc)}
            pos_r = {i: next((j for j in feasible_locs_per_sc[i] if getv(net['r'][i, j]) > 0.5), None) for i in net['critical']}

            ends_of = ends_of_t

            def _flow_cfg_scan(t, routes_):
                u, v = ends_of(t)
                rt = routes_.get(t)
                if rt is None or rt[3] is None:
                    return None
                src, dst = net['flow_ends'][t]
                _u, _v, w_if, p = rt
                pair = (min(u, v), max(u, v))
                cfg = [src[u], dst[v], net['sel'][pair + (w_if, p)]]
                if len(net['flow_elig'][t]) > 1:
                    cfg.append(net['w'][t, w_if])
                return w_if, p, net['paths'][pair + (w_if,)][p], cfg

            def flow_cfg(t, routes_):
                e = _fcfg.get((id(routes_), t))
                if e is None or e[0] is not routes_:
                    e = _fcfg[(id(routes_), t)] = (routes_, _flow_cfg_scan(t, routes_))
                return e[1]

            def flow_alive(t, kind, elem, routes_):
                u, v = ends_of(t)
                if u is None or v is None:
                    return False, [], False
                src, dst = net['flow_ends'][t]
                if kind == 'cell' and elem in (u, v):
                    return False, [src[u] if u == elem else dst[v]], False
                tr_t = net['flow_meta'][t]
                if kind == 'line' and tr_t['type'] in ('BUS_SC', 'SC_BUS') and tr_t['line'] == elem:
                    return False, [src[u] if tr_t['type'] == 'BUS_SC' else dst[v]], False
                if u == v:
                    return True, [src[u], dst[v]], False
                c = flow_cfg(t, routes_)
                if c is None:
                    return False, [], True
                w_if, p, path, cfg = c
                cfg = list(cfg)
                if kind in ('cell', 'switch') and elem in path[1:-1]:
                    return False, cfg, False
                if kind == 'switch' and w_if == 'ETH' and elem in (u, v):
                    return False, cfg, False
                if kind == 'cable' and w_if not in bus['types']:
                    a, b, wtype = elem
                    if wtype == w_if and any({pa, pb} == {a, b} for pa, pb in zip(path, path[1:])):
                        return False, cfg, False
                if kind == 'line' and (w_if, p) == elem:
                    return False, cfg, False
                return True, cfg, False

            def instance_served(i, inst, kind, elem, routes_):
                cell = pos_x[i] if inst == 'x' else pos_r[i]
                if cell is None or (kind == 'cell' and elem == cell):
                    return False, [], False
                involved, missing = [], False
                for t_base, role in flows_of[i]:
                    tr = net['flow_meta'][t_base]
                    if tr['type'] in ('SENS_SC', 'SC_ACT'):
                        keys = [('a', inst)] if tr['type'] == 'SENS_SC' else [(inst, 'a')]
                        att = ends_of(t_base)[0 if tr['type'] == 'SENS_SC' else 1]
                        if kind in ('cell', 'switch') and elem == att:
                            continue
                    elif tr['type'] in ('BUS_SC', 'SC_BUS'):
                        keys = [k for k in base_variants[t_base] if (k[1] if tr['type'] == 'BUS_SC' else k[0]) == inst]
                    else:
                        partner = tr['dst_idx'] if role == 'src' else tr['src_idx']
                        if partner not in pos_r:
                            continue
                        keys = [((inst, pi) if role == 'src' else (pi, inst)) for pi in ('x', 'r')]
                    served = False
                    for key in keys:
                        t = base_variants[t_base].get(key)
                        if t is None:
                            continue
                        ok, vs, miss = flow_alive(t, kind, elem, routes_)
                        involved += vs
                        missing = missing or miss
                        if ok:
                            served = True
                            break
                    if not served:
                        return False, involved, missing
                return True, involved, False

            def scenario_list(routes_):
                classes = set(net['fail_cfg'].get('classes', ['cell', 'switch', 'cable']))
                has_sw = 'SWITCH' in all_hw
                scenarios = []
                if 'cell' in classes:
                    scenarios += [('cell', c) for c in range(n_locs) if getv(y[c]) > 0.5 or (has_sw and getv(hw_use[c, 'SWITCH']) > 0.5)]
                if 'switch' in classes and has_sw:
                    scenarios += [('switch', c) for c in range(n_locs) if getv(hw_use[c, 'SWITCH']) > 0.5]
                if 'cable' in classes:
                    if routes_ is routes:
                        scenarios += [('cable', key) for key in comm_keys if cv[key] > 0.5]
                    else:
                        used = {(min(u, v), max(u, v), w_if) for rt in routes_.values() if rt for u, v, w_if, _p in [rt] if w_if not in bus['types']}
                        scenarios += [('cable', key) for key in used]
                    scenarios += [('line', ln) for ln in sorted({(rt[2], rt[3]) for rt in routes_.values()
                                                                if rt and rt[2] in bus['types'] and rt[3] is not None})]
                    if bus['gateway']:
                        have = {e for k, e in scenarios if k == 'line'}
                        scenarios += [('line', b) for b in bus['inst'] if b not in have and getv(bus['act'][b]) > 0.5]
                return scenarios

            VARIANT_ORDER = {'SC_SC': [('x', 'x'), ('x', 'r'), ('r', 'x'), ('r', 'r')],
                             'SENS_SC': [('a', 'x'), ('a', 'r')], 'SC_ACT': [('x', 'a'), ('r', 'a')]}
            n_lanes = max((sum(1 for b in bus['inst'] if b[0] == n) for n in bus['types']), default=0)
            VARIANT_ORDER['BUS_SC'] = [(f'a{k}', inst) for inst in ('x', 'r') for k in range(n_lanes)]
            VARIANT_ORDER['SC_BUS'] = [(inst, f'a{k}') for inst in ('x', 'r') for k in range(n_lanes)]

            def degraded_check(states, routes_, add_cuts):
                violations = 0
                for st in states:
                    msgs = None
                    for ch in net['chains']:
                        picks, broken = [], False
                        for t_base in ch['flows']:
                            pick = st['picks'].get(t_base)
                            if pick is None or pick == 'missing':
                                broken = True
                                break
                            t, reasons = pick
                            u, v = ends_of(t)
                            if u == v:
                                continue
                            ind = indicator(t, *routes_[t])
                            for rv in reasons:
                                ind = ind + rv - 1
                            picks.append((t, ind))
                        if st['kind'] == 'line' and any(not device_lines(net, d, getv, st['elem']) for d in ch['dev_ends']):
                            broken = True
                        if broken:
                            continue
                        if msgs is None:
                            msgs = resource_messages(net, routes_, getv, carried=set(st['carried']),
                                                     failed_line=(st['elem'] if st['kind'] == 'line' else None))
                        term = (lambda k, st=st: degraded_term(k, st['carried'][k], routes_))
                        fl = st['elem'] if st['kind'] == 'line' else None
                        total, _ = chain_terms(ch, picks, msgs, term, False, routes_, fl)
                        if total > ch['degraded'] + 1e-6:
                            violations += 1
                            if _GW_DEBUG and not add_cuts and m._cb_stats['repair_tries'] <= 6:
                                print(f"[gateway debug] repair {m._cb_stats['repair_tries']} degraded: chain {ch['id']} {total:.3f} > {ch['degraded']} in scenario {st['kind']} {st['elem']}; picks {[(t, routes_.get(t)) for t, _i in picks]}", flush=True)
                            if add_cuts:
                                _t, terms = chain_terms(ch, picks, msgs, term, True, routes_, fl)
                                m.cbLazy(gp.quicksum(terms) <= ch['degraded'] - ch['const'])
                                m._lazy_cuts += 1
                                m._fail_cuts += 1
                return violations

            def scenario_check(routes_, add_cuts):
                violations = 0
                scenarios = scenario_list(routes_)
                for kind, elem in scenarios:
                    for i in net['critical']:
                        ok_x, vs_x, miss_x = instance_served(i, 'x', kind, elem, routes_)
                        if ok_x:
                            continue
                        ok_r, vs_r, miss_r = instance_served(i, 'r', kind, elem, routes_)
                        if ok_r:
                            continue
                        if miss_x or miss_r:
                            continue
                        violations += 1
                        if add_cuts:
                            vars_ = [x[i, pos_x[i]], net['r'][i, pos_r[i]]] + vs_x + vs_r
                            for t_base, role in flows_of[i]:
                                tr = net['flow_meta'][t_base]
                                k = tr.get('dst_idx' if role == 'src' else 'src_idx')
                                if k is None:
                                    continue
                                if pos_x.get(k) is not None:
                                    vars_.append(x[k, pos_x[k]])
                                if pos_r.get(k) is not None:
                                    vars_.append(net['r'][k, pos_r[k]])
                            uniq = list({id(v): v for v in vars_}.values())
                            m.cbLazy(gp.quicksum(uniq) <= len(uniq) - 1)
                            m._lazy_cuts += 1
                            m._fail_cuts += 1
                        break
                return violations

            def scenario_states(routes_):
                states = []
                for kind, elem in scenario_list(routes_):
                    carried, picks = {}, {}
                    for t in net['flow_meta']:
                        if not silent(t):
                            ok, _vs, _miss = flow_alive(t, kind, elem, routes_)
                            if ok:
                                carried[t] = None
                    for t_base, variants in base_variants.items():
                        reasons, pick = [], None
                        for key in lane_keys(net, t_base, VARIANT_ORDER, getv):
                            t = variants.get(key)
                            if t is None:
                                continue
                            ok, vs, miss = flow_alive(t, kind, elem, routes_)
                            if miss:
                                pick = 'missing'
                                break
                            if ok:
                                pick = (t, reasons)
                                carried.setdefault(t, reasons)
                                break
                            reasons = reasons + vs
                        picks[t_base] = pick
                    states.append({'kind': kind, 'elem': elem, 'carried': carried, 'picks': picks})
                return states

            def degraded_traffic(states, routes_):
                out = []
                for st in states:
                    load, members = {}, {}
                    for t, reasons in st['carried'].items():
                        u, v = ends_of(t)
                        if u == v:
                            continue
                        _u, _v, w_if, p = routes_[t]
                        for res in resources(u, v, w_if, p):
                            load[res] = load.get(res, 0.0) + net['volumes'][t]
                            members.setdefault(res, []).append((t, reasons))
                    out.append((load, members))
                return out

            def degraded_term(t, reasons, routes_):
                u, v = ends_of(t)
                _u, _v, w_if, p = routes_[t]
                if reasons is None:
                    return traffic_indicator(t, u, v, w_if, p)
                ind = indicator(t, u, v, w_if, p)
                for rv in reasons:
                    ind = ind + rv - 1
                return ind

            if net['critical']:
                states = scenario_states(routes)
                for load, members in degraded_traffic(states, routes):
                    for res, ld in load.items():
                        if ld > capacity_left(res, cv) + 1e-6:
                            capacity_cut(res, [(degraded_term(t, reasons, routes), net['volumes'][t]) for t, reasons in members[res]])
                            m._fail_cuts += 1
                _c = tally('capacity after failure', _c)
                if scenario_check(routes, True) > 0:
                    placement_ok = False
                _c = tally('failure survival', _c)
                if net['chains'] and degraded_check(states, routes, True) > 0 and not bus['gw']:
                    placement_ok = False
                _c = tally('degraded latency', _c)
                for c in range(n_locs):
                    if getv(y[c]) < 0.5:
                        continue
                    activated = [i for i in net['critical'] if pos_x[i] == c and pos_r[i] is not None and hot_val[i] < 0.5]
                    groups = {}
                    for i in activated:
                        groups.setdefault((pos_r[i], scs[i].asil_req), []).append(i)
                    for (j, a), act in groups.items():
                        H = site_derating(locations[j], uncertainty)
                        _types = net.get('types')
                        for res in (('cpu', 'ram', 'tops') if _types and any(ty.get('tops_cap', 0) > 0 for ty in _types) else ('cpu', 'ram')):
                            hr_ = H if res in ('cpu', 'tops') else 1.0
                            if _types:
                                cap_now = sum(getv(net['Q'][j, a, t]) * ty[f'{res}_cap'] for t, ty in enumerate(_types))
                                cap_cut = gp.quicksum(net['Q'][j, a, t] * ty[f'{res}_cap'] for t, ty in enumerate(_types) if ty[f'{res}_cap'] > 0)
                            else:
                                cap = float(partitions.get(f'{res}_cap', 0) or 0)
                                if cap <= 0:
                                    continue
                                cap_now, cap_cut = getv(P[j, a]) * cap, P[j, a] * cap
                            dem = lambda i, res=res: getattr(scs[i], f'{res}_req') * sw_multiplier(scs[i], uncertainty, res)
                            if res == 'tops' and not any(dem(i) > 0 for i in act):
                                continue
                            nominal = sum(dem(i) for i in feasible_scs_per_loc[j] if scs[i].asil_req == a and getv(x[i, j]) > 0.5)
                            hot_here = sum(dem(i) for i in net['critical'] if (i, j) in net['hr'] and scs[i].asil_req == a and getv(net['hr'][i, j]) > 0.5)
                            if nominal + hot_here + sum(dem(i) for i in act) <= cap_now * hr_ + 1e-6:
                                continue
                            placement_ok = False
                            expr = gp.quicksum(dem(i) * x[i, j] for i in feasible_scs_per_loc[j] if scs[i].asil_req == a)
                            expr += gp.quicksum(dem(i) * net['hr'][i, j] for i in net['critical'] if (i, j) in net['hr'] and scs[i].asil_req == a)
                            for i in act:
                                term = x[i, c] + net['r'][i, j] - 1
                                if net['hot'][i] is not None:
                                    term = term - net['hot'][i]
                                expr += dem(i) * term
                            m.cbLazy(expr <= cap_cut * hr_)
                            m._lazy_cuts += 1
                            m._fail_cuts += 1
                    by_site = {}
                    for i in activated:
                        by_site.setdefault(pos_r[i], []).append(i)
                    for j, act in by_site.items():
                        H = site_derating(locations[j], uncertainty)
                        for res in ('cpu', 'ram'):
                            site_cap = partitions.get(f'site_{res}_cap')
                            if site_cap is None:
                                continue
                            dem = lambda i, res=res: getattr(scs[i], f'{res}_req') * sw_multiplier(scs[i], uncertainty, res)
                            nominal = sum(dem(i) for i in feasible_scs_per_loc[j] if getv(x[i, j]) > 0.5)
                            hot_here = sum(dem(i) for i in net['critical'] if (i, j) in net['hr'] and getv(net['hr'][i, j]) > 0.5)
                            if nominal + hot_here + sum(dem(i) for i in act) <= float(site_cap) * (H if res == 'cpu' else 1.0) + 1e-6:
                                continue
                            placement_ok = False
                            expr = gp.quicksum(dem(i) * x[i, j] for i in feasible_scs_per_loc[j])
                            expr += gp.quicksum(dem(i) * net['hr'][i, j] for i in net['critical'] if (i, j) in net['hr'])
                            for i in act:
                                term = x[i, c] + net['r'][i, j] - 1
                                if net['hot'][i] is not None:
                                    term = term - net['hot'][i]
                                expr += dem(i) * term
                            m.cbLazy(expr <= float(site_cap) * (H if res == 'cpu' else 1.0))
                            m._lazy_cuts += 1
                            m._fail_cuts += 1

            _c = tally('standby capacity after failure', _c)

            def repair_routes():
                chain_flows = set()
                for ch in net['chains']:
                    for tb in ch['flows']:
                        chain_flows.update(base_variants.get(tb, {}).values())
                chain_devs = {d for ch in net['chains'] for d in ch['dev_ends']}
                kb = net['kbit']
                placed, line_load = {}, {}
                for (ln, d), var in bus['dev'].items():
                    if getv(var) > 0.5:
                        placed.setdefault(('line', ln), []).append((message_prio(net, ('d', d)), bus['devices'][d]['kbit'], d in chain_devs))
                        line_load[ln] = line_load.get(ln, 0.0) + bus['devices'][d]['volume']

                def wait_on(r, pm):
                    msgs_r = placed.get(r, [])
                    block = max((min(k, frame_kbit(net, r)) for p_, k, _c in msgs_r if p_ > pm), default=0.0)
                    return sum(k for p_, k, _c in msgs_r if p_ <= pm) + block

                def delay_to_chains(r, pm, k):
                    return sum((k if pm <= p_ else min(k, frame_kbit(net, r))) for p_, _k, c in placed.get(r, []) if c)
                switch_at = {j for j in range(n_locs) if has_switch_hw and getv(hw_use[j, 'SWITCH']) > 0.5}
                pair_path, out = {}, {}
                flows = list(net['flow_meta'])
                flows.sort(key=lambda t: (0 if t in chain_flows else 1, -kb[t]))
                for t in flows:
                    u, v = ends_of(t)
                    if u is None or v is None or u == v:
                        out[t] = None
                        continue
                    pair = (min(u, v), max(u, v))
                    quiet = silent(t)
                    best = None
                    for n in net['flow_elig'][t]:
                        plist = net['paths'].get(pair + (n,), [])
                        cands = [pair_path[pair + (n,)]] if pair + (n,) in pair_path else range(len(plist))
                        for p in cands:
                            if n in bus['types']:
                                ln = (n, p)
                                if getv(bus['unit'][ln, u]) < 0.5 or getv(bus['unit'][ln, v]) < 0.5:
                                    continue
                                if not quiet and line_load.get(ln, 0.0) + net['volumes'][t] > cap_of[n] * net['bus_umax'] + 1e-9:
                                    continue
                            elif any(r not in switch_at for r in plist[p][1:-1]):
                                continue
                            rate = cap_of[n]
                            res_list = resources(u, v, n, p)
                            pm = message_prio(net, ('f', t))
                            own = sum(kb[t] / rate + net['hop_ms'][n] + wait_on(r, pm) / rate for r in res_list)
                            impact = 0.0 if quiet else sum(delay_to_chains(r, pm, kb[t]) / rate for r in res_list)
                            score = (own if t in chain_flows else 1e-3 * own) + impact
                            if best is None or score < best[0] - 1e-12:
                                best = (score, n, p)
                    if best is None:
                        return None
                    _score, n, p = best
                    pair_path.setdefault(pair + (n,), p)
                    out[t] = (u, v, n, p)
                    if not quiet:
                        for r in resources(u, v, n, p):
                            placed.setdefault(r, []).append((message_prio(net, ('f', t)), kb[t], t in chain_flows))
                        if n in bus['types']:
                            line_load[(n, p)] = line_load.get((n, p), 0.0) + net['volumes'][t]
                return (out,)

            if m._lazy_cuts > cuts_before:
                m._cb_stats['rejected'] += 1
            if not placement_ok:
                m._cb_stats['repair_fail']['placement'] = m._cb_stats['repair_fail'].get('placement', 0) + 1
            if m._lazy_cuts > cuts_before and placement_ok and m._repair is None:
                m._cb_stats['repair_tries'] += 1
                stage = None
                if bus['gw']:
                    moved = 0
                    for d, info in bus['devices'].items():
                        if not info['users']:
                            continue
                        lines_d = [b for b in bus['inst'] if (b, d) in bus['dev'] and getv(bus['dev'][b, d]) > 0.5]
                        hosts = [pos_x[i] for i in info['users'] if pos_x.get(i) is not None]
                        stby = [pos_r[i] for i in info['users'] if pos_r.get(i) is not None]
                        taken = set()
                        for k, b in enumerate(lines_d):
                            members = [j for j in range(n_locs) if (b, j) in bus['unit'] and getv(bus['unit'][b, j]) > 0.5]
                            cur = next((j for j in members if (d, b, j) in bus['gw'] and getv(bus['gw'][d, b, j]) > 0.5), None)
                            prefer = [j for j in (hosts if k == 0 else stby + hosts) if j in members and j not in taken]
                            new = prefer[0] if prefer else (cur if cur not in taken else next((j for j in members if j not in taken), cur))
                            if new is not None and new != cur:
                                for j in members:
                                    if (d, b, j) in bus['gw']:
                                        vals[id(bus['gw'][d, b, j])] = 1.0 if j == new else 0.0
                                moved += 1
                            if new is not None:
                                taken.add(new)
                    if moved:
                        _ends.clear()
                        _fcfg.clear()
                fix = {var: (getv(var) if id(var) in cont_ids else round(getv(var))) for var in cb_vars}
                for var in net['sel'].values():
                    fix[var] = 0
                for b in bus['inst']:
                    fix[bus['len'][b]] = bus_members.get(b, 0.0)
                routed = repair_routes()
                feasible = routed is not None
                stage = None if feasible else 'no route option'
                if feasible and any(bus_members.get(b, 0.0) > bus.get('len_ub', {}).get(b, float('inf')) + 1e-6 for b in bus['inst']):
                    feasible, stage = False, 'bus length'
                rep_routes, rload, need = (routed[0] if routed else {}), {}, set()
                for t, rt in rep_routes.items():
                    if rt is None:
                        continue
                    u, v, w_if, p = rt
                    pair = (min(u, v), max(u, v))
                    if len(net['flow_elig'][t]) > 1:
                        for n in net['flow_elig'][t]:
                            fix[net['w'][t, n]] = 1 if n == w_if else 0
                    fix[net['sel'][pair + (w_if, p)]] = 1
                    if w_if not in bus['types']:
                        need.add(pair + (w_if,))
                    if not silent(t):
                        for res in resources(u, v, w_if, p):
                            key = res if res[0] == 'line' else (min(res[0], res[1]), max(res[0], res[1]), res[2])
                            rload[key] = rload.get(key, 0.0) + net['volumes'][t]
                rep_states = scenario_states(rep_routes) if (feasible and net['critical']) else []
                if feasible and net['critical']:
                    for load, _members in degraded_traffic(rep_states, rep_routes):
                        for res, ld in load.items():
                            key = res if res[0] == 'line' else (min(res[0], res[1]), max(res[0], res[1]), res[2])
                            rload[key] = max(rload.get(key, 0.0), ld)
                for key in comm_keys:
                    fix[comm[key]] = 0
                for key in need:
                    n_cab = max(1, int(np.ceil(rload.get(key, 0.0) / cap_of[key[2]] - 1e-9)))
                    if n_cab > net['comm_ub'][key[2]]:
                        feasible, stage = False, stage or 'cable count'
                    fix[comm[key]] = n_cab
                for key, ld in rload.items():
                    if key[0] == 'line' and ld > capacity_left(key, None) + 1e-6:
                        feasible, stage = False, stage or 'bus line capacity'
                if feasible and net['chains']:
                    cl_rep = chain_latencies(net, rep_routes, getv)
                    feasible = all(c['ok'] for c in cl_rep.values())
                    stage = stage or (None if feasible else 'chain latency')
                    if _GW_DEBUG and not feasible and m._cb_stats['repair_tries'] <= 6:
                        bad = {k: (round(c['latency_ms'], 3), c['budget_ms'], round(c['device_bus_ms'], 3), [(x['from'], x['to'], x['iface'], x['hops'], x['ms']) for x in c['crossings']])
                               for k, c in cl_rep.items() if not c['ok']}
                        print(f"[gateway debug] repair {m._cb_stats['repair_tries']} chain latency violations: {bad}", flush=True)
                if feasible and net['critical']:
                    feasible = scenario_check(rep_routes, False) == 0
                    stage = stage or (None if feasible else 'failure survival')
                if feasible and net['critical'] and net['chains']:
                    feasible = degraded_check(rep_states, rep_routes, False) == 0
                    stage = stage or (None if feasible else 'degraded latency')
                if feasible:
                    m._repair = fix
                    m._cb_stats['repair_valid'] += 1
                else:
                    m._cb_stats['repair_fail'][stage] = m._cb_stats['repair_fail'].get(stage, 0) + 1
            if _CB_VERIFY:
                for t, r in _ends.items():
                    src, dst = net['flow_ends'][t]
                    fresh = (next((k for k, var in src.items() if getv(var) > 0.5), None),
                             next((k for k, var in dst.items() if getv(var) > 0.5), None))
                    if fresh != r:
                        raise RuntimeError(f'route self-check failed: endpoints of flow {t}: {r} vs {fresh}')
                for t, s_ in _silent.items():
                    if _silent_scan(t) != s_:
                        raise RuntimeError(f'route self-check failed: silent flag of flow {t}')
                for (_rid, t), (robj, c) in _fcfg.items():
                    fresh = _flow_cfg_scan(t, robj)
                    same = (fresh is None and c is None) or (fresh is not None and c is not None and fresh[0] == c[0] and fresh[1] == c[1]
                                                              and list(fresh[2]) == list(c[2]) and [id(v) for v in fresh[3]] == [id(v) for v in c[3]])
                    if not same:
                        raise RuntimeError(f'route self-check failed: route configuration of flow {t}')
                m._cb_stats['verified'] = m._cb_stats.get('verified', 0) + 1
            if _GW_DEBUG and bus['gw'] and m._cb_stats['candidates'] <= 8:
                hosts = {d: sorted({locations[pos_x[i]].id for i in info['users'] if pos_x.get(i) is not None}) for d, info in bus['devices'].items() if info['users']}
                lanes_dbg = [f"{d}@{b[0]}{b[1] + 1}->{locations[j].id} (hosts {hosts.get(d)})" for (d, b, j), var in bus['gw'].items() if getv(var) > 0.5]
                print(f"[gateway debug] candidate {m._cb_stats['candidates']}: cuts by {dict(m._cb_stats['cuts_by'])}; lanes: {lanes_dbg}", flush=True)
            m._cb_stats['time_s'] = round(m._cb_stats['time_s'] + (time.perf_counter() - _t0), 3)
            now = time.time()
            if now - m._cb_last_print >= 60.0:
                m._cb_last_print = now
                st = m._cb_stats
                print(f"[callback] candidates {st['candidates']}, rejected {st['rejected']}, repair tries {st['repair_tries']}, "
                      f"valid repairs {st['repair_valid']}, used {m._repairs_used}, repair failures {st['repair_fail']}, "
                      f"rejected by {st['rejected_by']}, cuts by {st['cuts_by']}, callback time {st['time_s']:.0f} s", flush=True)

        def _lns_improve(obj_expr, info, cfg, budget_s):
            rng = random.Random(int(cfg.get('seed', 1)))
            free_share, iter_s = float(cfg.get('free_share', 0.3)), float(cfg.get('iter_s', 120))
            all_vars = model.getVars()
            best_val, best = obj_expr.getValue(), model.getAttr('X', all_vars)
            bound = info['bound']
            x_items = list(x.items())
            t_end, it, improved = time.time() + budget_s, 0, 0
            print(f"\n[LNS] start {best_val:.4f}, bound {bound:.4f}, budget {budget_s:.0f} s, free share {free_share}, {iter_s:.0f} s per iteration", flush=True)
            while time.time() < t_end - 10 and (best_val - bound) / max(1e-9, abs(best_val)) > float(cfg.get('stop_gap', 0.01)):
                it += 1
                free = {i for i in range(n_sc) if rng.random() < free_share}
                vals = dict(zip((v.VarName for v in all_vars), best))
                for (i, j), var in x_items:
                    if i not in free:
                        var.LB = var.UB = round(vals[var.VarName])
                keep_steiner_cuts()
                model.setAttr('Start', all_vars, best)
                model.setParam('TimeLimit', min(iter_s, max(10.0, t_end - time.time())))
                model.optimize(lazy_cb)
                if model.SolCount > 0 and obj_expr.getValue() < best_val - 1e-6:
                    best_val, best, improved = obj_expr.getValue(), model.getAttr('X', all_vars), improved + 1
                    print(f"[LNS] iteration {it}: improved to {best_val:.4f} (gap to the phase-1 bound {100 * (best_val - bound) / best_val:.2f} %)", flush=True)
                for (i, j), var in x_items:
                    var.LB, var.UB = 0.0, 1.0
            vals = dict(zip((v.VarName for v in all_vars), best))
            for (i, j), var in x_items:
                var.LB = var.UB = round(vals[var.VarName])
            keep_steiner_cuts()
            model.setAttr('Start', all_vars, best)
            model.setParam('TimeLimit', 120.0)
            model.optimize(lazy_cb)
            for (i, j), var in x_items:
                var.LB, var.UB = 0.0, 1.0
            if model.SolCount > 0:
                best_val = min(best_val, obj_expr.getValue())
            info['objective'] = float(best_val)
            info['mip_gap'] = float((best_val - bound) / max(1e-9, abs(best_val)))
            print(f"[LNS] done: {it} iterations, {improved} improvements, final {best_val:.4f}, gap {100 * info['mip_gap']:.2f} %", flush=True)
            return {'iterations': it, 'improvements': improved, 'final': float(best_val)}

        def _solve_lexicographic():
            if tl is not None:
                model.setParam('TimeLimit', tl)
            if not resuming:
                model.setObjective(primary_expr, GRB.MINIMIZE)
            phases = []
            print(f"\n[Lexicographic] Phase 1: minimize {primary_name}")
            lns_cfg = dict(lns or {})
            lns_on = bool(lns_cfg.get('enabled')) and tl is not None
            if lns_on:
                model.setParam('TimeLimit', max(60.0, tl * (1.0 - float(lns_cfg.get('time_share', 0.5)))))
            model.optimize(lazy_cb)
            phases.append(self._phase_info(model, primary_name))
            print(f"[callback] after phase 1: {model._cb_stats}, repairs used {model._repairs_used}", flush=True)
            if model.status == GRB.MEM_LIMIT and model.SolCount > 0:
                print("\n[Lexicographic] Phase 1 stopped at the soft memory limit with a design; phase 2 is skipped", flush=True)
                phases.append({'objective_name': secondary_name, 'status': 'SKIPPED', 'objective': None, 'bound': None,
                               'mip_gap': None, 'runtime_s': 0.0, 'node_count': 0.0})
                return phases, None
            if lns_on and model.SolCount > 0 and model.status == GRB.TIME_LIMIT and phases[-1]['mip_gap'] is not None \
                    and phases[-1]['mip_gap'] > float(lns_cfg.get('stop_gap', 0.01)):
                phases[-1]['lns'] = _lns_improve(primary_expr, phases[-1], lns_cfg, tl - model.Runtime)
            lex_constr = None
            if model.SolCount > 0 and model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
                primary_val = primary_expr.getValue()
                all_vars = model.getVars()
                phase1 = model.getAttr('X', all_vars)
                keep_steiner_cuts()
                lex_constr = model.addConstr(
                    primary_expr <= primary_val + max(1e-2, 1e-6 * abs(primary_val)),
                    name=f"lex_fix_{primary_name}"
                )
                model.setObjective(secondary_expr, GRB.MINIMIZE)
                model.setAttr('Start', all_vars, phase1)
                if tl is not None:
                    model.setParam('TimeLimit', float(phase2_cap) if phase2_cap is not None else max(tl - model.Runtime, float(phase2_time)))
                print(f"\n[Lexicographic] Phase 2: minimize {secondary_name} with {primary_name} <= {primary_val:.6f}")
                model.optimize(lazy_cb)
                phases.append(self._phase_info(model, secondary_name))
                print(f"[callback] after phase 2: {model._cb_stats}, repairs used {model._repairs_used}", flush=True)
                if model.SolCount == 0:
                    print(f"\n[Lexicographic] Phase 2 found no solution in its time; the phase-1 design is kept")
                    model.remove(lex_constr)
                    lex_constr = None
                    keep_steiner_cuts()
                    model.setObjective(primary_expr, GRB.MINIMIZE)
                    model.setAttr('Start', all_vars, phase1)
                    model.setParam('TimeLimit', float(phase2_time))
                    model.optimize(lazy_cb)
                    phases[-1]['status'] = 'NO_SOLUTION'
            return phases, lex_constr

        partition_map, part_type = {}, {}
        packing_cuts, unpackable = 0, False
        while True:
            try:
                phases, lex_constr = _solve_lexicographic()
            except gp.GurobiError as e:
                if e.errno != GRB.Error.OUT_OF_MEMORY:
                    raise
                rss_now, rss_peak = process_memory_gb()
                print(f"\n✗ Gurobi stopped at its memory limit (MemLimit {model.Params.MemLimit:.1f} GB; process {rss_now:.1f} GB now, "
                      f"peak {rss_peak:.1f} GB): no design from this solve; the model is dropped and rebuilt by the next solve", flush=True)
                self.last_status = 'MEM_LIMIT'
                self._resumable = False
                self._session = None
                model.dispose()
                gc.collect()
                return None
            if not (model.SolCount > 0 and model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.MEM_LIMIT)):
                break
            partition_map, failed, part_type = pack_partitions(x, P, scs, locations, partitions, uncertainty, standby=net['hr'],
                                                               Q=net.get('Q') if net.get('types') else None,
                                                               r=net.get('r'), hot=net.get('hot'), critical=net.get('critical'))
            if not failed:
                break
            if packing_cuts >= MAX_PACKING_CUTS:
                print(f"\n[Packing] {len(failed)} (site, ASIL) group(s) still not packable after {packing_cuts} cuts: no design from this solve")
                unpackable = True
                break
            for (j, a, members, p_val, counts, demand, h_j, res_p) in failed:
                loc_id = locations[j].id
                scen = sorted({k[2] for k, _v in members if len(k) == 3})
                scen_txt = f" after the failure of {', '.join(locations[c].id for c in scen)}" if scen else ""
                scen_tag = ("_f" + "_".join(locations[c].id for c in scen)) if scen else ""
                if counts is None:
                    present = gp.quicksum(var for _key, var in members) - len(members) + 1
                    model.addConstr(P[j, a] >= (p_val + 1) * present, name=f"packing_cut_{packing_cuts}_{loc_id}_asil{a}{scen_tag}")
                    print(f"\n[Packing] {len(members)} SCs of ASIL {a} at {loc_id}{scen_txt} do not fit into {p_val} partition(s); cut added")
                else:
                    from optimizer_utils import _type_bins, minimal_unpackable, failing_mixes, useful_types
                    types_p = net['types']
                    var_of = dict(members)
                    card_cuts = 0
                    act_vars = net.get('act') or {}
                    cand = [i for i in feasible_scs_per_loc[j] if scs[i].asil_req == a]
                    for r_ in res_p:
                        der_r = h_j if r_ in ('cpu', 'tops') else 1.0
                        sizes = sorted((demand[k][r_] for k in demand if demand[k][r_] > 0), reverse=True)
                        for k_top, theta in enumerate(sizes, start=1):
                            slots_now = sum(int((types_p[t][f'{r_}_cap'] * der_r) // theta) * counts[t] for t in range(len(types_p)))
                            if k_top <= slots_now:
                                continue
                            size_of = lambda i: float(getattr(scs[i], f'{r_}_req', 0) or 0) * sw_multiplier(scs[i], uncertainty, r_)
                            n_cells_cut = 0
                            for j2 in range(n_locs):
                                big2 = [i for i in feasible_scs_per_loc[j2] if scs[i].asil_req == a and size_of(i) >= theta - 1e-9]
                                if not big2:
                                    continue
                                der2 = site_derating(locations[j2], uncertainty) if r_ in ('cpu', 'tops') else 1.0
                                rhs2 = gp.quicksum(int((types_p[t][f'{r_}_cap'] * der2) // theta) * net['Q'][j2, a, t] for t in range(len(types_p))
                                                   if (types_p[t][f'{r_}_cap'] * der2) // theta >= 1)
                                tag2 = f"packing_card_{packing_cuts}_{locations[j2].id}_asil{a}_{r_}"
                                if r_ == 'rom':
                                    model.addConstr(gp.quicksum(x[i, j2] for i in big2) + gp.quicksum(net['r'][i, j2] for i in big2 if (i, j2) in net['r']) <= rhs2, name=tag2)
                                else:
                                    base2 = gp.quicksum(x[i, j2] for i in big2) + gp.quicksum(net['hr'][i, j2] for i in big2 if (i, j2) in net['hr'])
                                    model.addConstr(base2 <= rhs2, name=tag2)
                                    for c2 in range(n_locs):
                                        woken2 = [act_vars[i, c2, j2] for i in big2 if (i, c2, j2) in act_vars]
                                        if woken2:
                                            model.addConstr(base2 + gp.quicksum(woken2) <= rhs2, name=f"{tag2}_f{locations[c2].id}")
                                n_cells_cut += 1
                            card_cuts += 1
                            print(f"\n[Packing] {len(members)} SCs of ASIL {a} at {loc_id}{scen_txt}: the {k_top} largest {r_} items ({theta:.0f} each or more) "
                                  f"exceed the {slots_now} slot(s) of the opened blocks: cardinality cut at {n_cells_cut} cells (nominal and after every cell failure)")
                            break
                    if card_cuts:
                        packing_cuts += 1
                        continue
                    _order, bins = _type_bins(types_p, counts, h_j, res_p)
                    core = minimal_unpackable([k for k, _v in members], demand, bins)
                    useful = useful_types(core, demand, types_p, h_j, res_p) or list(range(len(types_p)))
                    mixes = failing_mixes(core, demand, types_p, counts, h_j, res_p, partitions.get('site_max_partitions'), useful)
                    present = gp.quicksum(var_of[k] for k in core) - len(core) + 1
                    for m_no, c in enumerate(mixes):
                        tag = f"packing_cut_{packing_cuts}_{m_no}_{loc_id}_asil{a}{scen_tag}"
                        z = {t: model.addVar(vtype=GRB.BINARY, name=f"{tag}_z{t}") for t in useful}
                        for t, zt in z.items():
                            model.addConstr(net['Q'][j, a, t] >= (c[t] + 1) * zt, name=f"{tag}_t{t}")
                        model.addConstr(gp.quicksum(z.values()) >= present, name=tag)
                    mix_txt = ', '.join(f"{types_p[t]['id']} {n}" for t, n in enumerate(counts) if n)
                    extra = [types_p[t]['id'] for c in mixes[1:] for t in range(len(c)) if c[t] > counts[t]]
                    print(f"\n[Packing] {len(members)} SCs of ASIL {a} at {loc_id}{scen_txt} do not fit into {p_val} partition(s) ({mix_txt}): "
                          f"core of {len(core)} SCs, useful types {', '.join(types_p[t]['id'] for t in useful)}, {len(mixes)} mix(es) cut"
                          + (f" (one block more of {', '.join(extra)} fails too)" if extra else ""))
                packing_cuts += 1
            if lex_constr is not None:
                model.remove(lex_constr)
            model.setObjective(primary_expr, GRB.MINIMIZE)
            keep_steiner_cuts()
            print(f"[Packing] Solving again with {packing_cuts} cut(s)")

        result = None
        if unpackable:
            self.last_status = 'UNPACKABLE'
        elif model.SolCount > 0 and model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.MEM_LIMIT):
            time_limited = any(ph['status'] != 'OPTIMAL' for ph in phases)
            if time_limited:
                print(f"\n⚠ Time limit reached; keeping best solution found (gaps: " + ", ".join(
                    f"{ph['objective_name']} " + (f"{100.0 * ph['mip_gap']:.4f}%" if ph['mip_gap'] is not None else ph['status']) for ph in phases) + ")")
            else:
                print(f"\n✓ Optimal solution found!")
            formatted_solution = extract_solution(
                x, P, y, hw_use, if_use, comm, scs, locations, sensors, actuators,
                cable_types, partitions, hardwares, interfaces, comm_matrix=comm_matrix,
                traffic_flows=traffic_flows, net=net, attach_s=attach_s, attach_a=attach_a, bus=bus,
                partition_map=partition_map, standby=(net['r'], net['hot']), uncertainty=uncertainty, part_type=part_type
            )
            formatted_solution['solve_info'] = {
                'objective_type': 'min_cable_length' if minimize_cable_length else 'min_cost',
                'objectives': [primary_name, secondary_name],
                'cable_length_limit': cable_length_limit,
                'cost_limit': cost_limit,
                'phases': phases,
                'packing_cuts': packing_cuts,
                'backbone_candidate_paths': net['k_paths'],
                'backbone_interfaces': net['bb_ifaces'],
                'lazy_cuts': int(getattr(model, '_lazy_cuts', 0)),
                'chains': len(net.get('chains', [])),
                'critical': len(net['critical']),
                'failure_cuts': int(getattr(model, '_fail_cuts', 0)),
                'repairs_used': int(getattr(model, '_repairs_used', 0)),
                'callback_stats': dict(getattr(model, '_cb_stats', {})),
                'gurobi_params': gurobi_params,
                'runtime_s': sum(ph['runtime_s'] for ph in phases),
                'num_vars': int(model.NumVars),
                'num_constrs': int(model.NumConstrs),
                'num_bin_vars': int(model.NumBinVars),
                'num_int_vars': int(model.NumIntVars),
                'lp_file': lp_path,
                'uncertainty': {k: bool((uncertainty or {}).get(k)) for k in UNCERTAINTY_BLOCKS},
                'growth_alpha': float((uncertainty or {}).get('growth_alpha', 0.0)) if (uncertainty or {}).get('growth') else None,
                'rho': float((uncertainty or {}).get('rho', 1.0)) if any((uncertainty or {}).get(k) for k in UNCERTAINTY_BLOCKS) else None,
            }
            formatted_solution['status'] = ('MEM_LIMIT' if any(ph['status'] == 'MEM_LIMIT' for ph in phases)
                                            else 'TIME_LIMIT' if time_limited else 'OPTIMAL')
            names = model.getAttr('VarName', model.getVars())
            if len(set(names)) != len(names):
                print(f"  [WARN] {len(names) - len(set(names))} duplicate variable names; start solutions may be incomplete")
            self.last_values = dict(zip(names, model.getAttr('X', model.getVars())))
            self.last_status = formatted_solution['status']
            result = formatted_solution
        elif model.status == GRB.TIME_LIMIT:
            print(f"\n✗ Optimization reached time limit without any feasible solution. Skipping this Pareto point.")
            self.last_status = 'TIME_LIMIT'
            self._resumable = bool(reuse)
        elif model.status == GRB.MEM_LIMIT:
            print(f"\n✗ Optimization stopped at the soft memory limit without any feasible solution.")
            self.last_status = 'MEM_LIMIT'
        else:
            print(f"\n✗ Optimization failed. Status: {model.status}")
            self.last_status = 'INFEASIBLE' if model.status == GRB.INFEASIBLE else str(model.status)
        if lex_constr is not None:
            model.remove(lex_constr)
        rss_now, rss_peak = process_memory_gb()
        print(f"  [memory] after the solve: process {rss_now:.1f} GB now, peak {rss_peak:.1f} GB; Gurobi peak {float(model.MaxMemUsed):.1f} GB", flush=True)
        return result

    @staticmethod
    def _phase_info(model, objective_name):
        status_names = {GRB.OPTIMAL: 'OPTIMAL', GRB.TIME_LIMIT: 'TIME_LIMIT', GRB.INFEASIBLE: 'INFEASIBLE',
                        GRB.INF_OR_UNBD: 'INF_OR_UNBD', GRB.UNBOUNDED: 'UNBOUNDED', GRB.MEM_LIMIT: 'MEM_LIMIT'}
        has_sol = model.SolCount > 0
        return {
            'objective_name': objective_name,
            'status': status_names.get(model.status, str(model.status)),
            'objective': float(model.ObjVal) if has_sol else None,
            'bound': float(model.ObjBound) if has_sol else None,
            'mip_gap': float(model.MIPGap) if has_sol else None,
            'runtime_s': float(model.Runtime),
            'node_count': float(model.NodeCount),
        }

    def optimize_growth_sweep(self, scs, locations, sensors, actuators, cable_types, comm_matrix,
                              partitions=None, hardwares=None, interfaces=None, solve_time_limit=None, mip_gap=None,
                              lp_dir=None, uncertainty=None, backbone=None, phase2_time=600, gurobi_params=None,
                              chains=None, lns=None, **_ignored):
        params = (uncertainty or {}).get('params', {})
        levels = [float(a) for a in params.get('rho', {}).get('levels', params.get('growth', {}).get('alpha_levels', [0.0]))]
        print("=" * 80)
        print(f"ROBUSTNESS SWEEP (min cost per robustness level): rho in {levels}")
        print("=" * 80)
        solutions = []
        self.last_values = None
        for alpha in levels:
            unc = dict(uncertainty or {})
            unc['rho'] = alpha
            unc['growth_alpha'] = alpha
            if alpha <= 0.0:
                unc['failures'] = False
                unc['redundancy'] = False
            print(f"\n[Robustness] rho = {alpha:.2f}")
            unc = with_gradual_standbys(unc, scs, chains)
            sol = self.optimize(
                scs, locations, sensors, actuators, cable_types, comm_matrix,
                partitions=partitions, hardwares=hardwares, interfaces=interfaces,
                time_limit=solve_time_limit, mip_gap=mip_gap,
                lp_path=(os.path.join(lp_dir, f"model_alpha_{alpha:g}.lp.gz") if lp_dir else None),
                uncertainty=unc, backbone=backbone, phase2_time=phase2_time, gurobi_params=gurobi_params,
                chains=chains, start=self.last_values, lns=lns)
            if sol:
                solutions.append(sol)
            else:
                print(f"[Robustness] rho = {alpha:.2f}: no design found (infeasible, or no feasible design within the time limit); sweep stopped")
                break
        print(f"Robustness sweep: {len(solutions)} of {len(levels)} levels solved")
        return solutions

    def optimize_pareto_epsilon_constraint(self, scs, locations, sensors, actuators, cable_types, comm_matrix,
                                           partitions=None, hardwares=None, interfaces=None,
                                           enable_comm_bw_constraints=True, comm_bw_big_m=10000, num_points=5,
                                           solve_time_limit=None, mip_gap=None, lp_dir=None, uncertainty=None, backbone=None, phase2_time=600, gurobi_params=None, chains=None,
                                           lns=None, **_ignored):
        print("=" * 80)
        print("PARETO OPTIMIZATION (total_cost vs cable_length)")
        print("=" * 80)

        def _lp(name):
            return os.path.join(lp_dir, f"model_{name}.lp.gz") if lp_dir else None

        self.last_values = None
        sol_min_cost = self.optimize(
            scs, locations, sensors, actuators, cable_types, comm_matrix,
            partitions=partitions, hardwares=hardwares, interfaces=interfaces,
            enable_comm_bw_constraints=enable_comm_bw_constraints,
            comm_bw_big_m=comm_bw_big_m,
            time_limit=solve_time_limit,
            mip_gap=mip_gap,
            lp_path=_lp("min_cost"),
            uncertainty=uncertainty, backbone=backbone, phase2_time=phase2_time, gurobi_params=gurobi_params, chains=chains, lns=lns,
            start=self.last_values
        )

        if num_points is not None and int(num_points) == 1:
            print("[Pareto] num_points = 1: only the min-cost extreme is solved")
            return [sol_min_cost] if sol_min_cost else []

        sol_min_len = self.optimize(
            scs, locations, sensors, actuators, cable_types, comm_matrix,
            partitions=partitions, hardwares=hardwares, interfaces=interfaces,
            enable_comm_bw_constraints=enable_comm_bw_constraints,
            comm_bw_big_m=comm_bw_big_m,
            minimize_cable_length=True,
            time_limit=solve_time_limit,
            mip_gap=mip_gap,
            lp_path=_lp("min_cable"),
            uncertainty=uncertainty, backbone=backbone, phase2_time=phase2_time, gurobi_params=gurobi_params, chains=chains, lns=lns,
            start=self.last_values
        )

        pareto_solutions = []
        seen = set()

        def _add_unique(sol):
            if not sol or sol.get('status') not in ('OPTIMAL', 'TIME_LIMIT'):
                return
            key = (round(float(sol.get('total_cost', 0.0)), 4), round(float(sol.get('cable_length', 0.0)), 4))
            if key in seen:
                return
            seen.add(key)
            pareto_solutions.append(sol)

        _add_unique(sol_min_cost)
        _add_unique(sol_min_len)

        if not sol_min_cost or not sol_min_len:
            return pareto_solutions

        cost_at_min_cost  = float(sol_min_cost.get('total_cost', 0.0))
        cable_at_min_cost = float(sol_min_cost.get('cable_length', 0.0))
        cost_at_min_cable  = float(sol_min_len.get('total_cost', 0.0))
        cable_at_min_cable = float(sol_min_len.get('cable_length', 0.0))

        if num_points is None:
            num_points = 5
        num_points = max(2, int(num_points))

        cable_range = abs(cable_at_min_cost - cable_at_min_cable)
        cost_range  = abs(cost_at_min_cable - cost_at_min_cost)
        cable_mid   = max(1e-9, (cable_at_min_cost + cable_at_min_cable) / 2.0)
        cost_mid    = max(1e-9, (cost_at_min_cost + cost_at_min_cable) / 2.0)

        cable_relative = cable_range / cable_mid
        cost_relative  = cost_range / cost_mid

        print(f"\n[Pareto] Cable range: {cable_at_min_cable:.2f} - {cable_at_min_cost:.2f} (span={cable_range:.2f}, relative={cable_relative:.2%})")
        print(f"[Pareto] Cost  range: {cost_at_min_cost:.2f} - {cost_at_min_cable:.2f} (span={cost_range:.2f}, relative={cost_relative:.2%})")

        if num_points > 2:
            if cost_relative > cable_relative:
                c_min = cost_at_min_cost
                c_max = cost_at_min_cable
                print(f"[Pareto] → Sweeping on COST axis (relative {cost_relative:.2%} > {cable_relative:.2%})")
                print(f"[Pareto]   cost_min={c_min:.2f}, cost_max={c_max:.2f}")
                eps_values = np.linspace(c_min, c_max, num_points)[1:-1]
                for k, eps_cost in enumerate(eps_values, 1):
                    print(f"\n[Pareto] Solving with cost_limit={eps_cost:.2f} (minimize cable_length)...")
                    sol_eps = self.optimize(
                        scs, locations, sensors, actuators, cable_types, comm_matrix,
                        partitions=partitions, hardwares=hardwares, interfaces=interfaces,
                        enable_comm_bw_constraints=enable_comm_bw_constraints,
                        comm_bw_big_m=comm_bw_big_m,
                        minimize_cable_length=True,
                        cost_limit=float(eps_cost) + 1e-2,
                        time_limit=solve_time_limit,
                        mip_gap=mip_gap,
                        lp_path=_lp(f"eps_cost_{k}"),
                        uncertainty=uncertainty, backbone=backbone, phase2_time=phase2_time, gurobi_params=gurobi_params, chains=chains, lns=lns,
                        start=self.last_values
                    )
                    _add_unique(sol_eps)
            else:
                l_max = cable_at_min_cost
                l_min = cable_at_min_cable
                print(f"[Pareto] → Sweeping on CABLE axis (relative {cable_relative:.2%} >= {cost_relative:.2%})")
                print(f"[Pareto]   cable_min={l_min:.2f}, cable_max={l_max:.2f}")
                eps_values = np.linspace(l_max, l_min, num_points)[1:-1]
                for k, eps_len in enumerate(eps_values, 1):
                    print(f"\n[Pareto] Solving with cable_length_limit={eps_len:.4f}...")
                    sol_eps = self.optimize(
                        scs, locations, sensors, actuators, cable_types, comm_matrix,
                        partitions=partitions, hardwares=hardwares, interfaces=interfaces,
                        enable_comm_bw_constraints=enable_comm_bw_constraints,
                        comm_bw_big_m=comm_bw_big_m,
                        cable_length_limit=float(eps_len) + 1e-6,
                        time_limit=solve_time_limit,
                        mip_gap=mip_gap,
                        lp_path=_lp(f"eps_cable_{k}"),
                        uncertainty=uncertainty, backbone=backbone, phase2_time=phase2_time, gurobi_params=gurobi_params, chains=chains, lns=lns,
                        start=self.last_values
                    )
                    _add_unique(sol_eps)

        pareto_solutions.sort(key=lambda s: (float(s.get('total_cost', 0.0)), float(s.get('cable_length', 0.0))))

        if len(pareto_solutions) > 1:
            non_dominated = []
            for sol in pareto_solutions:
                c = float(sol.get('total_cost', 0.0))
                l = float(sol.get('cable_length', 0.0))
                dominated = False
                for other in pareto_solutions:
                    if other is sol:
                        continue
                    oc = float(other.get('total_cost', 0.0))
                    ol = float(other.get('cable_length', 0.0))
                    if oc <= c and ol <= l and (oc < c or ol < l):
                        dominated = True
                        break
                if not dominated:
                    non_dominated.append(sol)
            if non_dominated:
                removed = len(pareto_solutions) - len(non_dominated)
                if removed > 0:
                    print(f"[Pareto] Removed {removed} dominated solution(s)")
                pareto_solutions = non_dominated

        print(f"Pareto solutions found: {len(pareto_solutions)}")
        return pareto_solutions

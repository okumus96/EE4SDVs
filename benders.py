import gurobipy as gp
from gurobipy import GRB

from optimizer_utils import robust_ms


def routing_lp_cut(net, ends, cables, switches, comm, hw_use, act_expr):
    bus_types = set(net['bus'].get('types', []))
    flows = []
    for t, (u, v) in ends.items():
        if u is None or v is None or u == v or t not in act_expr:
            continue
        elig = list(net['flow_elig'][t])
        if any(w in bus_types for w in elig):
            continue
        if net['volumes'][t] <= 0:
            continue
        flows.append((t, u, v, [w for w in elig if w not in bus_types]))
    if not flows:
        return None
    lp = gp.Model('routing_lp')
    lp.Params.OutputFlag = 0
    lp.Params.Threads = 1
    lp.Params.Method = 1
    lp.Params.Presolve = 0
    f, slack = {}, {}
    link_terms, relay_terms, chain_terms = {}, {}, {}
    chain_of_flow = {}
    for ci, ch in enumerate(net['chains']):
        for t in ch['flows']:
            chain_of_flow.setdefault(t, []).append(ci)
    for t, u, v, types in flows:
        pair = (min(u, v), max(u, v))
        for w in types:
            for p, path in enumerate(net['paths'].get(pair + (w,), [])):
                var = lp.addVar(lb=0.0, name=f"f_{t}_{w}_{p}")
                f[t, w, p] = var
                seq = path if (u, v) == pair else path[::-1]
                lat = 0.0
                for a, b in zip(seq, seq[1:]):
                    link_terms.setdefault((a, b, w), []).append((net['volumes'][t], var))
                    lat += net['kbit'][t] / net['cap_of'][w] + net['hop_ms'][w]
                for r in path[1:-1]:
                    relay_terms.setdefault(r, []).append(var)
                lat = robust_ms(net, lat)
                for ci in chain_of_flow.get(t, []):
                    chain_terms.setdefault(ci, []).append((lat, var))
    demand_c, cap_c, relay_c, chain_c = {}, {}, {}, {}
    for t, u, v, types in flows:
        vs = [var for (tt, w, p), var in f.items() if tt == t]
        if not vs:
            continue
        slack[t] = lp.addVar(lb=0.0, obj=1.0, name=f"s_{t}")
        demand_c[t] = lp.addConstr(gp.quicksum(vs) + slack[t] >= 1.0, name=f"dem_{t}")
    for (a, b, w), terms in link_terms.items():
        key = (min(a, b), max(a, b), w)
        cap = net['cap_of'][w] * float(cables.get(key, 0))
        cap_c[a, b, w] = lp.addConstr(gp.quicksum(vol * var for vol, var in terms) <= cap, name=f"cap_{a}_{b}_{w}")
    for r, vs in relay_terms.items():
        relay_c[r] = [lp.addConstr(var <= (1.0 if r in switches else 0.0), name=f"relay_{r}_{k}") for k, var in enumerate(vs)]
    for ci, terms in chain_terms.items():
        ch = net['chains'][ci]
        chain_c[ci] = lp.addConstr(gp.quicksum(lat * var for lat, var in terms) <= ch['budget'] - ch['const'], name=f"chain_{ci}")
    lp.ModelSense = GRB.MINIMIZE
    lp.optimize()
    if lp.Status != GRB.OPTIMAL or lp.ObjVal <= 1e-6:
        lp.dispose()
        return None
    expr, value = gp.LinExpr(), 0.0
    for t, c in demand_c.items():
        mu = max(0.0, c.Pi)
        if mu > 1e-9:
            expr += mu * act_expr[t]
            value += mu
    for (a, b, w), c in cap_c.items():
        lam = min(0.0, c.Pi)
        if lam < -1e-9:
            key = (min(a, b), max(a, b), w)
            expr += lam * net['cap_of'][w] * comm[key]
            value += lam * net['cap_of'][w] * float(cables.get(key, 0))
    for r, cs in relay_c.items():
        nu = sum(min(0.0, c.Pi) for c in cs)
        if nu < -1e-9:
            sw = hw_use.get((r, 'SWITCH'))
            if sw is not None:
                expr += nu * sw
                value += nu * (1.0 if r in switches else 0.0)
            elif r in switches:
                value += nu
    for ci, c in chain_c.items():
        kap = min(0.0, c.Pi)
        if kap < -1e-9:
            ch = net['chains'][ci]
            expr += kap * (ch['budget'] - ch['const'])
            value += kap * (ch['budget'] - ch['const'])
    lp.dispose()
    if value <= 1e-6:
        return None
    return expr, value


def failure_lp_cut(net, c, ends_all, cables, switches, comm, hw_use, x, r, pos_x, pos_r):
    bus_types = set(net['bus'].get('types', []))
    crit = set(net['critical'])
    bases = {}
    for t, tr in net['flow_meta'].items():
        if tr['type'] != 'SC_SC':
            continue
        i, k = tr['src_idx'], tr['dst_idx']
        if i not in crit or k not in crit or net['volumes'][t] <= 0:
            continue
        if any(w in bus_types for w in net['flow_elig'][t]):
            continue
        bases.setdefault(tr.get('base', t), []).append(t)
    required = []
    for b, variants in bases.items():
        tr = net['flow_meta'][variants[0]]
        i, k = tr['src_idx'], tr['dst_idx']
        if pos_x.get(i) is None or pos_x.get(k) is None or pos_r.get(i) is None or pos_r.get(k) is None:
            continue
        alive, trivial = [], False
        for t in variants:
            u, v = ends_all.get(t, (None, None))
            if u is None or v is None or u == c or v == c:
                continue
            if u == v:
                trivial = True
                break
            alive.append((t, u, v, [w for w in net['flow_elig'][t] if w not in bus_types]))
        if trivial or not alive:
            continue
        required.append((b, alive, [x[i, pos_x[i]], x[k, pos_x[k]], r[i, pos_r[i]], r[k, pos_r[k]]]))
    if not required:
        return None
    lp = gp.Model('failure_lp')
    lp.Params.OutputFlag = 0
    lp.Params.Threads = 1
    lp.Params.Method = 1
    lp.Params.Presolve = 0
    f, slack = {}, {}
    link_terms, relay_terms, chain_terms = {}, {}, {}
    chain_of_base = {}
    for ci, ch in enumerate(net['chains']):
        for t in ch['flows']:
            chain_of_base.setdefault(net['flow_meta'][t].get('base', t), []).append(ci)
    base_vars = {}
    for b, alive, _ind in required:
        for t, u, v, types in alive:
            pair = (min(u, v), max(u, v))
            for w in types:
                for p, path in enumerate(net['paths'].get(pair + (w,), [])):
                    if c in path:
                        continue
                    var = lp.addVar(lb=0.0, name=f"f_{t}_{w}_{p}")
                    f[t, w, p] = var
                    base_vars.setdefault(b, []).append(var)
                    seq = path if (u, v) == pair else path[::-1]
                    lat = 0.0
                    for a, b2 in zip(seq, seq[1:]):
                        link_terms.setdefault((a, b2, w), []).append((net['volumes'][t], var))
                        lat += net['kbit'][t] / net['cap_of'][w] + net['hop_ms'][w]
                    for rr in path[1:-1]:
                        relay_terms.setdefault(rr, []).append(var)
                    lat = robust_ms(net, lat)
                    for ci in chain_of_base.get(b, []):
                        chain_terms.setdefault(ci, []).append((lat, var))
    demand_c, cap_c, relay_c, chain_c = {}, {}, {}, {}
    for b, alive, ind in required:
        slack[b] = lp.addVar(lb=0.0, obj=1.0, name=f"s_{b}")
        demand_c[b] = lp.addConstr(gp.quicksum(base_vars.get(b, [])) + slack[b] >= 1.0, name=f"dem_{b}")
    for (a, b2, w), terms in link_terms.items():
        key = (min(a, b2), max(a, b2), w)
        cap_c[a, b2, w] = lp.addConstr(gp.quicksum(vol * var for vol, var in terms) <= net['cap_of'][w] * float(cables.get(key, 0)), name=f"cap_{a}_{b2}_{w}")
    for rr, vs in relay_terms.items():
        relay_c[rr] = [lp.addConstr(var <= (1.0 if rr in switches else 0.0), name=f"relay_{rr}_{k}") for k, var in enumerate(vs)]
    req_bases = {b for b, _a, _i in required}
    for ci, terms in chain_terms.items():
        ch = net['chains'][ci]
        if all(net['flow_meta'][t].get('base', t) in req_bases or net['flow_meta'][t].get('base', t) in bases for t in ch['flows']):
            chain_c[ci] = lp.addConstr(gp.quicksum(lat * var for lat, var in terms) <= ch['degraded'] - ch['const'], name=f"chain_{ci}")
    lp.ModelSense = GRB.MINIMIZE
    lp.optimize()
    if lp.Status != GRB.OPTIMAL or lp.ObjVal <= 1e-6:
        lp.dispose()
        return None
    expr, value = gp.LinExpr(), 0.0
    ind_of = {b: ind for b, _a, ind in required}
    for b, cst in demand_c.items():
        mu = max(0.0, cst.Pi)
        if mu > 1e-9:
            expr += mu * (gp.quicksum(ind_of[b]) - 3.0)
            value += mu
    for (a, b2, w), cst in cap_c.items():
        lam = min(0.0, cst.Pi)
        if lam < -1e-9:
            key = (min(a, b2), max(a, b2), w)
            expr += lam * net['cap_of'][w] * comm[key]
            value += lam * net['cap_of'][w] * float(cables.get(key, 0))
    for rr, cs in relay_c.items():
        nu = sum(min(0.0, cst.Pi) for cst in cs)
        if nu < -1e-9:
            sw = hw_use.get((rr, 'SWITCH'))
            if sw is not None:
                expr += nu * sw
                value += nu * (1.0 if rr in switches else 0.0)
            elif rr in switches:
                value += nu
    for ci, cst in chain_c.items():
        kap = min(0.0, cst.Pi)
        if kap < -1e-9:
            ch = net['chains'][ci]
            expr += kap * (ch['degraded'] - ch['const'])
            value += kap * (ch['degraded'] - ch['const'])
    lp.dispose()
    if value <= 1e-6:
        return None
    return expr, value

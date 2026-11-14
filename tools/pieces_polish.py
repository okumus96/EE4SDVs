#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from config_reader import ConfigReader
from architecture_loader import load_architecture_from_json
from optimizer import AssignmentOptimizer, UNCERTAINTY_BLOCKS
import pieces


def main():
    ap = argparse.ArgumentParser(description='Continue a run from its saved state: restore or rebuild the design, then polish, merge step, final network '
                                             'clean-up and check, with the settings of the solver.json of the run.')
    ap.add_argument('run_dir', help='output directory of the run to continue')
    ap.add_argument('out_dir', help='directory for the result')
    ap.add_argument('--step', type=int, default=None, help='start from the dump of this build step (default: the latest saved state)')
    a = ap.parse_args()
    run, out = os.path.abspath(a.run_dir), os.path.abspath(a.out_dir)
    os.makedirs(out, exist_ok=True)
    info = json.load(open(os.path.join(run, 'run_info.json')))
    args = info['args']
    arch = os.path.join(run, info.get('architecture_copy') or 'architecture_used.json')
    shutil.copy(arch, os.path.join(out, 'architecture_used.json'))
    values_file, sol_file = os.path.join(run, 'values.json'), os.path.join(run, 'solution_1.json')
    from_values = not a.step and os.path.exists(values_file) and os.path.exists(sol_file)
    if from_values:
        dump_file = sol_file
        sol = json.load(open(sol_file))
        dump = {'placed_x': sol['assignment'], 'placed_r': {k: v['cell'] for k, v in (sol.get('standby') or {}).items()},
                'cost': float(sol['total_cost']), 'pieces': []}
    else:
        steps = sorted(glob.glob(os.path.join(run, 'pieces_step_*.json')), key=lambda f: int(re.search(r'_(\d+)\.json$', f).group(1)))
        polished = os.path.join(run, 'pieces_polish.json')
        dump_file = os.path.join(run, f'pieces_step_{a.step}.json') if a.step else (polished if os.path.exists(polished) else steps[-1])
        dump = json.load(open(dump_file))
    polish_values = None
    vpol_file = os.path.join(run, 'values_polish.json')
    if not from_values and dump_file == os.path.join(run, 'pieces_polish.json') and os.path.exists(vpol_file):
        vp = json.load(open(vpol_file))
        if vp.get('time') == (dump.get('polish') or {}).get('time') and abs(float(vp.get('cost') or 0.0) - float(dump['cost'])) < 1e-6:
            polish_values = vp['values']
        else:
            print(f"[polish] values_polish.json ({vp.get('time')}, {vp.get('cost')}) does not belong to pieces_polish.json "
                  f"({(dump.get('polish') or {}).get('time')}, {dump['cost']}): the design is rebuilt", flush=True)

    cr = ConfigReader(args['config_dir'])
    scs, comm, sens, acts, cables, locs = load_architecture_from_json(arch, cr, num_locs=args.get('num_locs'))
    partitions = cr.get_partitions()
    partitions['hw_capacity'] = cr.get_hardware_capacities()
    unc = {k: bool(args.get('uncertainty') or args.get(k)) for k in UNCERTAINTY_BLOCKS}
    unc['params'] = cr.get_uncertainty_config()
    rho = float(unc['params'].get('rho', {}).get('levels', [1.0])[0])
    unc['rho'] = unc['growth_alpha'] = rho
    if rho <= 0.0:
        unc['failures'] = unc['redundancy'] = False
    unc = pieces.optimizer_utils.with_gradual_standbys(unc, scs, cr.get_chains())
    cfg = {**pieces.DEFAULTS, **(cr.get_solver_config().get('pieces') or {}), 'dump_dir': out,
           'config_dir': args['config_dir'], 'architecture_json': os.path.join(out, 'architecture_used.json')}
    backbone = cr.get_backbone_config()
    if cfg.get('static_capacity', False):
        backbone = {**backbone, 'upfront_paths': True, 'cell_cut_capacity': True}
    gp = args.get('gurobi_params') or cr.get_solver_config().get('gurobi_params')
    common = dict(partitions=partitions, hardwares=cr.get_hardwares(), interfaces=cr.get_interfaces(), uncertainty=unc,
                  backbone=backbone, chains=cr.get_chains())

    print(f"[polish] {run}: design of {os.path.basename(dump_file)} ({len(dump['placed_x'])} components, cost {dump['cost']:.2f}), "
          f"polish {cfg['polish_rounds']} rounds by {cfg.get('polish_mode', 'cell')}, {cfg.get('polish_parallel', 1)} at a time, "
          f"{cfg['polish_time_s']} s", flush=True)
    t0 = time.time()
    opt = AssignmentOptimizer()
    opt.memory = cr.get_solver_config().get('memory') or opt.memory
    if from_values:
        vals = json.load(open(values_file))
        opt.last_values = {k: v for k, v in vals.items() if abs(v - round(v)) < 1e-9}
        start_cost, rebuild_s, gw_conv = float(sol['total_cost']), 0.0, None
        print(f"[polish] {len(vals) - len(opt.last_values)} fractional values of {len(vals)} left out of the start", flush=True)
        print(f"[polish] start from the run's finished design: cost {start_cost:.2f}, {len(opt.last_values)} variable values "
              f"({os.path.basename(values_file)}), {len(sol.get('bus_gateways') or {})} devices with gateways", flush=True)
    elif polish_values is not None:
        sol, start_cost, rebuild_s, gw_conv = _restore(opt, cfg, common, gp, dump, polish_values, scs, locs, sens, acts, cables, comm, t0)
    else:
        sol, start_cost, rebuild_s, gw_conv = _rebuild(opt, cfg, common, backbone, gp, dump, scs, locs, sens, acts, cables, comm, t0)
    t1 = time.time()
    sol, rounds = pieces.polish_fn(cfg)(opt, scs, locs, sens, acts, cables, comm, gurobi_params=gp, cfg=cfg, pieces=dump.get('pieces') or [],
                                        steps=[], sol=sol, traffic={}, **common)
    polish_s = time.time() - t1
    sol, merge, final = pieces.merge_and_final(opt, scs, locs, sens, acts, cables, comm, gurobi_params=gp, cfg=cfg, sol=sol,
                                               pieces=dump.get('pieces') or [], **common)
    pieces._dump_values(cfg, opt)
    _write(out, run, args, cfg, dump_file, sol, rounds, final, gw_conv, start_cost, rebuild_s, polish_s, merge)


def _restore(opt, cfg, common, gp, dump, values, scs, locs, sens, acts, cables, comm, t0):
    start = {k: v for k, v in values.items() if abs(v - round(v)) < 1e-9}
    sol = opt.optimize(scs, locs, sens, acts, cables, comm, time_limit=float(cfg['piece_time_s']), mip_gap=float(cfg['piece_gap']),
                       phase2_time=float(cfg['piece_phase2_s']), phase2_cap=float(cfg['piece_phase2_s']),
                       gurobi_params=','.join(x for x in (gp, cfg.get('step_gurobi_params')) if x), start=start,
                       fixed={'x': dump['placed_x'], 'r': dump['placed_r'], 'vars': values}, lns=None, reuse=True, **common)
    if sol is None:
        sys.exit('[polish] the saved design could not be restored from values_polish.json')
    start_cost, rebuild_s = float(sol['total_cost']), time.time() - t0
    print(f"[polish] design restored from values_polish.json: cost {start_cost:.2f} (saved {float(dump['cost']):.2f}), "
          f"{len(sol.get('bus_gateways') or {})} devices with gateways ({rebuild_s:.0f} s)", flush=True)
    return sol, start_cost, rebuild_s, None


def _rebuild(opt, cfg, common, backbone, gp, dump, scs, locs, sens, acts, cables, comm, t0):
    rebuild = common
    if backbone.get('bus_gateway') and not cfg.get('gateway_in_build', True):
        rebuild = {**common, 'backbone': {**backbone, 'bus_gateway': False}}
        print("[polish] rebuild with the bus gateway off (pieces.gateway_in_build false); the polish with it on", flush=True)
    sol = opt.optimize(scs, locs, sens, acts, cables, comm, time_limit=float(cfg['piece_time_s']), mip_gap=float(cfg['piece_gap']),
                       phase2_time=float(cfg['piece_phase2_s']), phase2_cap=float(cfg['piece_phase2_s']),
                       gurobi_params=','.join(x for x in (gp, cfg.get('step_gurobi_params')) if x),
                       fixed={'x': dump['placed_x'], 'r': dump['placed_r']}, lns=None, reuse=True, **rebuild)
    if sol is None:
        sys.exit('[polish] the dumped design could not be rebuilt')
    start_cost, rebuild_s = float(sol['total_cost']), time.time() - t0
    print(f"[polish] rebuilt design: cost {start_cost:.2f} ({rebuild_s:.0f} s)", flush=True)
    gw_conv = None
    if rebuild is not common:
        sol, gw_conv = pieces._gateway_convert(opt, scs, locs, sens, acts, cables, comm, gurobi_params=gp, cfg=cfg, sol=sol, **common)
    return sol, start_cost, rebuild_s, gw_conv


def _write(out, run, args, cfg, dump_file, sol, rounds, final, gw_conv, start_cost, rebuild_s, polish_s, merge=None):
    sol['solve_info']['pieces'] = {'polish_only': True, 'from': dump_file, 'start_cost': start_cost, 'rebuild_s': round(rebuild_s, 1),
                                   'polish_s': round(polish_s, 1), 'polish': rounds, 'gateway_convert': gw_conv, 'merge': merge or [], 'final_network': final,
                                   'settings': {k: v for k, v in cfg.items() if not k.startswith('_')}}
    lines = [f"polish only: {dump_file}", f"rebuilt design {start_cost:.2f} in {rebuild_s:.0f} s"]
    if gw_conv:
        lines.append(f"gateway step: {gw_conv['before']:.1f} -> {gw_conv['after']:.1f} in {gw_conv['seconds']:.0f} s ({gw_conv['result']}, "
                     f"{gw_conv['gateways']} devices with gateways)")
    lines += [f"polish {r['round']} (pass {r.get('pass', 1)}, batch {r.get('batch', '-')}): {r.get('kind', 'cell')} {r['group']} ({r['free']} free): {r['before']:.1f} -> {r['after']:.1f} in {r['seconds']:.0f} s ({r['result']})" for r in rounds]
    lines += [f"merge round {m.get('round')}: {m.get('close')}" + (f" into {m['target']} first" if m.get('target') else "")
              + f" ({m.get('components', '-')} components): {m.get('before')} -> {m.get('after')} "
              f"({m['result']}{', ' + m['reason'] if m.get('reason') else ''})" for m in (merge or [])]
    if final:
        lines.append(f"final network: {final['before']:.1f} -> {final['after']:.1f} in {final['seconds']:.0f} s ({final['result']}; "
                     f"{final['status']}, gap {100 * (final['gap'] or 0):.3f} %), switches at {final['switches']}")
    lines.append(f"final cost {float(sol['total_cost']):.2f}; polish {polish_s / 60:.0f} min")
    open(os.path.join(out, 'pieces_report.txt'), 'w').write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    json.dump(sol, open(os.path.join(out, 'solution_1.json'), 'w'), indent=2, default=str)
    json.dump({'args': {**args, 'output_dir': out}, 'architecture_copy': 'architecture_used.json', 'polish_of': run,
               'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')}, open(os.path.join(out, 'run_info.json'), 'w'), indent=2)
    subprocess.run([sys.executable, os.path.join(REPO, 'tools', 'check_solution.py'), os.path.join(out, 'solution_1.json')], check=False)


if __name__ == '__main__':
    main()

import json
import os
from models import Interface


class ConfigReader:

    def __init__(self, config_dir='configs'):
        if not os.path.exists(config_dir):
            raise FileNotFoundError(f"Configuration directory not found: {config_dir}")

        self.config_dir = config_dir

        self.sensors_config = self._load_json('sensors.json')
        self.actuators_config = self._load_json('actuators.json')
        self.bus_config = self._load_json('buses.json')
        self.partitions_config = self._load_json('partitions.json')
        self.hardware_config = self._load_json('hardwares.json')
        self.uncertainty_config = self._load_json('uncertainty.json')

    def _load_json(self, filename):
        filepath = os.path.join(self.config_dir, filename)
        with open(filepath, 'r') as f:
            return json.load(f)

    def get_sites(self):
        path = os.path.join(self.config_dir, 'sites.json')
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")
        return self._load_json('sites.json')['sites']

    def get_distance_table(self):
        path = os.path.join(self.config_dir, 'distances.json')
        if not os.path.exists(path):
            return None
        d = self._load_json('distances.json')
        return d['ids'], d['matrix']

    def get_interfaces(self):
        cable_types = {}
        for name, specs in self.bus_config['cable_types'].items():
            if not isinstance(specs, dict):
                continue
            cable_types[name] = Interface(
                name=name,
                cost_per_meter=specs['cost_per_meter'],
                latency_per_meter=float(specs.get('latency_per_meter', 0.0)),
                weight_per_meter=specs['weight_per_meter'],
                capacity=specs.get('capacity', float('inf')),
                port_cost = specs.get('port_cost', 0),
                hop_latency_ms=float(specs.get('hop_latency_ms', 0.0)),
                frame_kbit=float(specs.get('frame_kbit', float('inf'))),
                max_length_m=float(specs.get('max_length_m', float('inf')) or float('inf'))
            )
        return cable_types

    def get_chains(self):
        path = os.path.join(self.config_dir, 'chains.json')
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return list(json.load(f).get('chains', []))

    def get_backbone_config(self):
        return dict(self.bus_config.get('backbone', {}))

    def get_sensors_config(self):
        return self.sensors_config['sensors']

    def get_actuators_config(self):
        return self.actuators_config['actuators']


    def get_partitions(self):
        partition = self.partitions_config.get('partition', {})
        site = self.partitions_config.get('site', {})
        types = [{'id': str(t['id']), 'cpu_cap': float(t.get('cpu_cap') or 0), 'ram_cap': float(t.get('ram_cap') or 0),
                  'rom_cap': float(t.get('rom_cap') or 0), 'tops_cap': float(t.get('tops_cap') or 0), 'cost': float(t.get('cost') or 0)}
                 for t in (partition.get('types') or [])]
        return {
            'types': types or None,
            'tops_feature': partition.get('tops_feature'),
            'cpu_cap': partition.get('cpu_cap', 0),
            'ram_cap': partition.get('ram_cap', 0),
            'rom_cap': partition.get('rom_cap', 0),
            'cost': partition.get('cost', 0),
            'max_per_asil_per_site': partition.get('max_per_asil_per_site', 3),
            'site_max_partitions': site.get('max_partitions'),
            'site_cpu_cap': site.get('cpu_cap'),
            'site_ram_cap': site.get('ram_cap'),
            'site_rom_cap': site.get('rom_cap'),
            'site_fixed_cost': site.get('fixed_cost', 0),
            'switch_node_cost': site.get('switch_node_cost', 0),
        }

    def get_uncertainty_config(self):
        return self.uncertainty_config

    def get_solver_config(self):
        p = os.path.join(self.config_dir, 'solver.json')
        return self._load_json('solver.json') if os.path.exists(p) else {}

    def get_hardware_capacities(self):
        out = {}
        for h, spec in self.hardware_config.get('hardware_costs', {}).items():
            types = [{'id': str(t['id']), 'tops': float(t['tops']), 'cost': float(t.get('cost') or 0)} for t in (spec.get('types') or [])]
            if types:
                out[h] = {'tops': max(t['tops'] for t in types), 'max_units': int(spec.get('max_units_per_site', 1)), 'types': types}
            elif spec.get('tops'):
                out[h] = {'tops': float(spec['tops']), 'max_units': int(spec.get('max_units_per_site', 1))}
            if h in out and spec.get('max_units_total') is not None:
                out[h]['max_units_total'] = int(spec['max_units_total'])
        return out

    def get_hardwares(self):
        costs = {}
        for hw_name, spec in self.hardware_config.get('hardware_costs', {}).items():
            costs[hw_name] = spec.get('cost', 0)
        return costs

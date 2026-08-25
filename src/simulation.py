import numpy as np
from src.particles import ParticleEnsemble
from src.fields import ConstantElectricField, LinearZMagneticField, QuadrupoleMagneticField
from src.integrator import BorisIntegrator, RungeKutta4

class Simulation:
    def __init__(self, config):
        self.ensemble = self._create_ensemble(config['initial_beam'])
        self.electric_fields = self._create_electric_fields(config.get('fields', {}).get('electric_fields', []))
        self.magnetic_fields = self._create_magnetic_fields(config.get('fields', {}).get('magnetic_fields', []))
        self.integrator = self._create_integrator(config['integrator'])
        self.dt = config['time_step']
        self.n_steps = config.get('n_steps', 1000)
        self.save_interval = config.get('save_interval', 10)
        self.results = {'t': [], 'x': [], 'y': [], 'z': [], 'px': [], 'py': [], 'pz': []}

    def _create_ensemble(self, beam_cfg):
        ens = ParticleEnsemble()
        if beam_cfg['type'] == 'cylinder':
            ens.generate_cylinder(
                radius=beam_cfg['radius'],
                length=beam_cfg['length'],
                n_particles=beam_cfg['n_particles'],
                distribution=beam_cfg.get('distribution', 'uniform'),
                energy_eV=beam_cfg.get('energy_eV', 0.0),
                emittance=beam_cfg.get('emittance', None)
            )
        else:
            raise ValueError(f"Unknown beam type: {beam_cfg['type']}")
        return ens

    def _create_electric_fields(self, ef_cfg_list):
        fields = []
        for cfg in ef_cfg_list:
            if cfg['type'] == 'constant':
                fields.append(ConstantElectricField(
                    z_start=cfg['z_start'],
                    z_end=cfg['z_end'],
                    Ex=cfg.get('Ex', 0.0),
                    Ey=cfg.get('Ey', 0.0),
                    Ez=cfg.get('Ez', 0.0)
                ))
            else:
                raise ValueError(f"Unknown electric field type: {cfg['type']}")
        return fields

    def _create_magnetic_fields(self, mf_cfg_list):
        fields = []
        for cfg in mf_cfg_list:
            if cfg['type'] == 'linear_z':
                fields.append(LinearZMagneticField(
                    z_start=cfg['z_start'],
                    z_end=cfg['z_end'],
                    B_start=cfg['B_start'],
                    B_end=cfg['B_end'],
                    direction=cfg.get('direction', 'y')
                ))
            elif cfg['type'] == 'quadrupole':
                fields.append(QuadrupoleMagneticField(
                    z_start=cfg['z_start'],
                    z_end=cfg['z_end'],
                    gradient=cfg['gradient']
                ))
            else:
                raise ValueError(f"Unknown magnetic field type: {cfg['type']}")
        return fields

    def _create_integrator(self, int_cfg):
        method = int_cfg.get('method', 'boris')
        if method == 'boris':
            return BorisIntegrator()
        elif method == 'rk4':
            return RungeKutta4()
        else:
            raise ValueError(f"Unknown integrator: {method}")

    def _field_at(self, x, y, z, t):
        E = np.zeros(3)
        B = np.zeros(3)
        for ef in self.electric_fields:
            E += ef.E(x, y, z, t)
        for mf in self.magnetic_fields:
            B += mf.B(x, y, z, t)
        return E, B

    def _save_state(self, t):
        self.results['t'].append(t)
        self.results['x'].append(self.ensemble.x.copy())
        self.results['y'].append(self.ensemble.y.copy())
        self.results['z'].append(self.ensemble.z.copy())
        self.results['px'].append(self.ensemble.px.copy())
        self.results['py'].append(self.ensemble.py.copy())
        self.results['pz'].append(self.ensemble.pz.copy())

    def run(self):
        self._save_state(0.0)
        t = 0.0
        for step in range(self.n_steps):
            self.integrator.step(self.ensemble, self._field_at, self.dt)
            t += self.dt
            if step % self.save_interval == 0:
                self._save_state(t)
        # Сохранить конечное состояние
        self._save_state(t)

    # Методы для сохранения/загрузки (опционально)
    def save_results(self, filename):
        np.savez_compressed(filename, **self.results)

    def load_results(self, filename):
        data = np.load(filename)
        self.results = {key: data[key] for key in data.files}
import numpy as np
from particles import ParticleEnsemble
from integrator import RK4Integrator

class Simulation:
    def __init__(self, config, electric_field, magnetic_fields):
        self.ensemble = ParticleEnsemble()
        self.electric_field = electric_field
        self.magnetic_fields = magnetic_fields
        self.integrator = RK4Integrator()
        self.dt = config['time_step']
        self.n_steps = config['n_steps']
        self.save_interval = config['save_interval']
        self.results = {'t': [], 'x': [], 'y': [], 'z': [], 'px': [], 'py': [], 'pz': []}

    def _save_state(self, t):
        self.results['t'].append(t)
        self.results['x'].append(self.ensemble.x.copy())
        self.results['y'].append(self.ensemble.y.copy())
        self.results['z'].append(self.ensemble.z.copy())
        self.results['px'].append(self.ensemble.px.copy())
        self.results['py'].append(self.ensemble.py.copy())
        self.results['pz'].append(self.ensemble.pz.copy())

    def run(self):
        self.ensemble.generate_start_multitude([1, 1, 1, 1, 1, 1], 1000)
        self._save_state(0.0)
        t = 0.0
        for step in range(self.n_steps):
            self.integrator.step(self.ensemble, self.electric_field, self.magnetic_fields, self.dt)
            t += self.dt
            if step % self.save_interval == 0:
                print(f'Прошло итераций: {step}')
                self._save_state(t)
        # Сохранить конечное состояние
        self._save_state(t)

    # Методы для сохранения/загрузки (опционально)
    def save_results(self, filename):
        np.savez_compressed(filename, **self.results)

    def load_results(self, filename):
        data = np.load(filename)
        self.results = {key: data[key] for key in data.files}
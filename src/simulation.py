from dataclasses import dataclass
import numpy as np
from particles import ParticleEnsemble, PhaseSpaceDefinition
from integrator import RK4Integrator

@dataclass
class SimulationResult:
    t: np.ndarray
    states: np.ndarray
    phase_space: PhaseSpaceDefinition

    def save_results(self, filename, result):
        np.savez_compressed(
            filename,
            t=result.t,
            states=result.states,
            variables=np.array(result.phase_space.names),
        )

class Simulation:
    def __init__(self, config, electric_field, magnetic_fields, phase_space: PhaseSpaceDefinition):
        self.ensemble = ParticleEnsemble()
        self.electric_field = electric_field
        self.magnetic_fields = magnetic_fields
        self.integrator = RK4Integrator()
        self.dt = config['time_step']
        self.n_steps = config['n_steps']
        self.save_interval = config['save_interval']
        self.results = {'states': [], 't': []}

    def _save_state(self, t):
        self.results['t'].append(t)
        self.results['states'].append(self.ensemble.state)

    def run(self):
        self._save_state(0.0)
        t = 0.0
        for step in range(self.n_steps):
            self.integrator.step(self.ensemble, self.electric_field, self.magnetic_fields, self.dt, t)
            t += self.dt
            if step % self.save_interval == 0:
                print(f'Прошло итераций: {step}')
                self._save_state(t)
        # Сохранить конечное состояние
        self._save_state(t)

        return SimulationResult(t=self.results['t'], states=self.results['states'], phase_space=self.phase_space)

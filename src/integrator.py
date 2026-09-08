import numpy as np
from particles import ParticleEnsemble
from fields import MagneticField, ElectricField

class RK4Integrator:
    """
    Интегратор Рунге–Кутты 4-го порядка для ансамбля релятивистских заряженных частиц.
    """
    def __init__(self,
                derivatives_foo,
                derivatives_params,
                c=2.99792458e8):
        self.c = c  # скорость света, м/с
        self.derivatives = derivatives_foo

    def step(self, ensemble: ParticleEnsemble, fieldE: ElectricField, fieldB: MagneticField, dt, t) -> None:
        """
        Один шаг RK4 для всех частиц в ансамбле.
        ensemble – объект ParticleEnsemble с массивами координат и импульсов.
        """
        n = ensemble.n_particles
        # Для каждой частицы выполняем шаг RK4
        k1 = self.derivatives(ensemble.state, ensemble.q, ensemble.m, fieldE, fieldB, t)
        k2 = self.derivatives(ensemble.state + 0.5 * dt * k1, ensemble.q, ensemble.m, fieldE, fieldB, t + 0.5 * dt)
        k3 = self.derivatives(ensemble.state + 0.5 * dt * k2, ensemble.q, ensemble.m, fieldE, fieldB, t + 0.5 * dt)
        k4 = self.derivatives(ensemble.state + dt * k3, ensemble.q, ensemble.m, fieldE, fieldB, t + dt)

        ensemble.state = ensemble.state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

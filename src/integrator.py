import numpy as np
from particles import ParticleEnsemble
from fields import MagneticField, ElectricField

class RK4Integrator:
    """
    Интегратор Рунге–Кутты 4-го порядка для ансамбля релятивистских заряженных частиц.
    """
    def __init__(self, c=2.99792458e8):
        self.c = c  # скорость света, м/с

    def derivatives(self, state, q, m, fieldE: ElectricField, fieldB: MagneticField, t=0) -> np.ndarray:
        """
        Вычисляет производные для одной частицы.
        state = [x, y, z, px, py, pz]
        q, m – заряд и масса частицы
        field_func – функция, возвращающая (E, B) для заданной точки
        """
        x, y, z, px, py, pz = state
        E, B = fieldE.E(x, y, z, t), fieldB.B(x, y, z, t)  # время пока считаем = 0 (или добавить t)

        # Релятивистский фактор
        p2 = px*px + py*py + pz*pz
        gamma = np.sqrt(1 + p2 / (m * self.c)**2)
        # Скорость
        vx = px / (gamma * m)
        vy = py / (gamma * m)
        vz = pz / (gamma * m)

        # Сила Лоренца (в СИ)
        Fx = q * (E[0] + vy * B[2] - vz * B[1])
        Fy = q * (E[1] + vz * B[0] - vx * B[2])
        Fz = q * (E[2] + vx * B[1] - vy * B[0])

        return np.array([vx, vy, vz, Fx, Fy, Fz])

    def step(self, ensemble: ParticleEnsemble, fieldE: MagneticField, fieldB: MagneticField, dt) -> None:
        """
        Один шаг RK4 для всех частиц в ансамбле.
        ensemble – объект ParticleEnsemble с массивами координат и импульсов.
        """
        n = ensemble.n_particles
        # Для каждой частицы выполняем шаг RK4
        for i in range(n):
            state = np.array([
                ensemble.x[i], ensemble.y[i], ensemble.z[i],
                ensemble.px[i], ensemble.py[i], ensemble.pz[i]
            ])
            q = ensemble.q[i]
            m = ensemble.m[i]

            # Классический RK4
            k1 = self.derivatives(state, q, m, fieldE, fieldB)
            k2 = self.derivatives(state + 0.5 * dt * k1, q, m, fieldE, fieldB)
            k3 = self.derivatives(state + 0.5 * dt * k2, q, m, fieldE, fieldB)
            k4 = self.derivatives(state + dt * k3, q, m, fieldE, fieldB)

            new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            # Обновляем ансамбль
            ensemble.x[i], ensemble.y[i], ensemble.z[i] = new_state[0], new_state[1], new_state[2]
            ensemble.px[i], ensemble.py[i], ensemble.pz[i] = new_state[3], new_state[4], new_state[5]
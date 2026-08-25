from dataclasses import dataclass
import numpy as np

@dataclass
class Particle:
    x: float  # координаты [м]
    y: float
    z: float
    px: float  # импульсы [кг*м/с]
    py: float
    pz: float
    q: float   # заряд [Кл]
    m: float   # масса [кг]

class ParticleEnsemble:
    """Набор частиц (пучок). Хранит массивы координат и импульсов."""
    def __init__(self):
        self.x = np.array([])   # N частиц
        self.y = np.array([])
        self.z = np.array([])
        self.px = np.array([])
        self.py = np.array([])
        self.pz = np.array([])
        self.q = np.array([])
        self.m = np.array([])
        self.n_particles = 0

    def generate_start_multitude(self, semi_axes, n_particles, center=None,  charge=1.602e-19, mass=9.109e-31):
        semi_axes = np.array(semi_axes, dtype=float)
        if semi_axes.shape != (6,):
            raise ValueError("semi_axes должен быть списком из 6 элементов")
    
        if center is None:
            center = np.zeros(6)
        else:
            center = np.array(center, dtype=float)
            if center.shape != (6,):
                raise ValueError("center должен быть списком из 6 элементов")
            
        self.x = np.empty(0)
        self.y = np.empty(0)
        self.z = np.empty(0)
        self.px = np.empty(0)
        self.py = np.empty(0)
        self.pz = np.empty(0)
        self.q = np.empty(0)
        self.m = np.empty(0)

        accepted = 0
        max_attempts = n_particles * 1000  # ограничение на число попыток
        attempts = 0

        while accepted < n_particles and attempts < max_attempts:
            # Случайная точка в 6-мерном параллелепипеде
            candidate = np.random.uniform(-semi_axes, semi_axes)
            # Проверка попадания внутрь эллипсоида
            norm = np.sum((candidate / semi_axes) ** 2)
            if norm <= 1.0:
                point = candidate + center
                self.x = np.append(self.x, point[0])
                self.px = np.append(self.px, point[1])
                self.y = np.append(self.y, point[2])
                self.py = np.append(self.py, point[3])
                self.z = np.append(self.z, point[4])
                self.pz = np.append(self.pz, point[5])
                self.q = np.append(self.q, charge)
                self.m = np.append(self.m, mass)
                accepted += 1
            attempts += 1

        if accepted < n_particles:
            print(f"Предупреждение: принято только {accepted} частиц из {n_particles} "
                  f"после {attempts} попыток. Увеличьте max_attempts или измените параметры.")
            self.n_particles = accepted
        else:
            self.n_particles = accepted
from dataclasses import dataclass
import numpy as np

@dataclass
class PhaseSpaceDefinition:
    names: dict[str, int]

    def get(self, name: str) -> int:
        if name not in self.names:
            raise Exception('Нет такого параметра в фазовом просранстве')
        return self.names[name]


class ParticleEnsemble:
    """Набор частиц (пучок). Хранит массивы координат и импульсов."""
    #TODO прописать способ задать начальное распределение(пока внутри эллипса)
    def __init__(
            self, 
            n_particels: int,
            q: int, 
            m: int,
            start_multitude_gen_foo,
            start_multitude_gen_foo_metadata: dict):

        self.q = q
        self.m = m     
        self.generate_start_multitude = start_multitude_gen_foo
        self.start_multitude_params = start_multitude_gen_foo_metadata.copy()
        self.n_particles = n_particels
        self.state = self.generate_start_multitude(n=self.n_particles, **self.start_multitude_params)

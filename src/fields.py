from abc import ABC, abstractmethod
import numpy as np

class ElectricField(ABC):
    """Базовый класс для электрического поля."""
    @abstractmethod
    def E(self, x, y, z, t) -> np.ndarray:
        """Возвращает вектор напряжённости электрического поля (В/м)."""
        pass

class MagneticField(ABC):
    """Базовый класс для магнитного поля."""
    @abstractmethod
    def B(self, x, y, z, t) -> np.ndarray:
        """Возвращает вектор магнитной индукции (Тл)."""
        pass

class ConstantElectricField(ElectricField):
    """
    Электрическое поле, постоянное внутри интервала [z_start, z_end]
    и равное нулю вне его.
    """
    def __init__(self, z_start, z_end, Ex=0.0, Ey=0.0, Ez=0.0):
        self.z_start = z_start
        self.z_end = z_end
        self.E_vec = np.array([Ex, Ey, Ez])

    def E(self, x, y, z, t):
        if self.z_start <= z <= self.z_end:
            return self.E_vec
        else:
            return np.zeros(3)
        
class LinearZMagneticField(MagneticField):
    """
    Магнитное поле, линейно изменяющееся вдоль оси z внутри интервала
    [z_start, z_end]. Вне интервала равно нулю.
    Вектор B = (0, 0, Bz) или (0, By, 0) и т.д. по выбору направления.
    """
    def __init__(self, z_start, z_end, B_start, B_end, direction='y'):
        self.z_start = z_start
        self.z_end = z_end
        self.B_start = B_start
        self.B_end = B_end
        self.direction = direction  # 'x', 'y', 'z'

    def B(self, x, y, z, t):
        if self.z_start <= z <= self.z_end:
            # линейная интерполяция
            local_z = z - self.z_start
            length = self.z_end - self.z_start
            B_magnitude = self.B_start + (self.B_end - self.B_start) * (local_z / length)
            B = np.zeros(3)
            if self.direction == 'x':
                B[0] = B_magnitude
            elif self.direction == 'y':
                B[1] = B_magnitude
            elif self.direction == 'z':
                B[2] = B_magnitude
            return B
        else:
            return np.zeros(3)
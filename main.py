from src.particles import ParticleEnsemble
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    ensemble = ParticleEnsemble()
    ensemble.generate_start_multitude(semi_axes=[0.01, 1e-21, 0.01, 1e-21, 0.02, 1e-22],
                                      n_particles=1000)
    
# Доступ к данным
x = ensemble.x
y = ensemble.y
z = ensemble.z
px = ensemble.px
py = ensemble.py
pz = ensemble.pz

# Создаём набор графиков
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Проекции координат
axes[0,0].scatter(x, y, s=2)
axes[0,0].set_xlabel('x, м')
axes[0,0].set_ylabel('y, м')
axes[0,0].set_title('Поперечное сечение (x-y)')

axes[0,1].scatter(x, z, s=2)
axes[0,1].set_xlabel('x, м')
axes[0,1].set_ylabel('z, м')
axes[0,1].set_title('Продольная проекция (x-z)')

axes[0,2].scatter(y, z, s=2)
axes[0,2].set_xlabel('y, м')
axes[0,2].set_ylabel('z, м')
axes[0,2].set_title('Продольная проекция (y-z)')

# Фазовые проекции
axes[1,0].scatter(x, px, s=2)
axes[1,0].set_xlabel('x, м')
axes[1,0].set_ylabel('px, кг·м/с')
axes[1,0].set_title('Фазовая плоскость (x-px)')

axes[1,1].scatter(y, py, s=2)
axes[1,1].set_xlabel('y, м')
axes[1,1].set_ylabel('py, кг·м/с')
axes[1,1].set_title('Фазовая плоскость (y-py)')

axes[1,2].scatter(z, pz, s=2)
axes[1,2].set_xlabel('z, м')
axes[1,2].set_ylabel('pz, кг·м/с')
axes[1,2].set_title('Фазовая плоскость (z-pz)')

plt.tight_layout()
plt.show()
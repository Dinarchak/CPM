import matplotlib.pyplot as plt
import numpy as np
# self.results['t'].append(t)
#         self.results['x'].append(self.ensemble.x.copy())
#         self.results['y'].append(self.ensemble.y.copy())
#         self.results['z'].append(self.ensemble.z.copy())
#         self.results['px'].append(self.ensemble.px.copy())
#         self.results['py'].append(self.ensemble.py.copy())
#         self.results['pz'].append(self.ensemble.pz.copy())

def plot_trajectories(sim_results, axis='xz', show=True, save_path=None):
    """График траекторий в выбранной плоскости."""
    plt.plot(sim_results['x'], sim_results['y'])
    if show:
        plt.show()

def plot_beam_cross_section(sim_results, z_position=None, show=True):
    """Поперечное распределение частиц в заданной точке z."""
        # Если z_position не задана — берём выход пучка
    if z_position is None:
        index = -1
    else:
        # Для каждого сохранённого момента берём среднее z частиц
        z_mean = np.array([
            np.mean(z)
            for z in sim_results['z']
        ])

        # Находим ближайший сохранённый срез
        index = np.argmin(np.abs(z_mean - z_position))

    x = np.asarray(sim_results['x'][index])
    y = np.asarray(sim_results['y'][index])

    plt.figure()
    plt.scatter(x, y, s=5)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Поперечное сечение пучка')
    plt.axis('equal')
    plt.grid(True)

    if show:
        plt.show()
    

def plot_energy_spectrum(sim_results, show=True):
    px = np.asarray(sim_results['px'][-1])
    py = np.asarray(sim_results['py'][-1])
    pz = np.asarray(sim_results['pz'][-1])

    # Квадрат импульса каждой частицы
    p_squared = px**2 + py**2 + pz**2

    # Если используется нерелятивистская механика:
    energies = p_squared / 2

    plt.figure()
    plt.hist(energies, bins=50)

    plt.xlabel('Energy')
    plt.ylabel('Number of particles')
    plt.title('Энергетический спектр на выходе')
    plt.grid(True)

    if show:
        plt.show()
    
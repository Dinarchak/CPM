from simulation import Simulation
from fields import ConstantElectricField, LinearZMagneticField
from simulation import *

if __name__ == "__main__":
    sim = Simulation(
        config={'time_step':1e-12, 'save_interval':100, 'n_steps': 1000},
        electric_field=ConstantElectricField(z_start=0, z_end=1),
        magnetic_fields=LinearZMagneticField(z_start=1, z_end=2, B_start=12, B_end=50, direction='z')
        )
    sim.run()
    sim.save_results('save_res.txt')
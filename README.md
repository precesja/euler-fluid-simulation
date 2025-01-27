# Kelvin-Helmholtz Instability Simulation

This project implements a numerical simulation of the Kelvin-Helmholtz instability using the 2D Euler equations for fluid dynamics. The simulation is solved on a rectangular grid using the finite difference method and visualized in real-time. The goal is to demonstrate the behavior of shear instabilities that arise when two layers of fluid move at different velocities.

---

## Features

- Numerical solution of the **2D Euler equations** (mass, momentum, and energy conservation).
- Finite difference discretization with flux computations using the **local Lax-Friedrichs (Rusanov) scheme**.
- Real-time visualization of density fields.
- Configurable parameters for grid resolution, timestep, and physical properties of the fluid.

---

## Requirements

### Python Dependencies:
The project requires Python 3.8+ and the following Python libraries:

- `numpy`: For numerical computations.
- `matplotlib`: For visualizing the simulation.

To install all dependencies, run:

```bash
pip install numpy matplotlib
```

---

## How It Works

1. **Grid Setup:**
   - The simulation uses a 2D grid with uniform cells. Each cell contains variables like density, velocity, and pressure.
   
2. **Initial Conditions:**
   - Opposite-moving streams are initialized with a sinusoidal perturbation to trigger the Kelvin-Helmholtz instability.

3. **Time Integration:**
   - The simulation advances in time using a time-stepping scheme based on the CFL condition to ensure stability.

4. **Flux Computation:**
   - Fluxes are computed at the cell faces using the local Lax-Friedrichs method, with additional diffusive terms to stabilize the simulation.

---

## Usage

### Running the Simulation:
Run the script using Python:

```bash
python kelvin_helmholtz_simulation.py
```

### Configurable Parameters:
You can modify the following parameters directly in the script:

- **`N`**: Number of grid cells (resolution).
- **`boxsize`**: Size of the simulation domain.
- **`gamma`**: Specific heat ratio of the gas.
- **`cfl`**: CFL condition multiplier for timestep computation.

### Output:

- Real-time visualization of the density field.

---

## References

- Toro, E. F. (1999). *Riemann Solvers and Numerical Methods for Fluid Dynamics.* Springer.
- LeVeque, R. J. (2002). *Finite Volume Methods for Hyperbolic Problems.* Cambridge University Press.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For any questions or feedback, feel free to open an issue or contact me at **robert.niedziela.96@gmail.com**.

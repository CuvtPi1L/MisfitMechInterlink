import csv
import numpy as np

# Constants for the simulation
GRAVITY = np.array([0, -9.81])  # Gravity acting downwards
DAMPING = 0.98  # Damping to avoid endless oscillation
TIME_STEP = 0.01  # Time step for the simulation

class Mass:
    def __init__(self, position, mass, is_fixed=False):
        """
        Initialize a mass point.
        :param position: Initial position of the mass as a 2D numpy array.
        :param mass: The mass value.
        :param is_fixed: Whether the mass is fixed (immovable).
        """
        self.position = np.array(position, dtype=float)
        self.mass = mass
        self.is_fixed = is_fixed
        self.velocity = np.zeros(2)  # Initially at rest
        self.force = np.zeros(2)     # No forces initially

    def apply_force(self, force):
        """
        Apply a force to the mass point.
        :param force: A 2D numpy array representing the force vector.
        """
        if not self.is_fixed:
            self.force += force

    def integrate(self, dt):
        """
        Update the mass position and velocity based on applied forces.
        :param dt: The time step for the integration.
        """
        if not self.is_fixed:
            # Update acceleration from F = ma
            acceleration = self.force / self.mass
            # Update velocity
            self.velocity += acceleration * dt
            self.velocity *= DAMPING  # Apply some damping
            # Update position
            self.position += self.velocity * dt
            # Reset the force accumulator
            self.force = np.zeros(2)

class Spring:
    def __init__(self, mass1, mass2, rest_length, stiffness):
        """
        Initialize a spring connecting two masses.
        :param mass1: The first mass connected by the spring.
        :param mass2: The second mass connected by the spring.
        :param rest_length: The rest length of the spring.
        :param stiffness: The spring's stiffness (Hooke's Law constant).
        """
        self.mass1 = mass1
        self.mass2 = mass2
        self.rest_length = rest_length
        self.stiffness = stiffness

    def apply_spring_force(self):
        """
        Apply spring forces to the masses according to Hooke's Law.
        """
        # Vector from mass1 to mass2
        displacement = self.mass2.position - self.mass1.position
        distance = np.linalg.norm(displacement)
        if distance == 0:
            return  # Prevent division by zero

        # Hooke's law: F = -k * (x - rest_length)
        force_magnitude = self.stiffness * (distance - self.rest_length)
        force_direction = displacement / distance  # Normalized direction
        force = force_magnitude * force_direction

        # Apply equal and opposite forces to the two masses
        self.mass1.apply_force(force)
        self.mass2.apply_force(-force)

class SoftBody:
    def __init__(self, masses, springs):
        """
        Initialize a soft body with masses and springs.
        :param masses: A list of Mass objects.
        :param springs: A list of Spring objects connecting the masses.
        """
        self.masses = masses
        self.springs = springs

    def apply_gravity(self):
        """
        Apply gravity to all masses in the soft body.
        """
        for mass in self.masses:
            if not mass.is_fixed:
                mass.apply_force(mass.mass * GRAVITY)

    def apply_spring_forces(self):
        """
        Apply all spring forces in the soft body.
        """
        for spring in self.springs:
            spring.apply_spring_force()

    def integrate(self, dt):
        """
        Integrate (update) the positions and velocities of the masses.
        :param dt: The time step for the integration.
        """
        for mass in self.masses:
            mass.integrate(dt)

class Simulation:
    def __init__(self, soft_body, csv_filename):
        """
        Initialize the simulation with a soft body and output CSV file.
        :param soft_body: The SoftBody object to simulate.
        :param csv_filename: The CSV file to write the output to.
        """
        self.soft_body = soft_body
        self.csv_filename = csv_filename

    def step(self):
        """
        Perform a single step of the simulation.
        """
        # Apply gravity to all masses
        self.soft_body.apply_gravity()
        # Apply spring forces
        self.soft_body.apply_spring_forces()
        # Integrate positions and velocities
        self.soft_body.integrate(TIME_STEP)

    def run(self, steps):
        """
        Run the simulation for a given number of steps and log positions to CSV.
        :param steps: Number of simulation steps to run.
        """
        with open(self.csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header row
            header = []
            for i, mass in enumerate(self.soft_body.masses):
                header.append(f'Mass_{i}_X')
                header.append(f'Mass_{i}_Y')
            writer.writerow(header)

            # Simulation loop
            for _ in range(steps):
                # Record the positions of all masses
                row = []
                for mass in self.soft_body.masses:
                    row.append(mass.position[0])  # X coordinate
                    row.append(mass.position[1])  # Y coordinate
                writer.writerow(row)

                # Perform a simulation step
                self.step()

# Example usage
def create_soft_body_grid(rows, cols, mass_value, stiffness, spacing):
    """
    Create a grid-shaped soft body with rows and columns of masses connected by springs.
    :param rows: Number of rows of masses.
    :param cols: Number of columns of masses.
    :param mass_value: The mass of each mass point.
    :param stiffness: Stiffness of the springs.
    :param spacing: Spacing between adjacent mass points.
    :return: A SoftBody object.
    """
    masses = []
    springs = []
    
    # Create masses in a grid
    for i in range(rows):
        for j in range(cols):
            # Set the position of the mass
            position = [j * spacing, i * spacing]
            # Create a fixed mass at the top of the grid
            is_fixed = (i == rows - 1)  # Fix the top row
            mass = Mass(position, mass_value, is_fixed)
            masses.append(mass)

    # Connect masses with springs
    for i in range(rows):
        for j in range(cols):
            index = i * cols + j
            # Horizontal springs
            if j < cols - 1:
                right_index = index + 1
                springs.append(Spring(masses[index], masses[right_index], spacing, stiffness))
            # Vertical springs
            if i < rows - 1:
                below_index = index + cols
                springs.append(Spring(masses[index], masses[below_index], spacing, stiffness))

    return SoftBody(masses, springs)

# Driver Code
if __name__ == "__main__":
    # Create a soft body grid with 4 rows and 4 columns of masses
    soft_body = create_soft_body_grid(4, 4, mass_value=1.0, stiffness=100.0, spacing=1.0)
    
    # Initialize the simulation with output CSV filename
    simulation = Simulation(soft_body, 'soft_body_simulation.csv')

    # Run the simulation for 1000 steps and log positions to CSV
    simulation.run(1000)

    print("Simulation complete. Positions recorded in 'soft_body_simulation.csv'.")

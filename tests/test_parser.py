import pytest
import numpy as np
from trajpy.auxiliar_functions import parse_lammps_dump_yaml
import os


class TestParseLammpsDumpYaml:
    """Test suite for parse_lammps_dump_yaml function"""

    def test_basic_parsing(self):
        """Test that the parser correctly reads the sample YAML file"""
        sample_file = 'data/samples/sample_lammps.yml'
        positions = parse_lammps_dump_yaml(sample_file)

        # Check shape: (num_time_steps, num_atoms, 4)
        assert positions.shape == (4, 3, 4), f"Expected shape (4, 3, 4), got {positions.shape}"

    def test_time_values(self):
        """Test that time values are correctly extracted"""
        sample_file = 'data/samples/sample_lammps.yml'
        positions = parse_lammps_dump_yaml(sample_file)

        expected_times = [0.0, 0.5, 1.0, 1.5]
        for time_step, expected_time in enumerate(expected_times):
            actual_time = positions[time_step, 0, 0]
            assert actual_time == expected_time, f"Time at step {time_step} should be {expected_time}, got {actual_time}"

    def test_position_values(self):
        """Test that x, y, z coordinates are correctly extracted"""
        sample_file = 'data/samples/sample_lammps.yml'
        positions = parse_lammps_dump_yaml(sample_file)

        # Check first atom at first timestep
        assert positions[0, 0, 1] == 0.0, "Atom 1 x-coordinate should be 0.0"
        assert positions[0, 0, 2] == 0.0, "Atom 1 y-coordinate should be 0.0"
        assert positions[0, 0, 3] == 0.0, "Atom 1 z-coordinate should be 0.0"

        # Check second atom at first timestep
        assert np.isclose(positions[0, 1, 1], 8.397981e-01), "Atom 2 x-coordinate mismatch"
        assert np.isclose(positions[0, 1, 2], 8.397981e-01), "Atom 2 y-coordinate mismatch"
        assert positions[0, 1, 3] == 0.0, "Atom 2 z-coordinate should be 0.0"

    def test_consistency_across_timesteps(self):
        """Test that atom positions remain consistent across timesteps (stationary atoms in sample)"""
        sample_file = 'data/samples/sample_lammps.yml'
        positions = parse_lammps_dump_yaml(sample_file)

        # In the sample file, positions are identical across timesteps
        for time_step in range(1, 4):
            for atom in range(3):
                for coord in range(1, 4):  # x, y, z
                    assert positions[time_step, atom, coord] == positions[0, atom, coord], \
                        f"Position mismatch at timestep {time_step}, atom {atom}, coord {coord}"

    def test_return_type(self):
        """Test that the function returns a numpy array"""
        sample_file = 'data/samples/sample_lammps.yml'
        positions = parse_lammps_dump_yaml(sample_file)

        assert isinstance(positions, np.ndarray), "Return type should be numpy.ndarray"
        assert positions.dtype == np.float64, f"Array dtype should be float64, got {positions.dtype}"

    def test_file_not_found(self):
        """Test that appropriate error is raised for non-existent file"""
        with pytest.raises(FileNotFoundError):
            parse_lammps_dump_yaml('nonexistent_file.yml')

    def test_all_atoms_present(self):
        """Test that all atoms are present at each timestep"""
        sample_file = 'data/samples/sample_lammps.yml'
        positions = parse_lammps_dump_yaml(sample_file)

        num_time_steps = positions.shape[0]
        num_atoms = positions.shape[1]

        assert num_atoms == 3, f"Expected 3 atoms, got {num_atoms}"
        assert num_time_steps == 4, f"Expected 4 timesteps, got {num_time_steps}"
import numpy as np
from typing import Tuple
from rl.environment.dynamic_programming.jacks_car_rental import JacksCarRental
from rl.utils.general import set_filepath
import os
from rl.algorithms.dynamic_programming.viz import plot_policy_and_value


class PolicyIteration:
    def __init__(self, env: JacksCarRental, gamma: float = 0.9, theta: float = 1e-8) -> None:
        """
        Initialise the PolicyIteration algorithm with the given environment.

        Args:
            env (JacksCarRental): The environment to solve.
            gamma (float): The discount factor.
            theta (float): A small threshold for determining convergence in policy evaluation.
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta

        # TODO: might need to return to this if env refactored following Gymnasium API
        self.max_cars: int = env.max_cars
        self.policy: np.ndarray = np.zeros((self.max_cars + 1, self.max_cars + 1), dtype=np.int8)  # int8 fits action range    # noqa
        self.value: np.ndarray = np.zeros((self.max_cars + 1, self.max_cars + 1), dtype=np.float32)

    def save_artefacts(self, save_name: str) -> None:
        """
        Save the current policy and value function to disk.

        Args:
            save_name (str): The name to use when saving the policy and value arrays.
        """
        policy_dir = "./.data/dynamic_programming/policy_iteration/policy"
        value_dir = "./.data/dynamic_programming/policy_iteration/value"

        # Ensure directories exist
        os.makedirs(policy_dir, exist_ok=True)
        os.makedirs(value_dir, exist_ok=True)

        policy_filepath = policy_dir + "/" + save_name + ".npy"
        value_filepath = value_dir + "/" + save_name + ".npy"
        np.save(set_filepath(policy_filepath), self.policy)
        np.save(set_filepath(value_filepath), self.value)

    def _update_expected_return_array(self) -> None:
        """
        Helper function for calculating expected returns efficiently.

        This updates the `gamma P^{(1)T} V(s') P^{(2)}` matrix, for all states s' in S.

        See lecture notes for further details.
        """
        self.expected_value_matrix = self.env.get_expected_value(self.value, self.gamma)

    def _get_expected_return(self, state_1_next_morning: int, state_2_next_morning: int, action: int) -> float:
        """
        Calculates the expected return for a given state and action efficiently, using stored matrices.

        This evaluates:

            sum_{r,s'} p(s', r | s, a) (r + gamma v(s')) =
                EXPECTED_VALUE_MATRIX(s_1^dagger, s_2^dagger) + R_a |a|

        Args:
            state_1_next_morning (int): Number of cars at location 1 the next morning.
            state_2_next_morning (int): Number of cars at location 2 the next morning.
            action (int): The action taken (number of cars moved from location 1 to 2).

        Returns:
            float: The expected return for the given state and action.
        """
        expected_value_next_state = self.expected_value_matrix[state_1_next_morning, state_2_next_morning]
        expected_return = expected_value_next_state - self.env.move_cost * np.abs(action)
        return expected_return

    def policy_evaluation(self)-> None:
        """
        Perform policy evaluation to update the value function using the current policy.
        """
        pass  # TODO: Implement this function


    def policy_improvement(self)-> bool:
        """
        Perform policy improvement to update the policy based on the current value function.

        Returns:
            bool: True if the policy is stable (no changes), False otherwise.
        """
        pass  # TODO: Implement this function


    def policy_iteration(self)-> Tuple[np.ndarray, np.ndarray]:
        """
        Run the policy iteration algorithm.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The optimal policy and value function.
        """
        pass  # TODO: Implement this function


if __name__ == "__main__":
    env = JacksCarRental()
    policy_iteration = PolicyIteration(env)
    policy, value = policy_iteration.policy_iteration()

    # Plot the policy and value
    plot_policy_and_value(policy, value)

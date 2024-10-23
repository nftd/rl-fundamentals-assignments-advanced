"""
"""
# TODO: figure out strange policy plots


from rl.algorithms.monte_carlo.viz import plot_results
from rl.algorithms.common.mc_agent import MonteCarloAgent
# N.B., numpy argmax used in this instance to ensure ties are broken consistently

from rl.common.policy import EpsilonGreedyPolicy, DeterministicPolicy

import gymnasium as gym

from typing import Union, Tuple, List
from gymnasium import Env
from rl.common.results_logger import ResultsLogger

import matplotlib
matplotlib.use('TkAgg')


class MCOffPolicy(MonteCarloAgent):
    """
    Monte Carlo Off-Policy control implementation.
    Uses an epsilon-greedy behaviour policy and a deterministic target policy to explore and learn an optimal policy for
    the given environment.

    Args:
        env (Union[Env, object]): The environment to interact with.
        gamma (float): Discount factor for future rewards.
        epsilon (float, optional): Exploration parameter for epsilon-greedy behaviour policy.
        logger (ResultsLogger, optional): Logger for tracking results during training.
        random_seed (int, optional): Seed for reproducibility.
    """

    def __init__(
            self,
            env: Union[Env, object],
            gamma: float,
            epsilon: float = None,
            logger: ResultsLogger = None,
            random_seed: int = None,
    ) -> None:
        super().__init__(env, gamma, epsilon, logger, random_seed)

        self.name: str = "MC Off-Policy"  # For plotting

        self.policy: Union[DeterministicPolicy, None] = None
        self.behaviour_policy: Union[EpsilonGreedyPolicy, None] = None
        self.reset()

    def reset(self)-> None:
        """
        Resets the agent's attributes, including the target and behaviour policies.
        Initialises a deterministic target policy and an epsilon-greedy behaviour policy for
        off-policy Monte Carlo control.
        """
        pass  # TODO: Implement this function


    def act(self, state: Tuple[int, ...])-> int:
        """
        Selects an action based on the behaviour policy (epsilon-greedy with respect to the q-values).

        Args:
            state (Tuple[int, ...]): The current state of the environment.

        Returns:
            int: The action selected by the behaviour policy.
        """
        pass  # TODO: Implement this function


    def _update_q_and_pi(self, episode: List[Tuple[Tuple[int, ...], int, float]])-> None:
        """
        Updates q-values using first-visit Monte Carlo and updates the target policy.

        Args:
            episode (List[Tuple[Tuple[int, ...], int, float]]): The episode to update from,
            consisting of (state, action, reward) tuples.
        """
        pass  # TODO: Implement this function


    def learn(self, num_episodes: int = 10000) -> None:
        """
        Learns an optimal policy using Monte Carlo off-policy control.

        Args:
            num_episodes (int): The number of episodes to train the agent for.
        """
        for episode in range(num_episodes):

            # Print progress
            if episode % 1000 == 0:
                print(f"Episode {episode}/{num_episodes}")

            episode = self._generate_episode()
            self._update_q_and_pi(episode)


def run(num_episodes: int = 50000) -> None:
    """
    Runs the MCOffPolicy agent on the Blackjack environment and plots the results.

    Args:
        num_episodes (int): The number of episodes to train the agent for.
    """
    # Create the environment
    env: Env = gym.make("Blackjack-v1", sab=True)  # `sab` means rules following Sutton and Barto
    rl_loop: MCOffPolicy = MCOffPolicy(env, epsilon=0.1, gamma=1.0)
    rl_loop.learn(num_episodes=num_episodes)

    # Plot the results
    plot_results(rl_loop)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the MCOffPolicy agent on the Blackjack environment.")
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=50000,
        help="Number of episodes to train for. Use a larger number for more convergence."
    )
    args = parser.parse_args()

    run(num_episodes=args.num_episodes)

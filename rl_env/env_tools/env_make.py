from .vector_env import Monitor, DummyVecEnv, SubprocVecEnv
import gymnasium as gym


def make_vec_env(env: gym.Env, n_env: int):
    """
    Creates a vectorized environment, which allows for parallel execution of multiple environments.

    Args:
        env (gym.Env): The environment to be vectorized.
        n_env (int): The number of environments to be created.

    Returns:
        A vectorized environment (either SubprocVecEnv or DummyVecEnv).
    """
    if n_env > 1:
        return SubprocVecEnv([lambda: Monitor(env, allow_early_resets=True) for _ in range(n_env)])
    else:
        return DummyVecEnv([lambda: Monitor(env, allow_early_resets=True) for _ in range(n_env)])


def make_env(
    env: gym.Env,
    n_env: int = 0,
):
    """
    Then Create a vectorized environment using the wrapped environment

    Args:
        env (gym.Env): The environment to be wrapped.
        n_env (int): The number of environments to be created.

    Returns:
        A vectorized wrapped environment.
    """

    # n_env == 0 means performing testing with a single environment
    if n_env == 0:
        print(n_env)
        return env

    # n_env >= 1 means performing training with a single or multiple environments
    return make_vec_env(env, n_env)

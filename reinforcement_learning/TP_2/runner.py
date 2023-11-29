"""
This is the machinery that runs your agent in an environment.

This is not intended to be modified during the practical.
"""


class Runner:
    def __init__(self, environment, agent, verbose=False):
        self.environment = environment
        self.agent = agent
        self.verbose = verbose

    def step(self):
        action = self.agent.choose()
        regret, reward = self.environment.act(action)
        self.agent.update(action, reward)
        return (action, reward, regret)

    def loop(self, iterations):
        cumul_reward = 0.0
        list_cumul = []
        list_regret = []
        for i in range(1, iterations + 1):
            (act, rew, regret) = self.step()
            cumul_reward += rew
            if self.verbose:
                print("Simulation step {}:".format(i))
                print(" ->            action: {}".format(act))
                print(" ->            reward: {}".format(rew))
                print(" -> cumulative reward: {}".format(cumul_reward))
            list_cumul.append(cumul_reward)
            list_regret.append(regret)
        return cumul_reward, list_cumul, list_regret


def iter_or_loopcall_env(o, count, n_actions, variability):
    if callable(o):
        return [o(n_actions, variability=variability) for _ in range(count)]
    else:
        # must be iterable
        return list(iter(o))


def iter_or_loopcall_agent(
    o,
    count,
    n_actions,
    epsilon=None,
    temperature=None,
):
    if callable(o):
        if epsilon is not None:
            return [o(n_actions, epsilon=epsilon) for _ in range(count)]
        elif temperature is not None:
            return [o(n_actions, temperature=temperature) for _ in range(count)]
        else:
            return [o(n_actions) for _ in range(count)]
    else:
        # must be iterable
        return list(iter(o))


class BatchRunner:
    """
    Runs several instances of the same RL problem in parallel
    and aggregates the results.
    """

    def __init__(
        self,
        env_maker,
        agent_maker,
        count,
        n_actions,
        variability,
        epsilon=None,
        temperature=None,
        verbose=False,
    ):
        self.environments = iter_or_loopcall_env(
            env_maker, count, n_actions, variability
        )
        self.agents = iter_or_loopcall_agent(agent_maker, count, n_actions, epsilon, temperature)
        assert len(self.agents) == len(self.environments)
        self.verbose = verbose

    def step(self):
        actions = [agent.choose() for (agent) in self.agents]
        act_returns = [env.act(action) for (env, action) in zip(self.environments, actions)]
        regrets, rewards = zip(*act_returns)
        for agent, action, reward in zip(self.agents, actions, rewards):
            agent.update(action, reward)
        return sum(regrets) / len(regrets), sum(rewards) / len(rewards)

    def loop(self, iterations):
        cum_avg_reward = 0.0
        list_cumul = []
        list_regret = []
        for i in range(1, iterations + 1):
            avg_regret, avg_reward = self.step()
            cum_avg_reward += avg_reward
            if self.verbose:
                print("Simulation step {}:".format(i))
                print(" ->            average reward: {}".format(avg_reward))
                print(" -> cumulative average reward: {}".format(cum_avg_reward))
            list_cumul.append(cum_avg_reward)
            list_regret.append(avg_regret)
        return cum_avg_reward, list_cumul, list_regret

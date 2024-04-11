from stable_baselines3.common.callbacks import EvalCallback


class SelfPlayCallback(EvalCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, verbose: int = 0, *args, **kwargs):
        super(SelfPlayCallback, self).__init__(*args, **kwargs)

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            result = super(SelfPlayCallback, self)._on_step() #this will set self.best_mean_reward to the reward from the evaluation as it's previously -np.inf
            if self.best_mean_reward > 0.3:
                if self.best_mean_reward > self.training_env.get_best_mean_reward:
                    self.training_env.append_opponent_model(self.model, True, self.best_mean_reward)
                else:
                    self.training_env.append_opponent_model(self.model)

        return True
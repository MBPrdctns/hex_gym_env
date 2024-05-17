# from stable_baselines3.common.callbacks import EvalCallback
# from callbacks import EvalCallback

from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback, EvalCallback
import random
import numpy as np
import tensorflow as tf

class SelfPlayCallback(MaskableEvalCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, gym_env, model_log_name, verbose: int = 0, *args, **kwargs):
        super(SelfPlayCallback, self).__init__(*args, **kwargs)
        self.is_tb_set = False
        self.gym_env = gym_env
        self.model_log_name = model_log_name
        # n_eval_episodes # for evaluation in callback
    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.gym_env.set_eval(True)
            result = super(SelfPlayCallback, self)._on_step() #this will set self.best_mean_reward to the reward from the evaluation as it's previously -np.inf
            self.gym_env.set_eval(False)

            score = self.last_mean_reward * np.exp(np.mean(self.gym_env.get_scores())-1)
            if self.last_mean_reward > 0:
                opponent_models = self.gym_env.get_opponent_models()
                scores = self.gym_env.get_scores()
                if score > np.min(scores):
                    if score >= self.gym_env.best_score:
                        self.logger.record("eval/best_score", score)
                    worst_models = np.array(range(len(opponent_models)))[scores == np.min(scores)]
                    if len(worst_models) > 0:
                        i = random.choice(worst_models)
                        self.gym_env.set_opponent_model(i, self.model, score)
                    else:
                        i = worst_models[0]
                        self.gym_env.set_opponent_model(i, self.model, score)
                
                    self.logger.record("eval/buffer_score", score)
            self.logger.record("eval/score", score)
            #self.logger.dump(self.n_calls)
        if self.n_calls % 1000000 == 0:
            self.gym_env.save_best_model()
            self.model.save("models/experiments/" + self.model_log_name + "_" + str(self.n_calls//1000000))
        return True
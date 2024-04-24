# from stable_baselines3.common.callbacks import EvalCallback
# from callbacks import EvalCallback

from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
import random
import numpy as np

class SelfPlayCallback(MaskableEvalCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, gym_env, verbose: int = 0, *args, **kwargs):
        super(SelfPlayCallback, self).__init__(*args, **kwargs)
        self.gym_env = gym_env
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

            if self.last_mean_reward > 0:
                score = self.last_mean_reward * np.exp(np.mean(self.gym_env.get_scores())-1)
                opponent_models = self.gym_env.get_opponent_models()
                scores = self.gym_env.get_scores()
                if score > np.min(scores):
                    worst_models = np.array(range(len(opponent_models)))[scores == np.min(scores)]
                    if len(worst_models) > 0:
                #if self.last_mean_reward > 0:
                        i = random.choice(worst_models)
                    #i = np.argmax(self.gym_env.get_scores())
                    #self.gym_env.set_opponent_model(i, self.model, 0) 
                        self.gym_env.set_opponent_model(i, self.model, score)
                    else:
                        i = worst_models[0]
                        self.gym_env.set_opponent_model(i, self.model, score)
                    #opponent_models[i] = self.model
                    #scores[i] = score

                    #print(f"Number of models in buffer: {len(self.gym_env.opponent_models)}")
                    # print(f"Number of models in best buffer: {len(self.gym_env.best_model)}")
                    #if scor >= self.gym_env.get_best_mean_reward():
                    #    self.gym_env.append_opponent_model(self.model, True, self.last_mean_reward)
                    #else:
                    #    self.gym_env.append_opponent_model(self.model)
        if self.n_calls % 100000 == 0:
            self.gym_env.save_best_model()
            self.model.save("hex_selfplay_model")
        return True
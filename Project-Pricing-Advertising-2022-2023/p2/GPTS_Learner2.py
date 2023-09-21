from p2.GPTS_Learner import *
import utils.projectParameters as param


class GPTS_Learner2(Learner):
    def __init__(self, arms, class_id):
        super().__init__(arms.shape[0])
        self.n_click_learner = GPTS_Learner(arms)
        self.cumcost_learner = GPTS_Learner(arms)
        self.class_id = class_id

    def update(self, pulled_arm, n_clicks, cum_cost, reward):
        self.update_observations(pulled_arm, reward)
        self.n_click_learner.update(pulled_arm, n_clicks)
        self.cumcost_learner.update(pulled_arm, cum_cost)

    def pull_arm(self):
        sampled_n_clicks = np.random.normal(self.n_click_learner.means, self.n_click_learner.sigmas)
        sampled_cum_cost = np.random.normal(self.cumcost_learner.means, self.cumcost_learner.sigmas)
        optimal_price_idx = np.argmax(param.pricing_probabilities_per_user[self.class_id] * (param.prices - param.cost))
        sampled_reward = param.pricing_probabilities_per_user[self.class_id][optimal_price_idx] * sampled_n_clicks * (
                    param.prices[optimal_price_idx] - param.cost) - sampled_cum_cost
        return np.random.choice(np.where(sampled_reward == sampled_reward.max())[0])

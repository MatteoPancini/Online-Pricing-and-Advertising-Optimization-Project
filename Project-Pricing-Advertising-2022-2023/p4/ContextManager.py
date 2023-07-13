import pandas as pd
import numpy as np
import pandas as pd
from p4.Multi_TS_Learner import *
import parameters as param


class ContextManager:
    def __init__(self, features_names, features):
        self.features_names = features_names
        self.features = features
        self.samples = pd.DataFrame(
            columns=[*self.features_names, 'bid', 'price', 'n_conversions', 'n_clicks', 'cum_costs', 'reward'])

    def update(self, new_samples):
        new_samples_df = pd.DataFrame(new_samples,
                                      columns=[*self.features_names, 'bid', 'price', 'n_conversions', 'n_clicks',
                                               'cum_costs',
                                               'reward'])
        self.samples = pd.concat([self.samples, new_samples_df])

    def get_context(self):  # convert list of dictionaries into list of lists
        context_dict = self.get_context_recursive()
        context_list = []
        if len(context_dict) == 0:
            return [param.feature_combos]
        for dictionary in context_dict:
            class_list = []
            for i in range(len(param.features_names)):
                if param.features_names[i] not in dictionary.keys():
                    class_list_copy = class_list.copy()
                    if len(class_list_copy) == 0:
                        class_list = [['0'], ['1']]
                    else:
                        for class_name_list in class_list_copy:
                            class_name_list_0 = class_name_list + ['0']
                            class_name_list_1 = class_name_list + ['1']
                            class_list.append(class_name_list_0)
                            class_list.append(class_name_list_1)
                            class_list.remove(class_name_list)
                else:
                    if len(class_list) == 0:
                        class_list = [[str(dictionary[param.features_names[i]])]]
                    else:
                        for class_name_list in class_list:
                            class_name_list.append(str(dictionary[param.features_names[i]]))
            context_list.append([''.join(class_name_list) for class_name_list in class_list])
        return context_list

    def get_context_recursive(self, features=param.features_names, samples=None, lower_bound_mean=None):
        if len(features) == 0:
            return []
        features_names_to_be_split = features.copy()
        if samples is None:
            samples = self.samples
            conversion_per_price = samples.groupby(['price'])['n_conversions'].sum().astype(float)
            n_clicks_per_price = samples.groupby(['price'])['n_clicks'].sum().astype(float)
            conversion_rate_per_price = conversion_per_price.divide(n_clicks_per_price).astype(float)
            lb_per_price = conversion_rate_per_price - np.sqrt(
                np.array(-np.log(param.confidence) / (2 * n_clicks_per_price)).astype(float))
            lb_pricing_probability = lb_per_price.max()
            price_max_idx = lb_per_price.idxmax()
            margin = param.prices[price_max_idx] - param.cost

            n_clicks_per_bid = samples.groupby(['bid'])['n_clicks']
            cum_costs_per_bid = samples.groupby(['bid'])['cum_costs']
            n_clicks_per_bid_lb = n_clicks_per_bid.mean().to_numpy() - 1.96 * n_clicks_per_bid.std().fillna(
                1e6).to_numpy() / np.sqrt(n_clicks_per_bid.size().to_numpy())
            cum_costs_per_bid_lb = cum_costs_per_bid.mean().to_numpy() + 1.96 * cum_costs_per_bid.std().fillna(
                1e6).to_numpy() / np.sqrt(cum_costs_per_bid.size().to_numpy())
            lower_bound_mean = np.max(
                (lb_pricing_probability * n_clicks_per_bid_lb * margin - cum_costs_per_bid_lb) / n_clicks_per_bid_lb)
        max_split_value = -np.infty
        max_feature = None
        max_filtered_samples_0 = None
        max_lower_bound_0 = None
        max_filtered_samples_1 = None
        max_lower_bound_1 = None
        for feature in features:
            # filter df
            filtered_samples_per_feature_0 = samples.copy().loc[samples[feature] == '0']
            lower_bounds_0 = self.get_lower_bounds(filtered_samples_per_feature_0, samples)
            filtered_samples_per_feature_1 = samples.copy().loc[samples[feature] == '1']
            lower_bounds_1 = self.get_lower_bounds(filtered_samples_per_feature_1, samples)
            split_value = lower_bounds_0[0] * lower_bounds_0[1] + lower_bounds_1[0] * lower_bounds_1[1]
            if split_value >= lower_bound_mean and split_value > max_split_value:
                max_split_value = split_value
                max_filtered_samples_0 = filtered_samples_per_feature_0.copy()
                max_lower_bound_0 = lower_bounds_0[1]
                max_filtered_samples_1 = filtered_samples_per_feature_1.copy()
                max_lower_bound_1 = lower_bounds_1[1]
                max_feature = feature
        if max_feature is not None:
            features_names_to_be_split.remove(max_feature)
            context_0 = self.get_context_recursive(features_names_to_be_split, max_filtered_samples_0,
                                                   max_lower_bound_0)
            if len(context_0) == 0:
                context_0.append({max_feature: 0})
            else:
                for sub_context in context_0:
                    sub_context[max_feature] = 0
            context_1 = self.get_context_recursive(features_names_to_be_split, max_filtered_samples_1,
                                                   max_lower_bound_1)
            if len(context_1) == 0:
                context_1.append({max_feature: 1})
            else:
                for sub_context in context_1:
                    sub_context[max_feature] = 1
            return context_0 + context_1
        return []

    def get_lower_bounds(self, filtered_samples, total_samples):
        n_samples_filtered = np.sum(filtered_samples['n_clicks'].to_numpy())
        n_samples_total = np.sum(total_samples['n_clicks'].to_numpy())
        lb_probability = n_samples_filtered / n_samples_total - np.sqrt(
            -np.log(param.confidence) / (2 * n_samples_filtered))

        conversion_per_price = filtered_samples.groupby(['price'])['n_conversions'].sum().astype(float)
        n_clicks_per_price = filtered_samples.groupby(['price'])['n_clicks'].sum().astype(float)
        conversion_rate_per_price = conversion_per_price.divide(n_clicks_per_price).astype(float)
        lb_per_price = conversion_rate_per_price - np.sqrt(
            np.array(-np.log(param.confidence) / (2 * n_clicks_per_price)).astype(float))
        lb_pricing_probability = lb_per_price.max()
        price_max_idx = lb_per_price.idxmax()
        margin = param.prices[price_max_idx] - param.cost

        n_clicks_per_bid = filtered_samples.groupby(['bid'])['n_clicks']
        cum_costs_per_bid = filtered_samples.groupby(['bid'])['cum_costs']
        n_clicks_per_bid_lb = n_clicks_per_bid.mean().to_numpy() - 1.96 * n_clicks_per_bid.std().fillna(
            1e6).to_numpy() / np.sqrt(n_clicks_per_bid.size().to_numpy())
        cum_costs_per_bid_lb = cum_costs_per_bid.mean().to_numpy() + 1.96 * cum_costs_per_bid.std().fillna(
            1e6).to_numpy() / np.sqrt(cum_costs_per_bid.size().to_numpy())
        reward_mean_per_bid = (
                                          lb_pricing_probability * n_clicks_per_bid_lb * margin - cum_costs_per_bid_lb) / n_clicks_per_bid_lb
        return lb_probability, np.max(reward_mean_per_bid)

    def get_samples(self, context):
        filtered_samples = pd.DataFrame(
            columns=[*self.features_names, 'bid', 'price', 'n_conversions', 'n_clicks', 'cum_costs', 'reward'])
        for feature_combo in context:
            filtered_samples_per_feature = self.samples
            features = list(feature_combo)
            for i in range(len(param.features_names)):
                filtered_samples_per_feature = filtered_samples_per_feature.loc[
                    filtered_samples_per_feature[param.features_names[i]] == features[i]]
            filtered_samples = pd.concat([filtered_samples, filtered_samples_per_feature])
        return [filtered_samples[column].to_numpy() for column in filtered_samples.columns[2:]]


class ContextOptimizer:
    def __init__(self, optimizer_type):
        """
        param.features_names = [F1, F2]
        param.features_combos = [00, 01, 10, 11]
        optimizer_type: classe (MultiLearner: TS / UCB)
        """
        self.context_generator = ContextManager(param.features_names, param.feature_combos) #crea il context generator con la lista di feature e le possibili combinazioni
        self.optimizer_type = optimizer_type #assegna il learner
        self.context_wise_learner = {
            tuple(param.feature_combos): self.optimizer_type(param.bids, param.prices) #initializes one learner per context (each feature combination creates a context)
        }
        self.T = 0
        self.collected_rewards = []

    def pull_arms(self):
        bids_and_prices = {}
        for context in self.context_wise_learner.keys(): #for each possible context (feature combination)
            arms = self.context_wise_learner[context].pull_arms() #pull arm
            for feature in context:
                bids_and_prices[feature] = arms
        return bids_and_prices

    def update(self, input_per_feature):
        self.T += 1
        # update
        for context in self.context_wise_learner.keys():
            pulled_bids_arm = input_per_feature[context[0]][0]
            pulled_prices_arm = input_per_feature[context[0]][1]
            n_conversion_per_context = sum(input_per_feature[feature][2] for feature in context)
            n_clicks_per_context = sum(input_per_feature[feature][3] for feature in context)
            cum_cost_per_context = sum(input_per_feature[feature][4] for feature in context)
            reward_per_context = sum(input_per_feature[feature][5] for feature in context)
            self.context_wise_learner[context].update(pulled_bids_arm, pulled_prices_arm, n_conversion_per_context,
                                                n_clicks_per_context, cum_cost_per_context, reward_per_context)
        self.collected_rewards.append(sum(input_per_feature[feature][5] for feature in param.feature_combos))
        self.context_generator.update(
            [tuple(feature) + input_per_feature[feature] for feature in input_per_feature.keys()])
        # generate contexts
        if self.T % 14 == 0:
            context_structure = [tuple(i) for i in self.context_generator.get_context()]
            keys = list(self.context_wise_learner.keys()).copy()
            for context in keys:
                if context not in context_structure:
                    del self.context_wise_learner[context]
            for context in context_structure:
                if context not in keys:
                    self.context_wise_learner[tuple(context)] = self.optimizer_type(param.bids, param.prices)
                    # retrieve samples
                    samples = self.context_generator.get_samples(context)
                    # bulk update learner
                    self.context_wise_learner[tuple(context)].update_bulk(*samples)
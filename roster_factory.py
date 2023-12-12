import copy

import numpy as np
import pandas as pd

from helpers import list_to_binary_array
from main import n_days
from partial_roster import PartialRoster


def calculate_roster_df(nurse_df, n_days, n_work_shifts, cost_parameters, feasibility_parameters):
    roster_df = pd.DataFrame()
    for nurse_type, nurse_hours in nurse_df[['nurseType', 'nurseHours']].itertuples(index=False):
        base_roster = PartialRoster(n_days=n_days,
                                    nurse_type=nurse_type,
                                    nurse_hours=nurse_hours,
                                    n_work_shifts=n_work_shifts,
                                    cost_parameters=cost_parameters,
                                    feasibility_parameters=feasibility_parameters)

        n_shifts_for_nurse_type = feasibility_parameters.avg_shifts_per_period[nurse_type]

        rosters = []
        finished_rosters_data = []
        for shift in range(n_work_shifts + 1):
            roster = copy.deepcopy(base_roster)
            roster.increment(shift)
            rosters.append(roster)

        while len(rosters) > 0:
            roster = rosters.pop()
            shifts = set(roster.feasible_shifts())
            for shift in shifts:
                new_roster = copy.deepcopy(roster)
                new_roster.increment(shift)
                if new_roster.is_finished():
                    if new_roster.work_shifts_total >= n_shifts_for_nurse_type - 1:
                        individual_cost, fair_cost = new_roster.calculate_cost()
                        total_individual_cost, total_fair_cost = sum(individual_cost.values()), sum(fair_cost.values())
                        finished_rosters_data.append(
                            new_roster.plan + list(individual_cost.values()) + list(fair_cost.values())
                            + [total_individual_cost, total_fair_cost, total_individual_cost + total_fair_cost,
                               new_roster.nurse_type, new_roster.nurse_hours])
                else:
                    rosters.append(new_roster)

        roster_df_ = pd.DataFrame(finished_rosters_data)
        roster_df_.columns = np.arange(n_days).tolist() + list(individual_cost.keys()) + list(fair_cost.keys()) + \
                             ['totalIndividualCost', 'totalFairCost', 'totalCost', 'nurseType', 'nurseHours']

        roster_df_ = roster_df_.assign(workShifts=lambda x: np.sum(x.loc[:, 0:n_days - 1] < 3, axis=1)) \
            .assign(nurseHours=nurse_hours)

        roster_df_ = roster_df_[roster_df_.workShifts >= n_shifts_for_nurse_type - 1]
        print(roster_df_.shape[0])
        roster_df = pd.concat([roster_df, roster_df_])

    roster_df = roster_df.sort_values(['nurseType', 'totalCost'])
    roster_df['rosterIndex'] = np.arange(roster_df.shape[0])
    roster_df = roster_df.reset_index()
    print(roster_df.shape)

    return roster_df


def initial_solution_for_cg(nurse_df, roster_df, n_largest_for_each_nurse, n_smallest_for_each_nurse):
    """This initial solution contains expensive 0s ,1s and 2s plans for all nurse types along with a set of the
    cheapest plans for all nurse types"""
    roster_largest_cost_df = roster_df.merge(roster_df.rename_axis('rosterIndex').groupby('nurseType')['totalCost']
                                             .nlargest(n_largest_for_each_nurse).reset_index().drop(
        columns=['totalCost']),
                                             how='inner', on=['nurseType', 'rosterIndex'])
    largest_cost_array = np.tile(np.concatenate([0 * np.ones((1, n_days)),
                                                 1 * np.ones((1, n_days)),
                                                 2 * np.ones((1, n_days))]),
                                 (nurse_df.nurseType.nunique(), 1)).astype(int)
    roster_largest_cost_df.loc[:, [str(x) for x in range(n_days)]] = largest_cost_array
    roster_largest_cost_df.loc[:, 'rosterIndex'] = np.arange(roster_df.shape[0],
                                                             roster_df.shape[0] + roster_largest_cost_df.shape[0])
    roster_largest_cost_df.loc[:, 'totalCost'] = 9999
    roster_smallest_cost_df = roster_df.merge(roster_df.rename_axis('rosterIndex').groupby('nurseType')['totalCost']
                                              .nsmallest(n_smallest_for_each_nurse).reset_index().drop(
        columns=['totalCost']),
                                              how='inner', on=['nurseType', 'rosterIndex'])
    roster_indices = dict()
    for nurse_type in nurse_df.nurseType:
        small_cost_set = set(roster_smallest_cost_df[roster_smallest_cost_df.nurseType == nurse_type].rosterIndex)
        large_cost_set = set(roster_largest_cost_df[roster_largest_cost_df.nurseType == nurse_type].rosterIndex)
        roster_indices[nurse_type] = small_cost_set.union(large_cost_set)
    return roster_indices

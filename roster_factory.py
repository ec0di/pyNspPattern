import copy

import numpy as np
import pandas as pd

from helpers import list_to_binary_array
from partial_roster import PartialRoster


def calculate_roster_df(nurse_df, n_days, n_work_shifts, cost_parameters, feasibility_parameters):
    roster_df = pd.DataFrame()
    count = 0
    binary_plans = {}
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
                        binary_plans[count] = list_to_binary_array(new_roster.plan, n_days, n_work_shifts)
                        total_individual_cost, total_fair_cost = sum(individual_cost.values()), sum(fair_cost.values())
                        finished_rosters_data.append(
                            new_roster.plan + list(individual_cost.values()) + list(fair_cost.values())
                            + [total_individual_cost, total_fair_cost, total_individual_cost + total_fair_cost,
                               new_roster.nurse_type, new_roster.nurse_hours])
                        count += 1
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

    roster_df['rosterIndex'] = np.arange(roster_df.shape[0])
    roster_df = roster_df.reset_index()
    print(roster_df.shape)

    return roster_df, binary_plans

import numpy as np
import pandas
import pandas as pd
import os

OFF_SHIFT = 3
MAX_CONSECUTIVE_WORK_SHIFTS = 5
DELTA_NURSE_SHIFT = 1
SHIFT_LENGTH_IN_HOURS = 8

COSTS = {
    'consecutiveShifts': -0.04,
    'missingTwoDaysOffAfterNightShifts': 0.1,
    'moreThanTwoConsecutiveNightShifts': 1,
    'singleNightShift': 1,
    'moreThanFourConsecutiveWorkShifts': 1,
    'afternoonShiftsFair': None,
    'nightShiftsFair': None,
    'nightAndAfternoonShiftsFair': None,
    'weekendShiftsFair': None}


def is_weekend(j, k):
    if k == 6:  # this is free-shift, thus weekend
        return False
    elif np.mod(j, 7) == 4 and k in (1, 2):
        return True
    elif np.mod(j, 7) >= 5:
        return True
    else:
        return False


def bin_array_to_list(b, n_days, n_work_shifts):
    roster = []
    for j in range(n_days):
        if sum(b[j, :]):
            for k in range(n_work_shifts):
                if b[j, k] == 1:
                    roster.append(k)
        else:
            roster.append(6)
    return roster


class RecursiveRosterParameters:
    def __init__(self, max_nodes_per_day, min_all_node_days, max_roster_cost, max_roster_number, max_nodes_searched,
                 max_times_node_visited, shuffle_frequency):
        self.max_nodes_per_day = max_nodes_per_day
        self.min_all_node_days = min_all_node_days
        self.max_roster_cost = max_roster_cost
        self.max_roster_number = max_roster_number
        self.max_nodes_searched = max_nodes_searched
        self.max_times_node_visited = max_times_node_visited
        self.shuffle_frequency = shuffle_frequency


class CostParameters:
    def __init__(self, hard_shift_fair_per_period, avg_weekend_shifts_per_person_per_period,
                 hard_shifts_fair_plans_factor, weekend_shifts_fair_plan_factor):
        self.hard_shift_fair_per_period = hard_shift_fair_per_period
        self.avg_weekend_shifts_per_person_per_period = avg_weekend_shifts_per_person_per_period
        self.hard_shifts_fair_plans_factor = hard_shifts_fair_plans_factor
        self.weekend_shifts_fair_plan_factor = weekend_shifts_fair_plan_factor
        self.count_cost_cases = {cost: 0 for cost in COSTS.keys()}


class FeasibilityParameters:
    def __init__(self, avg_shifts_per_period):
        self.avg_shifts_per_period = avg_shifts_per_period


def get_solution_values(z, nurseTypes):
    z_dict = {k: v.solution_value() for k, v in z.items() if v.solution_value() >= 1}
    nurseType_and_roster_number_not_covered = []
    for (i, r) in z_dict.keys():
        if i == len(nurseTypes):
            nurseType_and_roster_number_not_covered.append((i,r))
    return z_dict, nurseType_and_roster_number_not_covered


def get_day_shift_from_nurse_roster(rosters, nurseType, roster_num):
    temp = np.where(rosters[nurseType][roster_num][0] >= 1)
    day, shift = temp[0][0], temp[1][0]
    return day, shift


def get_demand(base_demand, n_weeks):
    demand = base_demand * 1
    demand = np.tile(demand, (1, n_weeks))  # D, 98 in total per week
    return demand


def get_best_nurseTypes_sorted_low_to_high(beta, beta_dict=None):
    if beta_dict is None:
        beta_dict = {nurseType_: v.dual_value() for nurseType_, v in beta.items()}
    beta_dict_sorted = dict(sorted(beta_dict.items(), key=lambda item: item[1]))
    nurseType_list = list(beta_dict_sorted.keys())
    return nurseType_list


def calculate_parameters(n_weeks, n_work_shifts, nurse_df, base_demand, hard_shifts_fair_plans_factor,
                         weekend_shifts_fair_plan_factor, shift_length_in_hours):
    weekend_shifts_per_week = sum(
        [base_demand[k, j] for j in range(7) for k in range(n_work_shifts) if is_weekend(j, k)])
    night_shifts_per_week = sum(base_demand[2, :])

    avg_weekend_shifts_per_person_per_period = 1.0 * weekend_shifts_per_week / sum(nurse_df['nurseCount']) * n_weeks
    avg_shifts_per_period = {nurse_hours: nurse_hours / shift_length_in_hours * n_weeks
                             for nurse_hours in nurse_df.nurseHours.unique()}
    total_shifts_available_per_period = sum(nurse_df['nurseHours'] / shift_length_in_hours * nurse_df['nurseCount'] * n_weeks)
    hard_shift_fair_per_period = {nurse_hours: avg_shifts_per_period / total_shifts_available_per_period * night_shifts_per_week * n_weeks
                                  for nurse_hours, avg_shifts_per_period in avg_shifts_per_period.items()}

    cost_parameters = CostParameters(hard_shift_fair_per_period, avg_weekend_shifts_per_person_per_period,
                                     hard_shifts_fair_plans_factor, weekend_shifts_fair_plan_factor)

    feasibility_parameters = FeasibilityParameters(avg_shifts_per_period)
    return cost_parameters, feasibility_parameters


def list_to_binary_array(plan, n_days, n_work_shifts):
    x = np.zeros((n_days, n_work_shifts))
    for j, e in enumerate(plan):
        if e != OFF_SHIFT:
            x[j, e] = 1
    return x


def set_dataframe_print_width(desired_width = 320):
    pd.set_option('display.width', desired_width)
    np.set_printoptions(linewidth=desired_width)
    pd.set_option('display.max_columns', None)


def downcast_dataframe(df):
    """downcast to save memory"""
    return (
        df
        .apply(pd.to_numeric, downcast="float")
        .apply(pd.to_numeric, downcast="integer")
        .apply(pd.to_numeric, downcast="unsigned")
    )


def write_to_parquet(df, filename):
    full_filename = filename + '.parquet'
    print(f'Writing file: {full_filename}')
    df.to_parquet(full_filename, index=False)


def hotfix_for_pandas_merge(r_indices_df, roster_df):
    """Since downcasting changes the type of the columns, we need to change it back to standard dtypes
    for pandas merge to work on Windows"""
    r_indices_df.to_parquet('data/r_indices_df' + '.parquet', index=False)
    roster_df.to_parquet('data/roster_df' + '.parquet', index=False)

    r_indices_df = pd.read_parquet('data/r_indices_df' + '.parquet')
    roster_df = pd.read_parquet('data/roster_df' + '.parquet')
    # delete files
    os.remove('data/r_indices_df' + '.parquet')
    os.remove('data/roster_df' + '.parquet')
    return r_indices_df, roster_df

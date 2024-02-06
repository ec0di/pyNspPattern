import numpy as np
import pandas as pd
import os

from input_parameters import OFF_SHIFT, COSTS


def is_weekend(j, k):
    if k == 6:  # this is free-shift, thus weekend
        return False
    elif np.mod(j, 7) == 4 and k in (1, 2):
        return True
    elif np.mod(j, 7) >= 5:
        return True
    else:
        return False


class ColumnGenerationParameters:
    def __init__(self, max_time_sec, max_time_per_iteration_sec, max_iter, n_rosters_per_iteration):
        self.max_time_sec = max_time_sec
        self.max_time_per_iteration_sec = max_time_per_iteration_sec
        self.max_iter = max_iter
        self.n_rosters_per_iteration = n_rosters_per_iteration
        

class RosterCostParameters:
    def __init__(self, hard_shift_fair_per_period, avg_weekend_shifts_per_person_per_period,
                 hard_shifts_fair_plans_factor, weekend_shifts_fair_plan_factor):
        self.hard_shift_fair_per_period = hard_shift_fair_per_period
        self.avg_weekend_shifts_per_person_per_period = avg_weekend_shifts_per_person_per_period
        self.hard_shifts_fair_plans_factor = hard_shifts_fair_plans_factor
        self.weekend_shifts_fair_plan_factor = weekend_shifts_fair_plan_factor
        self.count_cost_cases = {cost: 0 for cost in COSTS.keys()}


class RosterFeasibilityParameters:
    def __init__(self, avg_shifts_per_period):
        self.avg_shifts_per_period = avg_shifts_per_period


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

    cost_parameters = RosterCostParameters(hard_shift_fair_per_period, avg_weekend_shifts_per_person_per_period,
                                           hard_shifts_fair_plans_factor, weekend_shifts_fair_plan_factor)

    feasibility_parameters = RosterFeasibilityParameters(avg_shifts_per_period)
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
    print(f'Writing file: {filename}')
    df.to_parquet(filename, index=False)


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

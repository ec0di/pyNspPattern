import numpy as np
import pandas as pd
from functools import wraps
from time import time


OFF_SHIFT = 3

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


def list_to_binary_array(plan, n_days, n_work_shifts):
    x = np.zeros((n_days, n_work_shifts))
    for j, e in enumerate(plan):
        if e != OFF_SHIFT:
            x[j, e] = 1
    return x


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


def calculate_cost(rosters):
    costs = 0
    for k, roster_list in rosters.items():
        for r in roster_list:
            cost = r[1]
            costs += cost
    return costs


def calculate_demand_cover(rosters, n_days, n_work_shifts, nurseType_and_roster_number=None):
    arr = np.zeros((n_days, n_work_shifts))
    if nurseType_and_roster_number is not None:
        for (nurseType, roster_num) in nurseType_and_roster_number:
            arr += rosters[nurseType][roster_num][0]
    else:
        for nurseType, roster_list in rosters.items():
            for r in roster_list:
                arr += r[0]
    return arr


def calculate_off_cover(rosters, n_days, n_work_shifts):
    arr = np.zeros(n_days)
    for k, roster_list in rosters.items():
        for r in roster_list:
            for day in range(n_days):
                if sum(r[0][day, :]) == 0:
                    arr[day] += 1
    return arr


def get_shifts_not_covered(rosters, n_days, n_work_shifts, nurseType_and_roster_number_not_covered):
    x, y = np.where(calculate_demand_cover(rosters, n_days, n_work_shifts,
                                           nurseType_and_roster_number=nurseType_and_roster_number_not_covered))
    hard_shifts_to_cover = [(x_, y_) for x_, y_ in zip(x, y)]
    return hard_shifts_to_cover


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


def calculate_parameters(n_weeks, n_work_shifts, nurse_df, base_demand, hard_shifts_fair_plans_factor, weekend_shifts_fair_plan_factor):
    weekend_shifts_per_week = sum(
        [base_demand[k, j] for j in range(7) for k in range(n_work_shifts) if is_weekend(j, k)])
    night_shifts = sum(base_demand[2, :])

    shift_length_in_hours = 8
    avg_weekend_shifts_per_person_per_period = 1.0 * weekend_shifts_per_week / sum(nurse_df['nurseCount'].values) * n_weeks
    avg_shifts_per_period = nurse_df['nurseHours'].values / shift_length_in_hours * n_weeks
    total_shifts_available_per_period = sum(avg_shifts_per_period * nurse_df['nurseCount'].values)
    hard_shift_fair_per_period = avg_shifts_per_period * night_shifts / total_shifts_available_per_period * n_weeks

    cost_parameters = CostParameters(hard_shift_fair_per_period, avg_weekend_shifts_per_person_per_period,
                                     hard_shifts_fair_plans_factor, weekend_shifts_fair_plan_factor)

    feasibility_parameters = FeasibilityParameters(avg_shifts_per_period)
    return cost_parameters, feasibility_parameters


def read_rosters_from_parquet(parquet_filename, n_days, n_work_shifts):
    start_time = time()
    roster_df = pd.read_parquet(parquet_filename)
    second_time = time()
    print(f'Read parquet file in {round(second_time - start_time, 1)} s')
    binary_plans = {}
    for plan in roster_df.loc[:, [str(x) for x in np.arange(n_days)] + ['rosterIndex']].itertuples(index=False):
        binary_plans[plan.rosterIndex] = list_to_binary_array(plan[:-1], n_days, n_work_shifts)
    print(f'Created binary plans in {round(time() - second_time, 1)} s')
    return roster_df, binary_plans


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap

import numpy as np
import pandas as pd
import json

from master_problem import master_problem_instance
from helpers import get_demand, calculate_parameters, set_dataframe_print_width, \
    DELTA_NURSE_SHIFT, SHIFT_LENGTH_IN_HOURS
from roster_factory import RosterFactory


n_weeks = 1  # works for 1 week with nurse_type from bla
read_roster_df = False
use_start_conditions_from_first_two_weeks = False
use_initial_solution = False
base_path = ''
start_condition_filename = f'{base_path}data/{n_weeks}WeekRosterSolutionOptimal'

max_time_sec = 30
max_iter = 1000
n_rosters_per_iteration = 300
set_dataframe_print_width()

n_work_shifts = 3
n_days = n_weeks * 7
parquet_filename = f'{base_path}data/{n_weeks}WeekUniqueRosters.parquet'
nurse_df_multiplier = 4

# demand per week
base_demand = np.array([[3, 4, 3, 4, 3, 2, 2],
                        [2, 2, 2, 2, 2, 2, 2],
                        [2, 2, 2, 2, 2, 2, 2]])

base_demand *= nurse_df_multiplier

demand = get_demand(base_demand, n_weeks)

nurse_df = pd.DataFrame({'nurseHours': [28, 28, 32, 32, 37, 37],
                         'nurseLevel': [1, 3, 1, 3, 1, 3],
                         'nurseCount': [1, 1, 1, 4, 4, 3]})
nurse_df['nurseIndex'] = np.arange(len(nurse_df))
nurse_df['lastOneWeekRosterIndex'] = -1  # means all rosters are available
nurse_df['lastTwoWeekRosterIndex'] = -1  # means all rosters are available
nurse_df = nurse_df.assign(nurseShifts=lambda x: x.nurseHours // 8 * n_weeks,
                           nurseShiftsMin=lambda x: x.nurseShifts - DELTA_NURSE_SHIFT,
                           nurseShiftsMax=lambda x: x.nurseShifts + DELTA_NURSE_SHIFT)
nurse_df.nurseCount *= nurse_df_multiplier

# quick check of demand vs supply
print('demand vs supply: ', sum(sum(base_demand)), 'vs', sum(nurse_df.nurseHours / 8 * nurse_df.nurseCount))

# factors
hard_shifts_fair_plans_factor = 0.5
weekend_shifts_fair_plan_factor = 0.5

cost_parameters, feasibility_parameters = calculate_parameters(n_weeks, n_work_shifts, nurse_df, base_demand,
                                                               hard_shifts_fair_plans_factor,
                                                               weekend_shifts_fair_plan_factor,
                                                               SHIFT_LENGTH_IN_HOURS)

roster_factory = RosterFactory(n_weeks, n_work_shifts, nurse_df, cost_parameters, feasibility_parameters)

if read_roster_df:
    roster_df = roster_factory.read_roster_df_from_parquet(parquet_filename)
else:
    roster_df = roster_factory.calculate_roster_df()
    roster_df.columns = [str(colname) for colname in roster_df.columns]

    # create 2 week roster df with 1week rosters matching
    if n_weeks == 2:
        roster1_df = roster_factory.read_roster_df_from_parquet(parquet_filename=f'{base_path}data/1WeekRosters.parquet')
        roster_df = roster_factory.append_one_week_roster_index_to_two_week_roster_df(roster1_df)
    roster_df.to_parquet(parquet_filename, index=False)
# create map of day, work_shifts to rosters
roster_factory.append_day_work_shift_flags()

roster_matching_file = f'data/1WeekRosterMatchingUnique.json'
if n_weeks == 1:
    roster_factory.calculate_roster_matching()
    # serialize roster matching
    with open(roster_matching_file, 'w') as fp:
        json.dump(roster_factory.roster_matching, fp)
if use_start_conditions_from_first_two_weeks:
    if n_weeks == 2:
        # deserialize roster matching
        with open(roster_matching_file, 'r') as fp:
            roster_matching = json.load(fp)
            roster_factory.roster_matching = {int(key): value for key, value in roster_matching.items()}

    # create nurse_df from solution file
    roster_solution_df = pd.read_parquet(f'{start_condition_filename}StartCondition.parquet')
    nurse_df = roster_solution_df.assign(lastOneWeekRosterIndex=lambda x: x.rosterIndexWeek2,
                                         lastTwoWeekRosterIndex=lambda x: x.rosterIndex).\
        groupby(['nurseHours', 'nurseLevel', 'lastOneWeekRosterIndex', 'lastTwoWeekRosterIndex']).\
        agg(nurseCount=('nRostersInSolution', 'sum')).reset_index().astype({'nurseCount': 'int32'})
    roster_factory.nurse_df = nurse_df

if use_initial_solution:
    n_largest_for_each_nurse = 3  # necessary with 3 to get full 0s, 1s, and 2s plans
    n_smallest_for_each_nurse = 5 ** n_weeks
    roster_indices, binary_plans, roster_costs = roster_factory.initial_solution_for_cg(n_largest_for_each_nurse,
                                                                                        n_smallest_for_each_nurse)

    roster_factory.run_column_generation(verbose=True,
                                         demand=demand,
                                         max_time_sec=max_time_sec,
                                         max_iter=max_iter,
                                         min_object_value=15,
                                         n_rosters_per_iteration=n_rosters_per_iteration,
                                         solver_id='GLOP',
                                         max_time_per_iteration_sec=300)

else:  # full set of rosters solution
    roster_indices, binary_plans, roster_costs = roster_factory.full_solution_for_mip()

# run final master problem with MIP
solver, nurse_c, demand_c, demand_comp_level_c, z, status = master_problem_instance(n_days=n_days,
                                                                                    n_work_shifts=n_work_shifts,
                                                                                    nurse_df=roster_factory.nurse_df,
                                                                                    roster_indices=roster_factory.roster_indices,
                                                                                    roster_costs=roster_factory.roster_costs,
                                                                                    binary_plans=roster_factory.binary_plans,
                                                                                    demand=demand,
                                                                                    max_time_solver_sec=1000, solver_id='CBC')

if status == 0:  # optimal solution found
    z_int = {key: value.solution_value() for key, value in z.items()}

    r_indices_df = pd.DataFrame(list(keys)+[value] for keys, value in z_int.items() if value >= 1)
    r_indices_df.columns = ['nurseHours', 'nurseLevel', 'lastOneWeekRosterIndex', 'lastTwoWeekRosterIndex', 'rosterIndex', 'nRostersInSolution']

    # nice to remember: .loc[lambda x: x.nurseType == 2]
    roster_solution_df = roster_factory.roster_df.merge(r_indices_df, how='inner', on=['nurseHours', 'rosterIndex'])
    print(roster_solution_df.loc[:, [str(x) for x in np.arange(n_days)]])
    path = f'data/{n_weeks}WeekRosterSolutionOptimal'
    roster_solution_df.to_parquet(path + '.parquet', index=False)
    if not use_start_conditions_from_first_two_weeks:
        roster_solution_df.to_parquet(path + 'StartCondition.parquet', index=False)

import numpy as np
import pandas as pd
pd.options.plotting.backend = "plotly"

from master_problem import master_problem_instance
from helpers import get_demand, calculate_parameters, set_dataframe_print_width, \
    SHIFT_LENGTH_IN_HOURS, write_to_parquet, hotfix_for_pandas_merge, N_WORK_SHIFTS, BASE_DEMAND, NURSE_DF_MULTIPLIER

from roster_factory import RosterFactory

n_weeks = 1  # works for 1 week with nurse_type from bla
read_roster_df = False
use_initial_solution = False
use_start_conditions_from_first_two_weeks = False
verbose = True

roster_matching_file = f'data/1WeekRosterMatching.json'
roster_df_file_by_n_weeks = lambda n_weeks: f'data/{n_weeks}WeekRosters.parquet'

#solution_export_path = f'data/{n_weeks}WeekRosterSolution' if use_initial_solution else f'data/{n_weeks}WeekRosterSolutionOptimal'
solution_write_path = f'data/{n_weeks}WeekRosterSolutionTest'

# parameters
pct_of_best_rosters_to_keep = 0.25

# column generation parameters
max_time_sec = 5
max_iter = 1000
n_rosters_per_iteration = 300

set_dataframe_print_width()
n_days = n_weeks * 7

demand = get_demand(BASE_DEMAND, n_weeks)

nurse_df = pd.DataFrame({'nurseHours': [28, 28, 32, 32, 37, 37],
                         'nurseLevel': [1, 3, 1, 3, 1, 3],
                         'nurseCount': [1, 1, 1, 4, 4, 3]})
nurse_df['nurseIndex'] = np.arange(len(nurse_df))
nurse_df['lastOneWeekRosterIndex'] = -1  # means all rosters are available
nurse_df['lastTwoWeekRosterIndex'] = -1  # means all rosters are available
nurse_df = nurse_df.assign(nurseShifts=lambda x: x.nurseHours // SHIFT_LENGTH_IN_HOURS * n_weeks)
nurse_df.nurseCount *= NURSE_DF_MULTIPLIER

# quick check of demand vs supply
print('demand vs supply: ', sum(sum(BASE_DEMAND)), 'vs', sum(nurse_df.nurseHours / SHIFT_LENGTH_IN_HOURS * nurse_df.nurseCount))

# factors
hard_shifts_fair_plans_factor = 0.5
weekend_shifts_fair_plan_factor = 0.5

cost_parameters, feasibility_parameters = calculate_parameters(n_weeks, N_WORK_SHIFTS, nurse_df, BASE_DEMAND,
                                                               hard_shifts_fair_plans_factor,
                                                               weekend_shifts_fair_plan_factor,
                                                               SHIFT_LENGTH_IN_HOURS)

roster_factory = RosterFactory(n_weeks, N_WORK_SHIFTS, nurse_df, cost_parameters, feasibility_parameters)

if read_roster_df:
    roster_df = roster_factory.read_roster_df_from_parquet(roster_df_file_by_n_weeks(n_weeks))
    if n_weeks == 2:
        roster_factory.filter_roster_df(pct_of_best_rosters_to_keep)
        roster_factory.load_roster_matching(roster_matching_file)
else:
    roster_df = roster_factory.calculate_roster_df()
    roster_df.columns = [str(colname) for colname in roster_df.columns]

    # create 2 week roster df with 1week rosters matching
    if n_weeks == 2:
        roster_df = roster_factory.append_one_week_roster_index_to_two_week_roster_df(roster_df_file_by_n_weeks(1))
    roster_df.to_parquet(roster_df_file_by_n_weeks(n_weeks), index=False)

    if n_weeks == 1:
        roster_factory.calculate_roster_matching()
        roster_factory.write_roster_matching(roster_matching_file)
    if n_weeks == 2:
        roster_factory.filter_roster_df(pct_of_best_rosters_to_keep)

# create map of day, work_shifts to rosters
roster_factory.append_day_work_shift_flags()


if use_start_conditions_from_first_two_weeks:

    # create nurse_df from solution file
    roster_solution_df = pd.read_parquet(f'{solution_write_path}StartCondition.parquet')
    nurse_df = roster_solution_df.assign(lastOneWeekRosterIndex=lambda x: x.rosterIndexWeek2,
                                         lastTwoWeekRosterIndex=lambda x: x.rosterIndex).\
        groupby(['nurseHours', 'nurseLevel', 'lastOneWeekRosterIndex', 'lastTwoWeekRosterIndex']).\
        agg(nurseCount=('nRostersInSolution', 'sum')).reset_index().astype({'nurseCount': 'int32'})
    roster_factory.nurse_df = nurse_df

if use_initial_solution:
    n_largest_for_each_nurse = 3  # necessary with 3 to get full 0s, 1s, and 2s plans
    n_smallest_for_each_nurse = 5 ** n_weeks
    roster_factory.run_initial_solution_for_cg(n_largest_for_each_nurse, n_smallest_for_each_nurse)

    roster_factory.run_column_generation(verbose=verbose,
                                         demand=demand,
                                         max_time_sec=max_time_sec,
                                         max_iter=max_iter,
                                         min_object_value=15,
                                         n_rosters_per_iteration=n_rosters_per_iteration,
                                         solver_id='GLOP',
                                         max_time_per_iteration_sec=300)

else:  # full set of rosters solution
    roster_factory.run_full_solution_for_mip()

# run final master problem with MIP
solver, status, demand_c, z = master_problem_instance(n_days=n_days, n_work_shifts=N_WORK_SHIFTS,
                                                      nurse_df=roster_factory.nurse_df,
                                                      roster_indices=roster_factory.roster_indices,
                                                      roster_costs=roster_factory.roster_costs,
                                                      binary_plans=roster_factory.binary_plans,
                                                      demand=demand,
                                                      max_time_solver_sec=1000, solver_id='CBC')

if status == 0:  # optimal solution found
    z = {key: value.solution_value() for key, value in z.items()}
    r_indices_df = pd.DataFrame(list(keys)+[value] for keys, value in z.items() if value >= 1)
    r_indices_df.columns = ['nurseHours', 'nurseLevel', 'lastOneWeekRosterIndex', 'lastTwoWeekRosterIndex', 'rosterIndex', 'nRostersInSolution']

    r_indices_df, roster_factory.roster_df = hotfix_for_pandas_merge(r_indices_df, roster_factory.roster_df)
    roster_solution_df = roster_factory.roster_df.merge(r_indices_df, how='inner', on=['nurseHours', 'rosterIndex'])
    if verbose:
        print(roster_solution_df.loc[:, [str(x) for x in np.arange(n_days)]])
    if use_start_conditions_from_first_two_weeks:
        write_to_parquet(roster_solution_df, solution_write_path)
    else:
        write_to_parquet(roster_solution_df, solution_write_path + 'StartCondition')


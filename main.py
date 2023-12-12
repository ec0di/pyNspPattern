import numpy as np
import pandas as pd
import random

from master_problem import master_problem_instance
from helpers import get_demand, calculate_parameters, calculate_binary_plans
from roster_factory import calculate_roster_df, initial_solution_for_cg
from partial_roster import PartialRoster


n_weeks = 1  # works for 1 week with nurse_type from bla
read_roster_df = True
use_initial_solution = True

n_work_shifts = 3
n_days = n_weeks * 7
base_path = ''
parquet_filename = f'{base_path}data/{n_weeks}WeekRosters.parquet'

# demand per week
base_demand = np.array([[3, 4, 3, 4, 3, 2, 2],
                        [2, 2, 2, 2, 2, 2, 2],
                        [2, 2, 2, 2, 2, 2, 2]])
demand = get_demand(base_demand, n_weeks)

# read in solution
nurse_df_base = pd.read_excel(base_path + 'data/NurseData.xlsx', sheet_name="personindstillinger")
nurse_df = nurse_df_base.groupby(['nurseHours', 'nurseLevel']).agg(nurseCount=('Person', 'count')).reset_index()\
    .rename_axis('nurseType').reset_index()

# factors
hard_shifts_fair_plans_factor = 0.5
weekend_shifts_fair_plan_factor = 0.5

cost_parameters, feasibility_parameters = calculate_parameters(n_weeks, n_work_shifts, nurse_df, base_demand,
                                                               hard_shifts_fair_plans_factor,
                                                               weekend_shifts_fair_plan_factor)

if read_roster_df:
    roster_df = pd.read_parquet(parquet_filename)
else:
    roster_df = calculate_roster_df(nurse_df, n_days, n_work_shifts, cost_parameters, feasibility_parameters)

    # write out solution
    roster_df.columns = [str(colname) for colname in roster_df.columns]  # write df to parquet
    roster_df.to_parquet(parquet_filename, index=False)


n_largest_for_each_nurse = 3  # necessary with 3 to get full 0s, 1s, and 2s plans
n_smallest_for_each_nurse = 5 ** n_weeks

if use_initial_solution:
    roster_indices = initial_solution_for_cg(nurse_df, roster_df, n_largest_for_each_nurse, n_smallest_for_each_nurse)
else:  # full set of rosters solution
    roster_indices = dict()
    for nurse_type in nurse_df.nurseType:
        roster_indices[nurse_type] = set(roster_df[roster_df.nurseType == nurse_type].rosterIndex)

# update roster_df with large cost rosters
roster_df_with_initial_solution = pd.concat([roster_df, roster_largest_cost_df])  # put into class
binary_plans = calculate_binary_plans(n_days, n_work_shifts, roster_df_with_initial_solution)
roster_costs = roster_df_with_initial_solution.set_index('rosterIndex')['totalCost'].to_dict()

# create map of day, work_shifts to rosters
for day in range(n_days):
    work_shift_dict = {f'day{day}_shift0': lambda x: x[f'{day}'] == 0,
                       f'day{day}_shift1': lambda x: x[f'{day}'] == 1,
                       f'day{day}_shift2': lambda x: x[f'{day}'] == 2}
    roster_df = roster_df.assign(**work_shift_dict)

# todo, update day_shift_to_roster_index
#day_shift_to_roster_index = dict()
#for day in range(n_days):
#    for shift in range(n_work_shifts):
#        roster_df_ = roster_df[~roster_df.rosterIndex.isin(set.union(*list(roster_indices.values())))]
#        day_shift_to_roster_index[(day, shift)] = roster_df_[roster_df_[f'day{day}_shift{shift}']].rosterIndex.values

object_value = 99999
max_iter = 10
iter = 0
n_rosters_per_nurse_per_iteration = 5 ** n_weeks

while iter <= max_iter and object_value >= 20 * n_weeks:
    solver, nurse_c, demand_c, demand_comp_level_c, z, status = master_problem_instance(n_days=n_days, n_work_shifts=n_work_shifts,
                                                                                        nurse_df=nurse_df, roster_indices=roster_indices,
                                                                                        roster_costs=roster_costs, binary_plans=binary_plans,
                                                                                        demand=demand,
                                                                                        t_max_sec=300, solver_id='GLOP')

    np.array([const.dual_value() for const in nurse_c.values()])
    np.array([const.dual_value() for const in demand_comp_level_c.values()]).reshape((n_days, n_work_shifts))
    demand_duals = np.array([const.dual_value() for const in demand_c.values()]).reshape((n_days, n_work_shifts))
    print(demand_duals)
    id_max = tuple(np.unravel_index(demand_duals.argmax(), demand_duals.shape))
    print(id_max)

    new_roster_indices = []
    roster_df_ = roster_df[~roster_df.rosterIndex.isin(set.union(*list(roster_indices.values())))]
    for nurse_type in nurse_df.nurseType:
        #lowest_cost_roster_index = np.arange(0, n_rosters_per_nurse_per_iteration)
        df = roster_df_[roster_df_.nurseType==nurse_type]
        df = df[df[f'day{id_max[0]}_shift{id_max[1]}']]
        # random numbers
        n_rosters_left = df.shape[0]
        random_numbers = random.sample(range(0, n_rosters_left), min(n_rosters_left, n_rosters_per_nurse_per_iteration))

        new_roster_indices = df.rosterIndex.values[random_numbers]
        #new_roster_indices = day_shift_to_roster_index[id_max][lowest_cost_roster_index]
        #np.delete(day_shift_to_roster_index[id_max], lowest_cost_roster_index)
        roster_indices[nurse_type] = roster_indices[nurse_type].union(set(new_roster_indices))
    print(len(set.union(*list(roster_indices.values()))), 'rosters in model')

    object_value = solver.Objective().Value()
    iter += 1
    print('------------')
    print(f'Iteration {iter}')
    print('------------')

# sub problem iterations where we find and add rosters

solver, nurse_c, demand_c, demand_comp_level_c, z, status = master_problem_instance(n_days=n_days, n_work_shifts=n_work_shifts,
                                                                                    nurse_df=nurse_df, roster_indices=roster_indices,
                                                                                    roster_costs=roster_costs, binary_plans=binary_plans,
                                                                                    demand=demand,
                                                                                    t_max_sec=300, solver_id='CBC')

if status == 0:
    z_int = {key: value.solution_value() for key, value in z.items()}

    r_indices_df = pd.DataFrame([key[0], key[1], value] for key, value in z_int.items() if value >= 1)
    r_indices_df.columns = ['nurseType', 'rosterIndex', 'nRostersInSolution']

    # nice to remember: .loc[lambda x: x.nurseType == 2]
    roster_solution_df = roster_df_with_initial_solution.merge(r_indices_df, how='inner', on=['rosterIndex', 'nurseType'])
    print(roster_solution_df.loc[:, [str(x) for x in np.arange(n_days)]])


# patch 2 rosters together
cost_parameters, feasibility_parameters = calculate_parameters(2 * n_weeks, n_work_shifts, nurse_df, base_demand,
                                                               hard_shifts_fair_plans_factor,
                                                               weekend_shifts_fair_plan_factor)

base_roster = PartialRoster(n_days=2 * n_days,
                            nurse_type=5,
                            nurse_hours=37,
                            n_work_shifts=n_work_shifts,
                            cost_parameters=cost_parameters,
                            feasibility_parameters=feasibility_parameters)

if False:
    start_time = time.time()

    finished_patched_rosters = []
    for roster1 in finished_rosters:
        for roster2 in finished_rosters:
            new_roster = copy.deepcopy(base_roster)
            for shift in roster1.plan + roster2.plan:
                feasible_shifts = new_roster.feasible_shifts()
                if shift not in feasible_shifts:
                    #print(f'break at {new_roster.plan} and increment with {shift}')
                    break
                new_roster.increment(shift)
            if new_roster.is_finished():
                finished_patched_rosters.append(new_roster)

    print(f'Patching together rosters took: {round(time.time() - start_time, 2)} s')
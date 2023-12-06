import numpy as np
import pandas as pd
import copy

from pyNspPattern.master_problem import model_master
from pyNspPattern.helpers import get_demand, list_to_binary_array, calculate_parameters, read_rosters_from_parquet
from pyNspPattern.partial_roster import PartialRoster

n_weeks = 1  # works for 1 week with nurse_type from bla
n_work_shifts = 3
n_days = n_weeks * 7
base_path = 'pyNspPattern/'
parquet_filename = f'{base_path}data/{n_weeks}WeekRosters.parquet'

# demand per week
base_demand = np.array([[3, 4, 3, 4, 3, 2, 2],
                        [2, 2, 2, 2, 2, 2, 2],
                        [2, 2, 2, 2, 2, 2, 2]])
demand = get_demand(base_demand, n_weeks)

# read in solution
nurse_df_base = pd.read_excel(base_path + 'data/wishes_ag.xlsx', sheet_name="personindstillinger")
nurse_df = nurse_df_base.groupby(['nurseHours', 'nurseLevel']).agg(nurseCount=('Person', 'count')).reset_index().rename_axis('nurseType').reset_index()


# first find out all possible combinations from 4 week days
# then find all combinations of 3 weekend days


#nurse_df = nurse_df_base.groupby(['nurseLevel, 'nurseHours'])['Person'].count().reset_index().rename(
#    columns={'Person': 'count'})
# artificial
#nurse_df.loc[:, 'nurse_count'] = [5, 5, 5, 5, 5, 5]


# factors
hard_shifts_fair_plans_factor = 0.5
weekend_shifts_fair_plan_factor = 0.5

cost_parameters, feasibility_parameters = calculate_parameters(n_weeks, n_work_shifts, nurse_df, base_demand,
                                                               hard_shifts_fair_plans_factor, weekend_shifts_fair_plan_factor)

finished_rosters = []
roster_df = pd.DataFrame()
count = 0
binary_plans = {}

for nurse_type, nurse_hours in nurse_df[['nurseHours']].itertuples():
    base_roster = PartialRoster(n_days=n_days,
                                nurse_type=nurse_type,
                                nurse_hours=nurse_hours,
                                n_work_shifts=n_work_shifts,
                                cost_parameters=cost_parameters,
                                feasibility_parameters=feasibility_parameters)

    n_shifts_for_nurse_type = feasibility_parameters.avg_shifts_per_period[nurse_type]

    rosters = []
    finished_rosters_data = []
    finished_rosters = []
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
                    # todo, add roster as binary roster to finished_rosters list
                    binary_plan = list_to_binary_array(new_roster.plan, n_days, n_work_shifts)
                    binary_plans[count] = binary_plan
                    total_individual_cost, total_fair_cost = sum(individual_cost.values()), sum(fair_cost.values())
                    finished_rosters_data.append(new_roster.plan + list(individual_cost.values()) + list(fair_cost.values())
                                                 + [total_individual_cost, total_fair_cost, total_individual_cost + total_fair_cost,
                                               new_roster.nurse_type, new_roster.nurse_hours])
                    finished_rosters.append(new_roster)
                    count += 1
            else:
                rosters.append(new_roster)

    roster_df_ = pd.DataFrame(finished_rosters_data)
    roster_df_.columns = np.arange(n_days).tolist() + list(individual_cost.keys()) + list(fair_cost.keys()) + \
                         ['totalIndividualCost', 'totalFairCost', 'totalCost', 'nurseType', 'nurseHours']

    # fix work shifts and look at plans!
    roster_df_ = roster_df_.assign(workShifts=lambda x: np.sum(x.loc[:, 0:n_days - 1] < 3, axis=1))\
        .assign(nurseHours=nurse_hours)

    roster_df_ = roster_df_[roster_df_.workShifts >= n_shifts_for_nurse_type - 1]
    #binary_plans.extend(np.array(binary_plans_))
    #binary_plans.extend(np.array(binary_plans_)[roster_df_.index.values])
    #finished_rosters.extend(np.array(finished_rosters_)[roster_df_.index.values])
    print(roster_df_.shape[0])
    roster_df = pd.concat([roster_df, roster_df_])

#roster_df = roster_df.reset_index()
roster_df['rosterIndex'] = np.arange(roster_df.shape[0])
roster_df = roster_df.reset_index()


print(roster_df.shape)


#roster_df.columns = [str(colname) for colname in roster_df.columns]  # write df to parquet
#roster_df.to_parquet(parquet_filename, index=False)


roster_df, binary_plans = read_rosters_from_parquet(parquet_filename, n_days, n_work_shifts)

# run optimization problem
#roster_df = pd.read_csv(csv_filename)


#nurse_df = nurse_df_base.groupby(['nurseHours', 'nurseLevel']).agg(nurseCount=('Person', 'count')).reset_index()

#nurse_df = pd.DataFrame({'nurseHours': [37, 37], 'nurseLevel': [1, 3], 'nurseCount': [6, 12], 'nurseType': [4, 5]})  # works!
#nurse_df = pd.DataFrame({'nurseHours': [37], 'nurseLevel': [3], 'nurseCount': [12], 'nurseType': [4]})  # works!
#solver, nurse_c, demand_c, z, status = model_master(n_days, n_work_shifts, nurse_df, roster_df, binary_plans, demand, t_max_sec=10, solver_id='GLOP')

#top_n_pct = 50
#roster_df_ = roster_df.sort_values('totalCost').iloc[0:round(roster_df.shape[0]/100*top_n_pct)]

n_smallest = 300
roster_df_ = roster_df.merge(roster_df.rename_axis('rosterIndex').groupby('nurseType')['totalCost'].nsmallest(n_smallest).reset_index().drop(columns=['totalCost']),
                             how='inner', on=['nurseType', 'rosterIndex'])

solver, nurse_c, demand_c, demand_comp_level_c, z, status = model_master(n_days, n_work_shifts, nurse_df, roster_df_,
                                                                         binary_plans, demand,
                                                                         t_max_sec=300, solver_id='CBC')
obj_int = solver.Objective().Value()
print(f"Integer Solution with Object {round(obj_int, 1)}")

if status == 0:
    z_int = {key: value.solution_value() for key, value in z.items()}

    r_indices_df = pd.DataFrame([key[0],key[1], value] for key, value in z_int.items() if value >= 1)
    r_indices_df.columns = ['nurseType', 'rosterIndex', 'nRostersInSolution']

    # nice to remember: .loc[lambda x: x.nurseType == 2]
    roster_solution_df = roster_df.merge(r_indices_df, how='inner', on=['rosterIndex', 'nurseType'])
    print(roster_solution_df.loc[:, 0:n_days-1])


# patch 2 rosters together
cost_parameters, feasibility_parameters = calculate_parameters(2 * n_weeks, n_work_shifts, nurse_df, base_demand,
                                                               hard_shifts_fair_plans_factor, weekend_shifts_fair_plan_factor)

base_roster = PartialRoster(n_days=2 * n_days,
                            nurse_type=nurse_type,
                            nurse_hours=nurse_hours,
                            n_work_shifts=n_work_shifts,
                            cost_parameters=cost_parameters,
                            feasibility_parameters=feasibility_parameters)

import time
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
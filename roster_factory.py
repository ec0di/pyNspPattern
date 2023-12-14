import copy
import random
import numpy as np
import pandas as pd
from contexttimer import timer

from helpers import list_to_binary_array
from partial_roster import PartialRoster
from master_problem import master_problem_instance


class RosterFactory:
    def __init__(self, n_weeks, n_work_shifts, nurse_df, cost_parameters, feasibility_parameters):
        self.n_weeks = n_weeks
        self.n_work_shifts = n_work_shifts
        self.nurse_df = nurse_df
        self.cost_parameters = cost_parameters
        self.feasibility_parameters = feasibility_parameters

        self.n_days = n_weeks * 7
        self.roster_df = pd.DataFrame()
        self.roster_indices = dict()
        self.binary_plans = dict()
        self.roster_costs = dict()

        # used primarily to create roster matching
        self.rosters = list()

    @timer()
    def calculate_roster_df(self, add_matching=False):
        nurse_type_minimum_hours = self.nurse_df.sort_values('nurseHours').iloc[0].nurseType
        nurse_type_maximum_hours = self.nurse_df.sort_values('nurseHours').iloc[-1].nurseType
        n_shifts_for_nurse_type_with_minimum_hours = self.feasibility_parameters.avg_shifts_per_period[nurse_type_minimum_hours]
        base_roster = PartialRoster(n_days=self.n_days,
                                    nurse_type=nurse_type_maximum_hours,
                                    n_work_shifts=self.n_work_shifts,
                                    cost_parameters=self.cost_parameters,
                                    feasibility_parameters=self.feasibility_parameters)

        rosters = []
        finished_rosters_data = []
        for shift in range(self.n_work_shifts + 1):
            roster = copy.deepcopy(base_roster)
            roster.increment(shift)
            rosters.append(roster)

        while len(rosters) > 0:
            roster = rosters.pop()
            shifts = set(roster.feasible_shifts())
            for shift in shifts:
                new_roster = copy.deepcopy(roster)
                new_roster.increment(shift)

                off_shifts_needed_later = ((self.n_days - new_roster.day) // 7) * 2  # 2 off shifts per week
                off_shifts_total = new_roster.day - new_roster.work_shifts_total + off_shifts_needed_later
                minimum_allowed_off_shifts = self.n_days - (self.feasibility_parameters.avg_shifts_per_period[nurse_type_minimum_hours] + 1)
                if off_shifts_total > minimum_allowed_off_shifts:
                    continue

                if new_roster.is_finished():
                    if new_roster.work_shifts_total >= n_shifts_for_nurse_type_with_minimum_hours - 1:
                        for nurse_type in self.nurse_df.nurseType:
                            if new_roster.work_shifts_total >= self.feasibility_parameters.avg_shifts_per_period[nurse_type] - 1:
                                new_roster_nurse_type = copy.deepcopy(new_roster)
                                new_roster_nurse_type.nurse_type = nurse_type

                                individual_cost, fair_cost = new_roster_nurse_type.calculate_cost()
                                total_individual_cost, total_fair_cost = sum(individual_cost.values()), sum(fair_cost.values())
                                finished_rosters_data.append(
                                    new_roster_nurse_type.plan + list(individual_cost.values()) + list(fair_cost.values())
                                    + [total_individual_cost, total_fair_cost, total_individual_cost + total_fair_cost,
                                       new_roster_nurse_type.nurse_type,
                                       new_roster_nurse_type.work_days_consecutive])
                else:
                    rosters.append(new_roster)

        roster_df = pd.DataFrame(finished_rosters_data)
        roster_df.columns = np.arange(self.n_days).tolist() + list(individual_cost.keys()) + list(fair_cost.keys()) + \
                             ['totalIndividualCost', 'totalFairCost', 'totalCost', 'nurseType', 'workDaysConsecutive']

        roster_df = roster_df.assign(workShifts=lambda x: np.sum(x.loc[:, 0:self.n_days - 1] < 3, axis=1))

        roster_df = roster_df.sort_values(['nurseType', 'totalCost'])
        roster_df['rosterIndex'] = np.arange(roster_df.shape[0])
        roster_df = roster_df.reset_index()
        print(roster_df.shape)

        self.roster_df = roster_df

        return self.roster_df

    def create_roster_matching(self):
        pass
        # take all unique roster plans

        # last_shift_constraints: concerns last_shift of 1st roster and first_shift of 2nd roster
        # worked_too_much_per_period_constraints: can be calculated easily with work_shifts_total for both rosters (DO OUTSIDE OF LOOP with roster_df)
        # worked_too_many_day_consecutive_constraints: most time consuming constraint to check. Have to go 5 days into second roster to check that this holds. Use work_days_consecutive from 1st roster

        # dump to pickle

    def read_roster_df_from_parquet(self, parquet_filename):
        self.roster_df = pd.read_parquet(parquet_filename)
        return self.roster_df

    def initial_solution_for_cg(self, n_largest_for_each_nurse, n_smallest_for_each_nurse):
        """This initial solution contains expensive 0s ,1s and 2s plans for all nurse types along with a set of the
        cheapest plans for all nurse types"""
        roster_largest_cost_df = self.roster_df.merge(self.roster_df.rename_axis('rosterIndex').groupby('nurseType')['totalCost']
                                                 .nlargest(n_largest_for_each_nurse).reset_index().drop(
            columns=['totalCost']),
                                                 how='inner', on=['nurseType', 'rosterIndex'])
        largest_cost_array = np.tile(np.concatenate([0 * np.ones((1, self.n_days)),
                                                     1 * np.ones((1, self.n_days)),
                                                     2 * np.ones((1, self.n_days))]),
                                     (self.nurse_df.nurseType.nunique(), 1)).astype(int)
        roster_largest_cost_df.loc[:, [str(x) for x in range(self.n_days)]] = largest_cost_array
        roster_largest_cost_df['rosterIndex'] = np.arange(self.roster_df.shape[0],
                                                          self.roster_df.shape[0] + roster_largest_cost_df.shape[0])
        roster_largest_cost_df['totalCost'] = 9999
        roster_smallest_cost_df = self.roster_df.merge(
            self.roster_df.rename_axis('rosterIndex').groupby('nurseType')['totalCost']
            .nsmallest(n_smallest_for_each_nurse).reset_index().drop(columns=['totalCost']),
            how='inner', on=['nurseType', 'rosterIndex'])
        roster_indices = dict()
        for nurse_type in self.nurse_df.nurseType:
            small_cost_set = set(roster_smallest_cost_df[roster_smallest_cost_df.nurseType == nurse_type].rosterIndex)
            large_cost_set = set(roster_largest_cost_df[roster_largest_cost_df.nurseType == nurse_type].rosterIndex)
            roster_indices[nurse_type] = small_cost_set.union(large_cost_set)

        print('initial expensive solution is added to roster_df (0s, 1s, 2s)')
        self.roster_df = pd.concat([self.roster_df, roster_largest_cost_df])  # put into class
        self.roster_indices = roster_indices
        self.binary_plans = self.calculate_binary_plans()
        self.roster_costs = self.roster_df.set_index('rosterIndex')['totalCost'].to_dict()
        print('binary_plans and roster_cost is added to class')
        return self.roster_indices, self.binary_plans, self.roster_costs

    def full_solution_for_mip(self):
        self.roster_indices = {nurse_type: set(self.roster_df[self.roster_df.nurseType == nurse_type].rosterIndex.values)
                               for nurse_type in self.nurse_df.nurseType.values}
        self.binary_plans = self.calculate_binary_plans()
        self.roster_costs = self.roster_df.set_index('rosterIndex')['totalCost'].to_dict()
        return self.roster_indices, self.binary_plans, self.roster_costs

    @timer()
    def calculate_binary_plans(self):
        binary_plans = {}
        for plan in self.roster_df.loc[:, [str(x) for x in np.arange(self.n_days)] + ['rosterIndex']].itertuples(index=False):
            binary_plans[plan.rosterIndex] = list_to_binary_array(plan[:-1], self.n_days, self.n_work_shifts)
        return binary_plans

    def append_day_work_shift_flags(self):
        for day in range(self.n_days):
            work_shift_dict = {f'day{day}_shift0': lambda x: x[f'{day}'] == 0,
                               f'day{day}_shift1': lambda x: x[f'{day}'] == 1,
                               f'day{day}_shift2': lambda x: x[f'{day}'] == 2}
            self.roster_df = self.roster_df.assign(**work_shift_dict)
        return self.roster_df

    @timer()
    def run_column_generation(self, demand, solver_id='GLOP', max_iter=10, min_object_value=20,
                              max_time_per_iteration_s=10, n_rosters_per_nurse_per_iteration=10, verbose=False):
        """This function runs column generation for a given demand. It starts with a full set of rosters and iteratively
        adds more. Currently, it adds a number of random rosters for each nurse type based on the dual values of the demand
        constraints."""

        object_value = 99999
        iter = 0
        while iter <= max_iter and min_object_value * self.n_weeks <= object_value:
            solver, nurse_c, demand_c, demand_comp_level_c, z, status = master_problem_instance(n_days=self.n_days,
                                                                                                n_work_shifts=self.n_work_shifts,
                                                                                                nurse_df=self.nurse_df,
                                                                                                roster_indices=self.roster_indices,
                                                                                                roster_costs=self.roster_costs,
                                                                                                binary_plans=self.binary_plans,
                                                                                                demand=demand,
                                                                                                t_max_sec=max_time_per_iteration_s,
                                                                                                solver_id=solver_id)

            # np.array([const.dual_value() for const in nurse_c.values()])
            # np.array([const.dual_value() for const in demand_comp_level_c.values()]).reshape((n_days, n_work_shifts))
            demand_duals = np.array([const.dual_value() for const in demand_c.values()]).reshape(
                (self.n_days, self.n_work_shifts))
            id_max = tuple(np.unravel_index(demand_duals.argmax(), demand_duals.shape))

            roster_df_ = self.roster_df[~self.roster_df.rosterIndex.isin(set.union(*list(self.roster_indices.values())))]
            for nurse_type in self.nurse_df.nurseType:
                # lowest_cost_roster_index = np.arange(0, n_rosters_per_nurse_per_iteration)
                df = roster_df_[roster_df_.nurseType == nurse_type]
                df = df[df[f'day{id_max[0]}_shift{id_max[1]}']]
                # random numbers
                n_rosters_left = df.shape[0]
                random_numbers = random.sample(range(0, n_rosters_left),
                                               min(n_rosters_left, n_rosters_per_nurse_per_iteration))

                new_roster_indices = df.rosterIndex.values[random_numbers]
                self.roster_indices[nurse_type] = self.roster_indices[nurse_type].union(set(new_roster_indices))

            object_value = solver.Objective().Value()
            iter += 1

            if verbose:
                print(demand_duals)
                print(id_max)
                print(len(set.union(*list(self.roster_indices.values()))), 'rosters in model')
                print('------------')
                print(f'Iteration {iter}')
                print('------------')

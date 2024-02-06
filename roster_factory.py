import copy
import json
import random
import numpy as np
import pandas as pd
from contexttimer import timer
from time import time

from helpers import list_to_binary_array, downcast_dataframe
from input_parameters import OFF_SHIFT, MAX_CONSECUTIVE_WORK_SHIFTS, DELTA_NURSE_SHIFT, SHIFT_LENGTH_IN_HOURS
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
        self.roster_matching = dict()

    @timer()
    def calculate_roster_df(self):
        delta_nurse_shift = 4 if self.n_weeks == 1 else DELTA_NURSE_SHIFT  # for 1-week rosters it is okay to create lots of rosters

        nurse_hours_list = self.nurse_df.nurseHours.unique()
        min_nurse_hours = nurse_hours_list.min()
        max_nurse_hours = nurse_hours_list.max()
        n_shifts_for_min_nurse_hours = self.feasibility_parameters.avg_shifts_per_period[min_nurse_hours]
        base_roster = PartialRoster(n_days=self.n_days,
                                    nurse_hours=max_nurse_hours,
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
                minimum_allowed_off_shifts = self.n_days - (
                            self.feasibility_parameters.avg_shifts_per_period[min_nurse_hours] - delta_nurse_shift)
                if off_shifts_total > minimum_allowed_off_shifts:
                    continue

                if new_roster.is_finished():
                    if new_roster.work_shifts_total >= n_shifts_for_min_nurse_hours - delta_nurse_shift:
                        for nurse_hours in nurse_hours_list:
                            if new_roster.work_shifts_total >= self.feasibility_parameters.avg_shifts_per_period[
                                nurse_hours] - delta_nurse_shift:
                                new_roster_nurse_hours = copy.deepcopy(new_roster)
                                new_roster_nurse_hours.nurse_hours = nurse_hours

                                individual_cost, fair_cost = new_roster_nurse_hours.calculate_cost()
                                total_individual_cost, total_fair_cost = sum(individual_cost.values()), sum(
                                    fair_cost.values())
                                finished_rosters_data.append(
                                    new_roster_nurse_hours.plan + list(individual_cost.values()) + list(
                                        fair_cost.values())
                                    + [total_individual_cost, total_fair_cost, total_individual_cost + total_fair_cost,
                                       new_roster_nurse_hours.work_days_consecutive,
                                       new_roster_nurse_hours.work_shifts_total,
                                       nurse_hours])
                else:
                    rosters.append(new_roster)

        roster_df = pd.DataFrame(finished_rosters_data)
        roster_df.columns = np.arange(self.n_days).tolist() + list(individual_cost.keys()) + list(fair_cost.keys()) + \
                            ['totalIndividualCost', 'totalFairCost', 'totalCost',
                             'workDaysConsecutive', 'workShifts', 'nurseHours']

        roster_df = roster_df.sort_values(['nurseHours', 'totalCost'])
        roster_df['rosterIndex'] = np.arange(roster_df.shape[0])
        roster_df = roster_df.reset_index(drop=True)

        roster_df = downcast_dataframe(roster_df)
        self.roster_df = roster_df

        return roster_df

    @timer()
    def calculate_roster_matching(self):
        avg_shifts_per_two_weeks = {nurse_hours: nurse_hours / SHIFT_LENGTH_IN_HOURS * 2
                                    for nurse_hours in self.nurse_df.nurseHours.unique()}

        roster_matching = {rosterIndex: {'rostersAllowedAfter': [], 'rostersAllowedBefore': []}
                           for rosterIndex in self.roster_df.rosterIndex}
        for nurse_hours in self.nurse_df.nurseHours.unique():
            roster_df_ = self.roster_df[self.roster_df.nurseHours == nurse_hours]
            for roster_1 in roster_df_.itertuples(index=False):
                for roster_2 in roster_df_.itertuples(index=False):
                    # last_shift_constraints:
                    last_shift_roster_1 = roster_1[self.n_days - 1]
                    first_shift_roster_2 = roster_2[0]
                    if (last_shift_roster_1 == 1 and first_shift_roster_2 == 2) or \
                            (last_shift_roster_1 == 2 and first_shift_roster_2 in [0, 1]):
                        continue
                    # worked_too_much_per_period_constraints:
                    if roster_1.workShifts + roster_2.workShifts > avg_shifts_per_two_weeks[nurse_hours] + DELTA_NURSE_SHIFT:
                        continue
                    # worked_too_many_day_consecutive_constraints:
                    try:
                        work_days_consecutive_start_2 = np.where(np.array(roster_2[0:MAX_CONSECUTIVE_WORK_SHIFTS])==OFF_SHIFT)[0][0]
                    except:
                        work_days_consecutive_start_2 = MAX_CONSECUTIVE_WORK_SHIFTS
                    if roster_1.workDaysConsecutive + work_days_consecutive_start_2 > MAX_CONSECUTIVE_WORK_SHIFTS:
                        continue
                    roster_matching[roster_1.rosterIndex]['rostersAllowedAfter'].append(roster_2.rosterIndex)
                    roster_matching[roster_2.rosterIndex]['rostersAllowedBefore'].append(roster_1.rosterIndex)
        self.roster_matching = roster_matching

    def read_roster_df_from_parquet(self, parquet_filename, is_returned_only=False):
        print(f'Loading file: {parquet_filename}')
        df = pd.read_parquet(parquet_filename)
        df = downcast_dataframe(df)
        if not is_returned_only:
            self.roster_df = df
        return df

    def run_initial_solution_for_cg(self, n_largest_for_each_nurse, n_smallest_for_each_nurse):
        """This initial solution contains expensive 0s ,1s and 2s plans for all nurse types along with a set of the
        cheapest plans for all nurse types"""
        roster_largest_cost_df = self.roster_df.merge(
            self.roster_df.rename_axis('rosterIndex').groupby('nurseHours')['totalCost']
            .nlargest(n_largest_for_each_nurse).reset_index().drop(columns=['totalCost']),
            how='inner', on=['nurseHours', 'rosterIndex'])
        largest_cost_array = np.tile(np.concatenate([0 * np.ones((1, self.n_days)),
                                                     1 * np.ones((1, self.n_days)),
                                                     2 * np.ones((1, self.n_days))]),
                                     (self.nurse_df.nurseHours.nunique(), 1)).astype(int)
        roster_largest_cost_df.loc[:, [str(x) for x in range(self.n_days)]] = largest_cost_array
        roster_largest_cost_df['rosterIndex'] = np.arange(self.roster_df.shape[0],
                                                          self.roster_df.shape[0] + roster_largest_cost_df.shape[0])
        roster_largest_cost_df['totalCost'] = 9999
        # roster_smallest_cost_df = self.roster_df.merge(
        #    self.roster_df.rename_axis('rosterIndex').groupby(['nurseHours'])['totalCost']
        #    .nsmallest(n_smallest_for_each_nurse).reset_index().drop(columns=['totalCost']),
        #    how='inner', on=['nurseHours', 'rosterIndex'])
        roster_smallest_cost_df = self.roster_df.sort_values(['nurseHours', 'totalCost'])
        roster_indices = dict()
        for nurse_hours, last_one_week_roster_index in self.nurse_df.groupby(
                ['nurseHours', 'lastOneWeekRosterIndex']).groups.keys():
            small_cost_df = roster_smallest_cost_df[roster_smallest_cost_df.nurseHours == nurse_hours]
            if self.roster_matching and last_one_week_roster_index != -1:
                small_cost_df = small_cost_df[small_cost_df.rosterIndexWeek1.isin(
                    self.roster_matching[last_one_week_roster_index]['rostersAllowedAfter'])]
            small_cost_set = set(small_cost_df[:min(n_smallest_for_each_nurse, small_cost_df.shape[0])].rosterIndex)
            large_cost_set = set(roster_largest_cost_df[roster_largest_cost_df.nurseHours == nurse_hours].rosterIndex)
            roster_indices[nurse_hours, last_one_week_roster_index] = small_cost_set.union(large_cost_set)

        print('Expensive illegal Initial solution plans are added to roster_df (full 0s, 1s, 2s) for Column Generation')
        self.roster_df = pd.concat([self.roster_df, roster_largest_cost_df])  # put into class
        self.roster_indices = roster_indices
        self.binary_plans = self.calculate_binary_plans()
        self.roster_costs = self.roster_df.set_index('rosterIndex')['totalCost'].to_dict()
        print('binary_plans and roster_cost is added to roster_factory')

    def run_full_solution_for_mip(self):
        for nurse_hours, last_one_week_roster_index in self.nurse_df[['nurseHours', 'lastOneWeekRosterIndex']].itertuples(index=False):
            df = self.roster_df[self.roster_df.nurseHours == nurse_hours]
            if self.roster_matching and last_one_week_roster_index != -1:
                df = df[df.rosterIndexWeek1.isin(
                    self.roster_matching[last_one_week_roster_index]['rostersAllowedAfter'])]
            self.roster_indices[nurse_hours, last_one_week_roster_index] = set(df.rosterIndex.values)
        self.binary_plans = self.calculate_binary_plans()
        self.roster_costs = self.roster_df.set_index('rosterIndex')['totalCost'].to_dict()
        #return self.roster_indices, self.binary_plans, self.roster_costs

    @timer()
    def calculate_binary_plans(self):
        binary_plans = {}
        for plan in self.roster_df.loc[:, [str(x) for x in np.arange(self.n_days)] + ['rosterIndex']].itertuples(
                index=False):
            binary_plans[plan.rosterIndex] = list_to_binary_array(plan[:-1], self.n_days, self.n_work_shifts)
        return binary_plans

    def append_day_work_shift_flags(self):
        for day in range(self.n_days):
            work_shift_dict = {f'day{day}_shift0': lambda x: x[f'{day}'] == 0,
                               f'day{day}_shift1': lambda x: x[f'{day}'] == 1,
                               f'day{day}_shift2': lambda x: x[f'{day}'] == 2}
            self.roster_df = self.roster_df.assign(**work_shift_dict)
        return self.roster_df

    def append_one_week_roster_index_to_two_week_roster_df(self, parquet_filename_for_roster1_df):
        roster1_df = self.read_roster_df_from_parquet(parquet_filename=parquet_filename_for_roster1_df,
                                                      is_returned_only=True)
        roster2_df = self.roster_df
        first_week_cols = [str(x) for x in np.arange(7)]
        second_week_cols = [str(x) for x in np.arange(7, 14)]
        roster2_with_roster1_index_df = roster2_df.merge(roster1_df.rename(columns={'rosterIndex': 'rosterIndexWeek1'})[first_week_cols+['nurseHours', 'rosterIndexWeek1']],
                         how='inner', on=['nurseHours']+first_week_cols)\
                  .merge(roster1_df.rename(columns={'rosterIndex': 'rosterIndexWeek2', **{col: str(int(col)+7) for col in first_week_cols}})
                              [second_week_cols+['nurseHours', 'rosterIndexWeek2']],
                         how='inner', on=['nurseHours']+second_week_cols)
        print('Appending one week roster index to two week roster_df. Shape before and after: ', roster2_df.shape, roster2_with_roster1_index_df.shape)
        self.roster_df = roster2_with_roster1_index_df
        return roster2_with_roster1_index_df

    @timer()
    def run_column_generation(self, demand, solver_id='GLOP', max_time_sec=30, max_iter=10, min_object_value=20,
                              max_time_per_iteration_sec=10, n_rosters_per_iteration=10, verbose=True):
        """This function runs column generation for a given demand. It starts with a full set of rosters and iteratively
        adds more. Currently, it adds a number of random rosters for each nurse type based on the dual values of the demand
        constraints."""
        nurse_df_groups = self.nurse_df.groupby(['nurseHours', 'lastOneWeekRosterIndex']).groups

        n_rosters_per_nurse_per_iteration = n_rosters_per_iteration // len(nurse_df_groups)

        start_time = time()
        object_value = 99999
        iter = 1
        while iter <= max_iter and time() - start_time < max_time_sec and min_object_value * self.n_weeks <= object_value:
            solver, status, demand_c, z = master_problem_instance(n_weeks=self.n_weeks,
                                                                  n_work_shifts=self.n_work_shifts,
                                                                  nurse_df=self.nurse_df,
                                                                  roster_indices=self.roster_indices,
                                                                  roster_costs=self.roster_costs,
                                                                  binary_plans=self.binary_plans,
                                                                  demand=demand,
                                                                  max_time_solver_sec=max_time_per_iteration_sec,
                                                                  solver_id=solver_id,
                                                                  verbose=verbose)

            # Note currently only using demand_dual values, since this is enough, and we create plans for all nurse types
            demand_duals = np.array([const.dual_value() for const in demand_c.values()]).reshape(
                (self.n_days, self.n_work_shifts))

            id_max = tuple(np.unravel_index(demand_duals.argmax(), demand_duals.shape))

            roster_df_ = self.roster_df[
                ~self.roster_df.rosterIndex.isin(set.union(*list(self.roster_indices.values())))]
            for nurse_hours, last_one_week_roster_index in nurse_df_groups.keys():
                df = roster_df_[roster_df_.nurseHours == nurse_hours]
                df = df[df[f'day{id_max[0]}_shift{id_max[1]}']]
                if self.roster_matching and last_one_week_roster_index != -1:
                    df = df[df.rosterIndexWeek1.isin(
                        self.roster_matching[last_one_week_roster_index]['rostersAllowedAfter'])]
                # use random numbers to pick random plans within the plans that will increase worst demand dual values
                n_rosters_left = df.shape[0]
                random_numbers = random.sample(range(0, n_rosters_left),
                                               min(n_rosters_left, n_rosters_per_nurse_per_iteration))

                new_roster_indices = df.rosterIndex.values[random_numbers]
                self.roster_indices[nurse_hours, last_one_week_roster_index] = self.roster_indices[
                    nurse_hours, last_one_week_roster_index].union(set(new_roster_indices))

            object_value = solver.Objective().Value()
            if verbose:
                print(demand_duals)
                print(id_max)
                print(len(set.union(*list(self.roster_indices.values()))), 'rosters in model')
                print('------------')
                print(f'Iteration {iter}')
                print('------------')
            iter += 1

    def load_roster_matching(self, roster_matching_file):
        """deserialize roster matching"""
        print(f'Loading file: {roster_matching_file}')
        with open(roster_matching_file, 'r') as fp:
            roster_matching = json.load(fp)
            self.roster_matching = {int(key): value for key, value in roster_matching.items()}

    def write_roster_matching(self, roster_matching_file):
        # serialize roster matching
        print(f'Exporting file {roster_matching_file}')
        with open(roster_matching_file, 'w') as fp:
            json.dump(self.roster_matching, fp)

    def filter_roster_df(self, pct_of_best_rosters_to_keep):
        """This function filters the roster_df to keep only the best rosters for each nurse type"""
        roster_df = self.roster_df
        before = roster_df.shape[0]

        def top_n_percent(group):
            return group.nsmallest(int(len(group) * pct_of_best_rosters_to_keep), 'totalCost')

        roster_df = roster_df.groupby('nurseHours', group_keys=False).apply(top_n_percent)
        print(f'{roster_df.shape[0]} of {before} rows left after filtering {pct_of_best_rosters_to_keep*100} pct best rosters for each nurse type')
        self.roster_df = roster_df
        return roster_df

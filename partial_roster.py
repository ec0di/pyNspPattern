import numpy as np
import copy

from pyNspPattern.helpers import is_weekend, COSTS as cost_elements


class PartialRoster(object):
    def __init__(self, n_days: int, nurse_type: int, nurse_hours: int, n_work_shifts: int,
                 cost_parameters: object, feasibility_parameters: object):
        self.n_days = n_days
        self.nurse_type = nurse_type
        self.nurse_hours = nurse_hours
        self.n_work_shifts = n_work_shifts
        self.off_shift = n_work_shifts
        self.day = 0  # [0, ..., n_days-1]
        self.plan = []
        self.work_shifts_total = 0
        self.last_shift = None  # is None or integer
        self.shift_before_last_shift = None
        self.work_days_consecutive = 0
        self.days_off_this_week = 0
        self.shifts_this_week = [False, False, False]
        self.is_work_weekend = None
        self.cost_parameters = cost_parameters
        self.feasibility_parameters = feasibility_parameters
        # mode info
        self.night_shift_consecutive = 0
        self.night_shift_total = 0
        self.afternoon_shift_total = 0
        self.weekend_shift_total = 0
        self.off_shifts_consecutive = 0

        # cost counters
        self.n_consecutive_shifts = 0
        self.n_missing_two_days_off_after_night_shifts = 0
        self.n_more_than_two_concecutive_night_shifts = 0
        self.n_single_night_shifts = 0
        self.n_more_than_four_consecutive_work_shifts = 0
        #
        self.cost = 0
        self.fair_cost = 0
        self.dual_cost = 0
        # fair cost
        self.afternoon_shifts_fair_cost = 0
        self.night_shifts_fair_cost = 0
        self.night_and_afternoon_shifts_fair_cost = 0
        self.weekend_shifts_fair_cost = 0

    def increment(self, shift_type: int):
        k = shift_type
        if k != self.off_shift:
            self.off_shifts_consecutive = 0
            self.work_shifts_total += 1
            self.work_days_consecutive += 1
            if k == 0:
                self.shifts_this_week[0] = True
                self.night_shift_consecutive = 0
            elif k == 1:
                self.shifts_this_week[1] = True
                self.night_shift_consecutive = 0
                self.afternoon_shift_total += 1
            else:
                self.shifts_this_week[2] = True
                self.night_shift_consecutive += 1
                self.night_shift_total += 1
            if is_weekend(self.day, k):
                self.weekend_shift_total += 1
        else:
            self.night_shift_consecutive = 0
            self.work_days_consecutive = 0
            self.days_off_this_week += 1
            self.off_shifts_consecutive += 1

        # update cost counters
        if self.last_shift == k and k != self.off_shift:
            self.n_consecutive_shifts += 1
        if self.shift_before_last_shift == 2 and self.last_shift != 2 and not self.off_shifts_consecutive == 2:
            self.n_missing_two_days_off_after_night_shifts += 1
        if k == 2 and self.night_shift_consecutive > 2:
            self.n_more_than_two_concecutive_night_shifts += 1
        if self.shift_before_last_shift != 2 and self.last_shift == 2 and k != 2:
            self.n_single_night_shifts += 1
        if self.work_days_consecutive > 4:
            self.n_more_than_four_consecutive_work_shifts += 1

        # update parameters
        self.plan.append(k)
        self.shift_before_last_shift = self.last_shift
        self.last_shift = k
        self.day += 1  # self.day + 1 if self.day is not None else 0
        day_of_week = np.mod(self.day, 7)
        if day_of_week == 0:  # first day of week
            self.shifts_this_week = [False, False, False]
            self.days_off_this_week = 0

    def feasible_shifts(self):
        day_of_week = np.mod(self.day, 7)
        a = np.arange(self.n_work_shifts + 1)

        if self.day == 0:  # base case, we have not started yet
            return a

        if self.work_shifts_total > self.feasibility_parameters.avg_shifts_per_period[self.nurse_type] + 1 \
                or self.work_days_consecutive == 5:
            a = a[a == self.off_shift]
        if (self.days_off_this_week == 0 and day_of_week == 5) or (self.days_off_this_week == 1 and day_of_week == 6):
            a = a[a == self.off_shift]

        # we now give work shifts
        a = self.only_two_shift_types_per_week(a)

        if self.last_shift == 1:
            a = a[(a != 0)]
        elif self.last_shift == 2:
            a = a[(a == 2) | (a == 3)]

        if day_of_week == 5:
            if self.last_shift in (1, 2):
                a = a[a != self.off_shift]
            if self.last_shift == 0:
                a = a[(a == 0) | (a == 3)]
            if self.last_shift == 3:
                a = a[a == self.off_shift]
        if day_of_week == 6:  # we can work and if we work, we must work all weekend
            if self.last_shift == self.off_shift:
                a = a[a == self.off_shift]
            else:
                a = a[a != self.off_shift]
                self.is_work_weekend = True

        return a

    def only_two_shift_types_per_week(self, a):
        D, A, N = self.shifts_this_week
        if (D + A + N) == 2:
            if not D:
                a = a[(a != 0)]
            elif not A:
                a = a[(a != 1)]
            else:
                a = a[(a != 2)]
        return a

    def is_finished(self):
        return self.day == self.n_days

    def calc_fair_costs(self, current_or_next='current'):

        start_value = 0 if current_or_next == 'current' else 1  # then its == 'next'
        cp = self.cost_parameters
        hard_shift_fair = cp.hard_shift_fair_per_period[self.nurse_type]

        ce = copy.deepcopy(cost_elements)
        ce['afternoonShiftsFair'] = cp.hard_shifts_fair_plans_factor * \
                                      max(start_value, start_value + self.afternoon_shift_total - hard_shift_fair) \
                                      - self.afternoon_shifts_fair_cost
        ce['nightShiftsFair'] = cp.hard_shifts_fair_plans_factor * max(start_value,
                                                                         start_value + self.night_shift_total -
                                                                         hard_shift_fair) - self.night_shifts_fair_cost
        ce['nightAndAfternoonShiftsFair'] = cp.hard_shifts_fair_plans_factor * \
                                                max(start_value, start_value + self.afternoon_shift_total
                                                    + self.night_shift_total - 2 * hard_shift_fair) \
                                                - self.night_and_afternoon_shifts_fair_cost
        ce['weekendShiftsFair'] = cp.weekend_shifts_fair_plan_factor * \
                                    max(start_value, start_value + self.weekend_shift_total -
                                        cp.avg_weekend_shifts_per_person_per_period) - self.weekend_shifts_fair_cost
        return ce

    def cost_next_shift(self, feasible_shifts):
        cp = self.cost_parameters
        ce = self.calc_fair_costs(current_or_next='next')

        hard_shift_fair = cp.hard_shift_fair_per_period[self.nurse_type]

        costs = []
        for k in feasible_shifts:
            cost = {'fair_cost': 0, 'individual_cost': 0}
            if self.last_shift == k and k != self.off_shift:
                cost['individual_cost'] += ce['consecutiveShifts']
                cp.count_cost_cases['consecutiveShifts'] += 1
            if self.shift_before_last_shift == 2 and self.last_shift != self.off_shift and k != self.off_shift:
                cost['individual_cost'] += ce['missingTwoDaysOffAfterNightShifts']
                cp.count_cost_cases['missingTwoDaysOffAfterNightShifts'] += 1
            if k == 2 and self.night_shift_consecutive >= 2:
                cost['individual_cost'] += ce['moreThanTwoConsecutiveNightShifts']
                cp.count_cost_cases['moreThanTwoConsecutiveNightShifts'] += 1
            if self.night_shift_consecutive == 1 and k != 2:
                cost['individual_cost'] += ce['singleNightShift']
                cp.count_cost_cases['singleNightShift'] += 1
            if self.work_days_consecutive >= 4 and k != self.off_shift:
                cost['individual_cost'] += ce['moreThanFourConsecutiveWorkShifts']
                cp.count_cost_cases['moreThanFourConsecutiveWorkShifts'] += 1

            # penalty related to fair plans
            if k == 1 and self.afternoon_shift_total + 1 > hard_shift_fair:
                cost['fair_cost'] += ce['afternoonShiftsFair']
                cp.count_cost_cases['afternoonShiftsFair'] += 1
            if k == 2 and self.night_shift_total + 1 > hard_shift_fair:
                cost['fair_cost'] += ce['nightShiftsFair']
                cp.count_cost_cases['nightShiftsFair'] += 1
            if (k in [1, 2]) and self.night_shift_total + self.afternoon_shift_total + 1 > 2 * hard_shift_fair:
                cost['fair_cost'] += ce['nightAndAfternoonShiftsFair']
                cp.count_cost_cases['nightAndAfternoonShiftsFair'] += 1
            if is_weekend(self.day, k) and self.weekend_shift_total + 1 > cp.avg_weekend_shifts_per_person_per_period:
                cost['fair_cost'] += ce['weekendShiftsFair']
                cp.count_cost_cases['weekendShiftsFair'] += 1
            costs.append(cost)
        return costs

    def calculate_fair_costs(self):

        cp = self.cost_parameters
        hard_shift_fair = cp.hard_shift_fair_per_period[self.nurse_type]

        costs = {'afternoonShiftsFair': cp.hard_shifts_fair_plans_factor * \
                                          max(0, self.afternoon_shift_total - hard_shift_fair) \
                                          - self.afternoon_shifts_fair_cost,
                 'nightShiftsFair': cp.hard_shifts_fair_plans_factor * max(0, self.night_shift_total -
                                                                             hard_shift_fair) - self.night_shifts_fair_cost,
                 'nightAndAfternoonShiftsFair': cp.hard_shifts_fair_plans_factor * \
                                                    max(0, self.afternoon_shift_total
                                                        + self.night_shift_total - 2 * hard_shift_fair) \
                                                    - self.night_and_afternoon_shifts_fair_cost,
                 'weekendShiftsFair': cp.weekend_shifts_fair_plan_factor * \
                                        max(0, self.weekend_shift_total -
                                            cp.avg_weekend_shifts_per_person_per_period) - self.weekend_shifts_fair_cost}
        return costs

    def calculate_cost(self):
        if not self.is_finished():
            print('Roster is not finished')
            return
        fair_costs = self.calculate_fair_costs()

        individual_costs = {
            'consecutiveShifts': cost_elements['consecutiveShifts'] * self.n_consecutive_shifts,
            'missingTwoDaysOffAfterNightShifts': cost_elements['missingTwoDaysOffAfterNightShifts'] * self.n_missing_two_days_off_after_night_shifts,
            'moreThanTwoConsecutiveNightShifts': cost_elements['moreThanTwoConsecutiveNightShifts'] * self.n_more_than_two_concecutive_night_shifts,
            'singleNightShift': cost_elements['singleNightShift'] * self.n_single_night_shifts,
            'moreThanFourConsecutiveWorkShifts': cost_elements['moreThanFourConsecutiveWorkShifts'] * self.n_more_than_four_consecutive_work_shifts,
        }
        return individual_costs, fair_costs

    def __str__(self):
        if self.day == self.n_days:
            s = f'Finished Roster with plan: {self.plan} for nurse type {self.nurse_type} and work weekend ' \
                f'{self.is_work_weekend}'
        else:
            s = f'Unfinished Roster with plan: {self.plan} and\n' \
                f"day: {self.day}\n" \
                f"work_shift_total:{self.work_shifts_total}\n" \
                f"last_shift:{self.last_shift}\n" \
                f"shift_before_last_shift:{self.shift_before_last_shift}\n" \
                f"work_days_consecutive:{self.work_days_consecutive}\n" \
                f"days_off_this_week:{self.days_off_this_week}\n" \
                f"shifts_this_week:{self.shifts_this_week}"
        return s

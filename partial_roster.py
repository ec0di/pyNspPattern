import numpy as np
import copy

from helpers import is_weekend, COSTS as cost_elements, FeasibilityParameters, CostParameters, OFF_SHIFT


class PartialRoster(object):
    def __init__(self, n_days: int, nurse_hours: int, n_work_shifts: int,
                 cost_parameters: CostParameters, feasibility_parameters: FeasibilityParameters):
        self.n_days = n_days
        self.nurse_hours = nurse_hours
        self.n_work_shifts = n_work_shifts
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

        # individual roster cost
        self.n_consecutive_shifts = 0
        self.n_missing_two_days_off_after_night_shifts = 0
        self.n_more_than_two_concecutive_night_shifts = 0
        self.n_single_night_shifts = 0
        self.n_more_than_four_consecutive_work_shifts = 0
        # fair distribution of hard shifts cost
        self.afternoon_shifts_fair_cost = 0
        self.night_shifts_fair_cost = 0
        self.night_and_afternoon_shifts_fair_cost = 0
        self.weekend_shifts_fair_cost = 0

    def increment(self, shift_type: int):
        k = shift_type
        if k != OFF_SHIFT:
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
        if self.last_shift == k and k != OFF_SHIFT:
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
        allowed_shifts = np.arange(self.n_work_shifts + 1)

        if self.day == 0:  # base case, we have no constraints yet
            return allowed_shifts

        allowed_shifts = self.worked_too_much_per_period_constraints(allowed_shifts)

        allowed_shifts = self.worked_too_many_day_consecutive_constraints(allowed_shifts)

        allowed_shifts = self.last_shift_constraints(allowed_shifts)

        allowed_shifts = self.worked_too_much_per_week_constraints(allowed_shifts, day_of_week)

        allowed_shifts = self.only_two_shift_types_per_week_constraints(allowed_shifts)

        allowed_shifts = self.weekend_per_week_constraints(allowed_shifts, day_of_week)

        return allowed_shifts

    def worked_too_much_per_period_constraints(self, allowed_shifts):
        if self.work_shifts_total > self.feasibility_parameters.avg_shifts_per_period[self.nurse_hours] + 1:
            allowed_shifts = allowed_shifts[allowed_shifts == OFF_SHIFT]
        return allowed_shifts

    def worked_too_many_day_consecutive_constraints(self, allowed_shifts):
        if self.work_days_consecutive == 5:
            allowed_shifts = allowed_shifts[allowed_shifts == OFF_SHIFT]
        return allowed_shifts

    def worked_too_much_per_week_constraints(self, allowed_shifts, day_of_week):
        if (self.days_off_this_week == 0 and day_of_week == 5) \
                or (self.days_off_this_week == 1 and day_of_week == 6):
            allowed_shifts = allowed_shifts[allowed_shifts == OFF_SHIFT]
        return allowed_shifts

    def weekend_per_week_constraints(self, a, day_of_week):
        if day_of_week == 5:
            if self.last_shift in (1, 2):
                a = a[a != OFF_SHIFT]
            if self.last_shift == 0:
                a = a[(a == 0) | (a == 3)]
            if self.last_shift == 3:
                a = a[a == OFF_SHIFT]
        if day_of_week == 6:  # we can work and if we work, we must work all weekend
            if self.last_shift == OFF_SHIFT:
                a = a[a == OFF_SHIFT]
            else:
                a = a[a != OFF_SHIFT]
                self.is_work_weekend = True
        return a

    def last_shift_constraints(self, a):
        if self.last_shift == 1:
            a = a[(a != 0)]
        elif self.last_shift == 2:
            a = a[(a == 2) | (a == 3)]
        return a

    def only_two_shift_types_per_week_constraints(self, a):
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
        hard_shift_fair = cp.hard_shift_fair_per_period[self.nurse_hours]

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

    def calculate_fair_costs(self):

        cp = self.cost_parameters
        hard_shift_fair = cp.hard_shift_fair_per_period[self.nurse_hours]

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
            s = f'Finished Roster with plan: {self.plan} for nurse hours {self.nurse_hours} and work weekend ' \
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

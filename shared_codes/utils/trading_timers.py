import time
from abc import ABC, abstractmethod
from functools import total_ordering
import datetime as dt
import heapq

@total_ordering  # This decorator automatically provides other comparison methods
class Timer(ABC):
    """
    Abstract base class representing a generic timer.

    Parameters
    ----------
    target_time : datetime.datetime
        The time at which the timer is set to trigger. Must be a future time.
    callback : callable
        The callback function to be executed when the timer triggers.

    Raises
    ------
    AssertionError
        If the target_time is not a datetime.datetime instance or is not set to a future time.

    Methods
    -------
    check(current_time)
        Checks if the current time is greater than or equal to the target time.
    implement()
        Abstract method to be implemented by subclasses, defining the timer's behavior upon triggering.
    """
    
    def __init__(self, target_time, callback):
        assert isinstance(target_time, dt.datetime), "must be a datetime.datetime object"

        self.target_time = target_time
        self.callback = callback

    def check(self, current_time):
        return current_time >= self.target_time

    @abstractmethod
    def implement(self):
        pass

    def __eq__(self, other):
        if not isinstance(other, Timer):
            return NotImplemented
        return self.target_time == other.target_time

    def __lt__(self, other):
        if not isinstance(other, Timer):
            return NotImplemented
        
        ##
        return self.target_time < other.target_time


class PeriodicTimer(Timer):
    """
    A timer that triggers periodically starting from the next minute. For example, if the interval is 60 seconds, the timer will trigger at the next minute and then every minute thereafter.
    
    Parameters
    ----------
    sec_timedelta : int | float
        The time interval in seconds between each trigger.
    callback : callable
        The callback function to be executed each time the timer triggers.
    start_time : datetime.datetime | None
        The time at which the timer is set to trigger. If None, the timer will trigger at the next minute.
    
    Methods
    -------
    implement()
        Adjusts the target_time by the specified interval and calls the callback function.
    """
    
    def __init__(self, sec_timedelta, callback, start_time=None):
        # next minute
        if start_time is None:
            start_time = dt.datetime.now().replace(second=0, microsecond=0) + dt.timedelta(minutes=1)

        target_time = start_time + dt.timedelta(seconds=sec_timedelta)
        super().__init__(target_time, callback)
        self.sec_timedelta = sec_timedelta

    def implement(self):
        self.target_time += dt.timedelta(seconds=self.sec_timedelta)
        self.callback()


class OnceTimer(Timer):
    """
    A timer that triggers only once.

    Parameters
    ----------
    target_time : datetime.datetime
        The time at which the timer is set to trigger.
    callback : callable
        The callback function to be executed when the timer triggers.

    Methods
    -------
    implement()
        Sets the target_time to None and calls the callback function, indicating the timer will not run again.
    """
    
    def __init__(self, target_time, callback):
        super().__init__(target_time, callback)
        

    def implement(self):
        self.target_time = None
        self.callback()




class trading_timer(ABC):
    """
    A class that manages a list of Timer objects and triggers them when their target time is reached.

    Parameters
    ----------
    timers_list : list[Timer]
        A list of Timer objects to be managed by the TradingTimer instance.

    Methods
    -------
    add_timer(timer)
        Adds a Timer object to the timers_list.
    remove_timer(timer)
        Removes a Timer object from the timers_list.
    run()
        Runs the TradingTimer instance, checking the target time of each Timer object and triggering the callback function when the target time is reached.
    """
    
    def __init__(self):
        self.timers_list = []
        self.coroutine_dict = {}

    def add_timer(self, timer):
        isinstance(timer, PeriodicTimer), "Timer should be of type PeriodicTimer."
        heapq.heappush(self.timers_list, timer)
    
    def remove_timer(self, timer):
        self.timers_list.remove(timer)

    def check_timers(self):
        curr_time = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        while self.timers_list and self.timers_list[0].target_time <= curr_time:
            timer: Timer = heapq.heappop(self.timers_list)
            if timer.check(curr_time):
                timer.implement()
                if timer.target_time:
                    heapq.heappush(self.timers_list, timer)        


import numpy as np
import time

class IterTimer():
    '''Timer for iterations. Usage:

    n_iters = 50
    timer = IterTimer(n_iters=n_iters, ...)

    for args in arguments:
        timer.start_iter()
        your_function(args)
        t_end, mean_iter = timer.end_iter()
    '''

    def __init__(self, n_iters, default_print=True, window=0, log_times=True, log_folder="F:/Code/misc/IterTimer/timer_data"):
        '''
        args:
            n_iters: (int) Total number of iterations.
            default_print: (bool) Whether to let IterTimer print at the end of the iteration. WARNING: uses in-place priting that might interfere with other printing (default: True).
            window: (int) Number of iterations to consider when making the average, -10 is the last 10 iterations, 0 is all (default: 0).
            log_times: (bool) Whether to save the iteration times in an .npy file, useful to make the ETA prediciton code better (default: True).
            log_folder: (str) Folder location of where to save iteration times, using an absolute path is recommended (default: F:/Code/misc/IterTimer/timer_data).
        '''

        self.iter_times = []
        self.n_iters = n_iters
        self.default_print = default_print
        self.window = window
        self.current_iter = 0
        self.log_times = log_times

        if self.log_times:
            #saving the iter times just to get some data about what I tend to measure to find best method
            import uuid
            self.save_file = f"{log_folder}/{uuid.uuid4()}.{self.n_iters}.npy"

    def start_iter(self):
        '''Start measuring time.'''
        self.start = time.perf_counter()

    def end_iter(self, default_print=False):
        '''Finish measuring time and calculate relevant metrics. Will print if defaul_print=True.

        args:
            default_print: (bool) Whether to let IterTimer print at the end of the iteration, overwrites Class default_print (default: False).

        returns:
            t_end: (float) Absolute ETA in seconds.
            mean_iter: (float) Mean iteration time.
        '''

        #keep track of time and calculate ETA based on mean iteration time
        self.iter_times.append(time.perf_counter()-self.start)
        self.mean_iter = np.array(self.iter_times[self.window:]).mean() #only take the last few iterations to calculate mean to account for changing conditions
        self.t_end = time.time() + self.mean_iter*(self.n_iters-self.current_iter-1)

        #if default_print call the function to print
        if self.default_print or default_print: self.print_default()

        #incremenent the number of iterations
        self.current_iter += 1

        #saving the iter times just to get some data about what I tend to measure to find best method
        if self.log_times: np.save(self.save_file, self.iter_times)

        return self.t_end, self.mean_iter

    def print_default(self):
        '''Default printing routine.'''
        if self.current_iter+1 == self.n_iters:
            print(f"Iteration {self.current_iter+1}/{self.n_iters}, ETA: {self.get_time(time.localtime(self.t_end))} (average time per iteration: {self.mean_iter:.3f}s)")
        else:
            print(f"Iteration {self.current_iter+1}/{self.n_iters}, ETA: {self.get_time(time.localtime(self.t_end))} (average time per iteration: {self.mean_iter:.3f}s)")
            # print(f"Iteration {self.current_iter+1}/{self.n_iters}, ETA: {self.get_time(time.localtime(self.t_end))} (average time per iteration: {self.mean_iter:.3f}s)", end="\r")

    def get_time(self, time):
        '''Converts a time.struct_time into HH:mm:ss.'''
        return f"{time.tm_hour:02}:{time.tm_min:02}:{time.tm_sec:02}"

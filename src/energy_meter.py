#!/usr/bin/env python
"""
This module implements the class EnergyMeter, to measure the energy consumption of Python 
functions or code chunks, segregating their energy usage per component (CPU, DRAM, GPU and 
Hard Disk). 
"""
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyRAPL
from pynvml.smi import nvidia_smi

import subprocess
import os
import shlex
import json
import pandas as pd
import time
import threading


class ThreadGpuSamplingCmd(threading.Thread):
    """Thread to sample the power draw of the GPU. It uses nvidia-smi via subprocess check_output 
    to get the immediate power draw of the GPU in Watts every SECONDS_BETWEEN_SAMPLES seconds 
    until self.stop is set to True. The samples are stored in the array self.power_draw_history.
    Note that this process takes much longer than using pynvml (76ms vs. 1ms) but obtains also 
    info about the processes running on the GPU while pynvml doesn't.
    """
    
    SECONDS_BETWEEN_SAMPLES = 0.5
    
    def __init__(self, name):
        """Init the thread variables and the nvsmi instance to be queried later on.
        """
        threading.Thread.__init__(self)
        self.name = name
        self.stop = False
        self.power_draw_history = []
        self.activity_history = []

    def run(self):
        """Start the sampling and stop when self.stop == True.
        """
        # We stop when self.stop is set to True.
        while self.stop == False:
            # Get power draw.
            pd = subprocess.check_output(shlex.split("nvidia-smi --query-gpu=power.draw --format=csv")).decode().split()
            self.power_draw_history.append(float(pd[2]))
            
            # Get utilization of python/python3 at each time step.
            o = subprocess.check_output(shlex.split("nvidia-smi pmon -c 1")).decode().split()
            processes_util = {o[i]: o[i-4] for i in range(25, len(o), 8)}
            activity = 0
            if processes_util.get("python") and processes_util.get("python") != "-":
                activity += float(processes_util.get("python"))
            if processes_util.get("python3") and processes_util.get("python3") != "-":
                activity += float(processes_util.get("python3"))
            self.activity_history.append(activity)

            # Sleep until next cycle.
            time.sleep(ThreadGpuSampling.SECONDS_BETWEEN_SAMPLES)


class ThreadGpuSamplingPyNvml(threading.Thread):
    """Thread to sample the power draw of the GPU. It uses pynvml to get the immediate power
    draw of the GPU in Watts every SECONDS_BETWEEN_SAMPLES seconds until self.stop is set to
    True. The samples are stored in the array self.power_draw_history.
    """
    
    SECONDS_BETWEEN_SAMPLES = 0.5
    
    def __init__(self, name):
        """Init the thread variables and the nvsmi instance to be queried later on.
        """
        threading.Thread.__init__(self)
        self.name = name
        self.stop = False
        self.power_draw_history = []
        self.activity_history = []
        self.nvsmi = nvidia_smi.getInstance()

    def run(self):
        """Start the sampling and stop when self.stop == True.
        """
        while self.stop == False:
            nvml_output = self.nvsmi.DeviceQuery("power.draw,utilization.gpu").get("gpu")[0]
            # Get power draw.
            self.power_draw_history.append(nvml_output.get("power_readings").get("power_draw"))
            # Get utilization at each time step.
            self.activity_history.append(nvml_output.get("utilization").get("gpu_util"))


class EnergyMeter:
    """
    The consumption of each component is measured as follows:

    - CPU: the energy consumption of the CPU is measured with RAPL via the pyRAPL
        library. RAPL is an API from Intel, which is also semi-compatible with
        AMD. RAPL on Intel has been shown to be accurate thanks to the usage of
        embedded sensors in the processor and memory while AMD uses performance
        counters and is therefore, not so accurate.

    - DRAM: the energy used by the memory is also measure with RAPL via pyRAPL.
        This might not be available for AMD processors and pre-Haswell Intel
        processors. You can find more info here:
        https://dl.acm.org/doi/pdf/10.1145/2989081.2989088.

    - GPU: we measure the energy consumption of the GPU with nvidia-smi. For this, we
        run a separate thread that samples the power draw of the GPU while the meter
        is running. We then calculate the mean of this and multiply it by the 
        duration of the meter.

    - Disk: we cannot directly measure the energy consumption of the disk in the same
        way that we do for the other components, so we have implemented an bpftrace
        probe that tracks all the bytes read and written to disk. This probe is run 
        as a separate thread inside EnergyMeter. We then calculate the energy 
        consumption with the following formulae:
        disk_active_time = (bytes_read + bytes_written) / DISK_SPEED
        disk_idle_time = total_meter_time - disk_active_time
        total_energy = disk_active_time * DISK_ACTIVE_POWER +
                        disk_idle_time * DISK_IDLE_POWER
        Note that you are required the provide the parameters DISK_SPEED,
        DISK_ACTIVE_POWER and DISK_IDLE_POWER.
    """

    #################################### CONSTANTS ####################################
    SCRIPT = (
        "tracepoint:syscalls:sys_enter_write {@wbytes[comm] = sum(args->count);} "
        "tracepoint:syscalls:sys_enter_read {@rbytes[comm] = sum(args->count);}"
    )
    ###################################################################################

    def __init__(self, disk_avg_speed, disk_active_power, disk_idle_power, 
                 label=None, include_idle=False):
        """Initiates the variables required to meter the energy consumption of all
        components and sets up the pyRAPL library.
        :param disk_avg_speed: the average read and write speed of the hard disk where
            the code will be run. We recommend measuring this with a speed test such as
            this: https://cloud.google.com/compute/docs/disks/benchmarking-pd-performance.
        :param disk_active_power: the power used by the disk when active. This information
            is usually included in the disk specs. Hint: run lshw to get info about the
            host's disk.
        :param disk_idle_power: the average power used by the disk when idle. Just as for
            disk_active_power, this is usually included in the disk specs.
        :param label: this is just an optional string to identify the meter.
        :include_idle: if energy used during idle times should be included for disk and 
            GPU.
        """
        if label:
            self.label = label
        else:
            self.label = "Meter"

        self.include_idle = include_idle

        # Setup pyRAPL to measure CPU and DRAM.
        pyRAPL.setup()

        # Setup disk parameters.
        self.disk_avg_speed = disk_avg_speed
        self.disk_active_power = disk_active_power
        self.disk_idle_power = disk_idle_power

        # Create thread for sampling the power draw of the GPU, this sets up pynvm.
        self.thread_gpu = ThreadGpuSamplingPyNvml("GPU Sampling Thread")

        # Create the pyRAPL meter to measure CPU and DRAM energy consumption.
        self.meter = pyRAPL.Measurement(self.label)

        # Create command for bpftrace subprocess that will count the bytes read and
        # written to disk.
        self.bpftrace_command = shlex.split(
            "sudo bpftrace -f json -e '{}'".format(EnergyMeter.SCRIPT)
        )

    def begin(self):
        """Begin measuring the energy consumption. This sets the starting datetime and
        reads the current RAPL counters. You should have start the bash script
        start_meters.sh BEFORE calling this function.
        """
        # pyRAPL for CPU and DRAM.
        self.meter.begin()

        # bpftrace for disk.
        self.popen = subprocess.Popen(
            self.bpftrace_command, stdout=subprocess.PIPE, preexec_fn=os.setpgrp
        )
        # We save the bpftrace pid to kill it later.
        self.bpftrace_pid = os.getpgid(self.popen.pid)

        # Thread for GPU.
        self.thread_gpu.start()

    def end(self):
        """Finish the measurements and calculate results for CPU and DRAM. This sets the
        duration of the meter and reads again the RAPL counters, calculating how much energy
        was used since the meter began. You should stop running the bash script
        start_meters.sh AFTER calling this method.
        """
        # PyRAPL.
        self.meter.end()

        # Kill bpftrace subprocess.
        subprocess.check_output(shlex.split("sudo kill {}".format(self.bpftrace_pid)))

        # Stop tracking GPU power usage.
        self.thread_gpu.stop = True

        # Process bpftrace output.
        po = self.popen.stdout.read()
        self.total_rbytes, self.total_wbytes = self.__preprocess_bpftrace_output(po)

    def __preprocess_bpftrace_output(self, bpftrace_output):
        """Preprocess the output of out bpftrace script and extract the bytes read and
        written. Attention: This must be changed if the bpftrace script changes!
        :param bpftrace_output: the output of the bpftrace script.
        :returns: total_rbytes (float), total_wbytes (float).
        """
        bpftrace_output = bpftrace_output.decode()
        if len(bpftrace_output.strip()) > 0:
            po = bpftrace_output.split("\n")
            rbytes = json.loads(po[3]).get("data").get("@rbytes")
            wbytes = json.loads(po[4]).get("data").get("@wbytes")
            # Do we want to measure other programs disk IO too?
            total_rbytes = rbytes.get("python", 0) + rbytes.get("python3", 0)
            total_wbytes = wbytes.get("python", 0) + wbytes.get("python3", 0)
        else:
            # bpftrace produced no output, which means there was no IO activity in the
            # disk. This only happens when the code run has a very short duration.
            total_rbytes = 0
            total_wbytes = 0

        return total_rbytes, total_wbytes

    def get_total_joules_disk(self):
        """We calculate the disk's energy consumption while the meter was running. For this,
        we require the csv file that was generated by running the bash script start_meters.sh.
        In this case, we utilize the speed and energy consumption parameters given when this
        object was initiated to estimate the disk's energy consumption. The formula used here
        was derived from:
        [1] Kansal, A., Zhao, F., Liu, J., Kothari, N., & Bhattacharya, A. A. (2010, June). 
        Virtual machine power metering and provisioning. In Proceedings of the 1st ACM 
        symposium on Cloud computing (pp. 39-50).

        :param filename: the path to the csv file generated by start_meters.sh (or where the
            output of disk_io.bt was saved.)
        :returns: the total joules used by the disk between meter.begin() and meter.end().
        """
        tot_bytes = self.total_rbytes + self.total_wbytes

        # disk_active_time (in seconds) = (bytes_read + bytes_written) / DISK_SPEED
        disk_active_time = tot_bytes / self.disk_avg_speed

        # disk_idle_time (in seconds) = total_meter_time - disk_active_time
        disk_idle_time = self.meter.result.duration * 1e-6 - disk_active_time

        # total_energy = disk_active_time * DISK_ACTIVE_POWER (+ disk_idle_time * DISK_IDLE_POWER)
        te = disk_active_time * self.disk_active_power
        if self.include_idle:
            te += disk_idle_time * self.disk_idle_power
        return te

    def get_total_joules_cpu(self):
        """We obtain the total joules consumed by the CPU from pyRAPL.
        :returns: the total joules used by the CPU between meter.begin() and meter.end().
        """
        # pyRAPL returns the microjoules, so we convert them to joules.
        return np.array(self.meter.result.pkg) * 1e-6

    def get_total_joules_dram(self):
        """We obtain the total joules consumed by the DRAM from pyRAPL.
        :returns: the total joules used by the DRAM between meter.begin() and meter.end().
        """
        # pyRAPL returns the microjoules, so we convert them to joules.
        return np.array(self.meter.result.dram) * 1e-6

    def get_total_joules_gpu(self):
        """We calculate the GPU's energy consumption while the meter was running. For this,
        we require the csv file that was generated by running the bash script start_meters.sh.
        This is calculated as the mean power used between meter.begin() and meter.end() times
        the total time in seconds.
        :param filename: the path to the csv file generated by start_meters.sh (or the file
            generated by gpu_stats.sh.)
        :returns: the total joules used by the GPU between meter.begin() and meter.end().
        """
        if len(self.thread_gpu.activity_history) == 0:
            return 0

        # total_energy = mean_GPU_power_draw * min(meter_duration_ns * 1e-6, GPU_active_time_s)
        if self.include_idle:
            # We use the mean power draw thoughout the whole time (including idle time.)
            mean_p = np.mean(self.thread_gpu.power_draw_history)
            te = mean_p * self.meter.result.duration * 1e-6
        else:
            # First, check if there was any activity, otherwise return 0.
            if sum(self.thread_gpu.activity_history) == 0:
                return 0
                
            # We calculate the mean power draw during intervals of time at which
            # python or python3 was active on the GPU.
            pdh = self.thread_gpu.power_draw_history
            ah = self.thread_gpu.activity_history
            assert len(pdh) == len(ah), "Power draw and activity history have diff lengths!"

            sbs = self.thread_gpu.SECONDS_BETWEEN_SAMPLES
            mean_p = np.mean([pdh[i] for i in range(len(pdh)) if ah[i] > 0])
            # We estimate the task was running for length of samples * the time between samples,
            # or the duration of the meter if this was shorter. Our error in this estimation is
            # bounded to (t - 2*SECONDS_BETWEEN_SAMPLES, t + SECONDS_BETWEEN_SAMPLES).
            te = mean_p * min(self.meter.result.duration * 1e-6, len(ah)*sbs)
            
        return te

    def get_total_joules_per_component(self):
        """This returns the total energy consumption in joules between meter.begin() and
        meter.end() segregated by component (CPU, DRAM, GPU and disk).
        :param foldername: the path to the folder generated by start_meters.sh.
        :returns: a dictionary with the total joules used by each component.
        """
        cpu = self.get_total_joules_cpu()
        dram = self.get_total_joules_dram()
        gpu = self.get_total_joules_gpu()
        disk = self.get_total_joules_disk()
        res = {
            "cpu": cpu,
            "dram": dram,
            "gpu": gpu,
            "disk": disk,
        }
        return res

    def plot_total_joules_per_component(self, include_total=True):
        """This plots the total energy consumption in joules between meter.begin() and
        meter.end() and the total consumption by each component (CPU, DRAM, GPU and disk).
        :param foldername: the path to the folder generated by start_meters.sh.
        """
        data = self.get_total_joules_per_component()
        if include_total:
            data["total"] = (
                np.sum(data.get("cpu"))
                + np.sum(data.get("dram"))
                + data.get("disk")
                + data.get("gpu")
            )
        keys = data.keys()
        values = [float(val) for val in data.values()]

        fig, ax = plt.subplots()
        bars = ax.bar(list(keys), values)
        ax.bar_label(bars)
        plt.xlabel("Components")
        plt.ylabel("Jules")
        plt.title(self.meter.label)

        plt.show()


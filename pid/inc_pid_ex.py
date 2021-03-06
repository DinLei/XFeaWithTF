#!/usr/bin/env python
# encoding: utf-8

# @author: ba_ding
# @contact: dinglei_1107@outlook.com
# @file: inc_pid_ex.py
# @time: 2020/10/14 17:05


import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import os


class PID_Prama:
    def __init__(self):
        self.Kp = 0
        self.Ki = 0
        self.Kd = 0
        self.set_val = 0
        self.error_last = 0
        self.error_prev = 0
        self.error_sum = 0


# 增量计算公式：
# Pout=Kp*[e(t) - e(t-1)] + Ki*e(t) + Kd*[e(t) - 2*e(t-1) +e(t-2)]
def PID_Controller_Increa(pid, out_now):
    error = pid.set_val - out_now
    Res = pid.Kp * (error - pid.error_last) + pid.Ki * error + \
          pid.Kd * (error - 2 * pid.error_last + pid.error_prev)
    pid.error_prev = pid.error_last
    pid.error_last = error
    return float(Res)


standard_out = 100
PID_val = PID_Prama()
Time = 100

# PID参数
PID_val.Kp = 0.01
PID_val.Ki = 0.1
PID_val.Kd = 0.05
PID_val.set_val = standard_out  # 标准输出值

# 增量型PID控制器输出值
PID_Controller_Increa_Out = []
Sys_In = []
# 0时刻系统输入值
Sys_In.append(5)
Last_Sys_State = float(Sys_In[0])
# 系统响应函数
SystemFunc = lambda x: float(5.0 * x + np.random.normal(0, 0.5, 1)[0])

Sys_Out = []
# 0时刻系统输出值
Sys_Out.append(SystemFunc(Sys_In[0]))

# 加和更新
for t_slice in range(Time):
    Diff = PID_Controller_Increa(PID_val, Sys_Out[t_slice])  # 系统误差
    PID_Controller_Increa_Out.append(Diff)  # 记录所有的系统误差
    Last_Sys_State += Diff
    # Sys_In.append(Sys_In[0] + np.sum(PID_Controller_Increa_Out))  # 计算增量之后的新的系统输入
    Sys_In.append(float(Last_Sys_State))
    Sys_Out.append(SystemFunc(Sys_In[t_slice + 1]))  # 计算下一时刻系统新的输出值

standard = np.linspace(PID_val.set_val, PID_val.set_val, Time)

plt.figure('PID_Controller_Inc')
plt.xlim(0, Time)
plt.ylim(0, 2 * standard_out)
plt.plot(Sys_Out)
plt.plot(standard)

plt.show()

# i = 0
# while True:
#     i += 1

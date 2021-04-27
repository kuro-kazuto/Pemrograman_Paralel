# -*- coding: utf-8 -*-

"""
Spyder Editor

This is a temporary script file.
"""
import pycuda.driver as drv

drv.init()

print("Total Device is %s" % (drv.Device.count()))

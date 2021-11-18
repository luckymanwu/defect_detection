# -*- encoding=utf-8 -*-
import serial
import time

class COM:
    def __init__(self, port, baud):
        self.port = port
        self.baud = int(baud)
        self.open_com = None
        self.get_data_flag = True
        self.real_time_data = ''


    # return real time data form com
    def get_real_time_data(self):
        return self.real_time_data

    def clear_real_time_data(self):
        self.real_time_data = ''

    # set flag to receive data or not
    def set_get_data_flag(self, get_data_flag):
        self.get_data_flag = get_data_flag

    def open(self):
        try:
            self.open_com = serial.Serial(self.port, self.baud)
        except Exception as e:
            self.log.error('Open com fail:{}/{}'.format(self.port, self.baud))
            self.log.error('Exception:{}'.format(e))

    def close(self):
        if self.open_com is not None and self.open_com.isOpen:
            self.open_com.close()

    def send_data(self, data):
        if self.open_com is None:
            self.open()
        success_bytes = self.open_com.write(data.encode('UTF-8'))
        return success_bytes

    def get_data(self, over_time=30):
        if self.open_com is None:
            self.open()
        start_time = time.time()
        end_time = time.time()
        if end_time - start_time < over_time and self.get_data_flag:
            data = self.open_com.readline()
            # data = self.open_com.read() # read 1 size
            data = str(data,encoding="UTF-8")
            if data != '':
                return data.strip()
        else:
            self.set_get_data_flag(True)


com = COM('COM4', 115200)


#
# def get_speed():
#     com = COM('COM4', 115200)
#     com.open()
#     while True:
#     # com.send_data('data')
#         com.get_data(50)
#         print(com.real_time_data)

import json
import threading
import time
import socket
import quaternion
import numpy as np

class quest_teleop:
    def __init__(self):
        self.last_input = None
        self.joint_states = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.offset_tcp_pos = np.array([0., 0., 0.])
        self.offset_tcp_quat = quaternion.quaternion(0.,0.,0.,0.)
        thread1 = threading.Thread(target=self.udp_receiver)
        thread2 = threading.Thread(target=self.udp_sender)
        thread1.daemon = True
        thread2.daemon = True
        thread1.start()
        thread2.start()

    def get_input_frame(self):
        a = self.offset_tcp_pos.copy()
        b = self.offset_tcp_quat.copy()
        # self.offset_tcp_pos = np.array([0., 0., 0.])
        # self.offset_tcp_quat = quaternion.quaternion(1., 0., 0., 0.)
        return self.last_input, a, b

    def hand_input(self, input):
        if self.last_input is None:
            self.last_input = input

        if input['rightHand'] > 0.5:
            self.offset_tcp_pos = np.array([input['rightPos']["z"], -input['rightPos']["x"], input['rightPos']["y"]]) - np.array([self.last_input['rightPos']["z"], -self.last_input['rightPos']["x"], self.last_input['rightPos']["y"]])
            # start_quat = quaternion.quaternion(self.last_input['rightQuat']["w"], self.last_input['rightQuat']["x"],
            #                        self.last_input['rightQuat']["y"], self.last_input['rightQuat']["z"])
            # self.offset_tcp_quat = quaternion.quaternion.inverse(start_quat) * quaternion.quaternion(input['rightQuat']["w"], input['rightQuat']["x"],
            #                        input['rightQuat']["y"], input['rightQuat']["z"])
        else:
            self.offset_tcp_pos = np.array([0.,0.,0.])
            self.offset_tcp_quat = quaternion.quaternion(1.,0.,0.,0.)

        self.last_input = input

    def udp_receiver(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('0.0.0.0', 10001))
        while True:
            # Receive data
            data, addr = s.recvfrom(102400)
            data = data.decode('utf-8')
            data = json.loads(data)
            self.hand_input(data)

    def udp_sender(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('0.0.0.0', 0))
        address = ("192.168.2.250", 10004)
        while True:
            joint_state = self.joint_states * 180 / np.pi
            joint_state = {"jointPositions": joint_state.tolist()}
            joint_state = json.dumps(joint_state)
            joint_state = joint_state.encode('utf-8')
            s.sendto(joint_state, address)
            time.sleep(0.02)


if __name__ == "__main__":
    quest_teleop = quest_teleop()

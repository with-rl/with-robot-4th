import time
import numpy as np

import mujoco
from mujoco.glfw import glfw


class TidyBot:
    def __init__(self, xml_fn, title):
        self.run_flag = True
        self.xml_fn = xml_fn
        self.title = title

    # MuJoCo 환경 초기화
    def init_mujoco(self):
        # MuJoCo data structures
        self.model = mujoco.MjModel.from_xml_path(self.xml_fn)  # MuJoCo model
        self.data = mujoco.MjData(self.model)  # MuJoCo data
        self.cam = mujoco.MjvCamera()  # Abstract camera
        self.opt = mujoco.MjvOption()  # visualization options

        # Init GLFW, create window, make OpenGL context current, request v-sync
        glfw.init()
        self.window = glfw.create_window(1200, 900, self.title, None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        # initialize visualization data structures
        mujoco.mjv_defaultCamera(self.cam)
        mujoco.mjv_defaultOption(self.opt)
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

        # install GLFW mouse and keyboard callbacks
        glfw.set_key_callback(self.window, self.keyboard_cb)

        # initialize camera
        self.init_cam()
        # initialize the controller
        self.init_controller(self.model, self.data)

    # 카메라 위치 초기화
    def init_cam(self):
        # initialize camera
        self.cam.azimuth = 90
        self.cam.elevation = -45
        self.cam.distance = 2.5
        self.cam.lookat = np.array([0.0, 0.0, 1.0])

    # 키 입력 처리
    def keyboard_cb(self, window, key, scancode, act, mods):
        if act == glfw.PRESS and key == glfw.KEY_Q:  # q: 종료
            self.run_flag = False

    # 초기 MuJoCo 제어 정보 입력
    def init_controller(self, model, data):
        pass

    # 스텝별 MuJoCo 제어 정보 입력 (제어할 내용이 있으면 True 리턴)
    def controller_cb(self, model, data):
        # data.ctrl[0] += 0.005  # joint_x
        # data.ctrl[1] += 0.005  # joint_y
        # data.ctrl[2] += 0.005  # joint_th
        # data.ctrl[3] += 0.005  # joint_1
        # data.ctrl[4] += 0.005  # joint_2
        # data.ctrl[5] += 0.005  # joint_3
        # data.ctrl[6] += 0.005  # joint_4
        # data.ctrl[7] += 0.005  # joint_5
        # data.ctrl[8] += 0.005  # joint_6
        # data.ctrl[9] += 0.005  # joint_7
        # data.ctrl[10] += 0.5  # fingers_actuator
        return True

    def run_mujoco(self, ft=0.02):
        while self.run_flag and not glfw.window_should_close(self.window):
            time_prev = self.data.time
            while self.data.time - time_prev < ft:
                mujoco.mj_step(self.model, self.data)

            # self
            if self.controller_cb(self.model, self.data):
                mujoco.mj_forward(self.model, self.data)

            # get framebuffer viewport
            viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
            viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)

            # Update scene and render
            mujoco.mjv_updateScene(
                self.model,
                self.data,
                self.opt,
                None,
                self.cam,
                mujoco.mjtCatBit.mjCAT_ALL.value,
                self.scene,
            )
            mujoco.mjr_render(viewport, self.scene, self.context)

            # swap OpenGL buffers (blocking call due to v-sync)
            glfw.swap_buffers(self.window)

            # process pending GUI events, call GLFW callbacks
            glfw.poll_events()

        glfw.terminate()


if __name__ == "__main__":
    simulator = TidyBot("./model/stanford_tidybot/scene.xml", "TidyBot")
    simulator.init_mujoco()
    simulator.run_mujoco()

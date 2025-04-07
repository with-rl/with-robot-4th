# Copyright 2025 @with-RL
# Reference from
#    - https://pab47.github.io/legs.html
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import matplotlib.pyplot as plt

import mujoco
from mujoco.glfw import glfw


class TidyBot:
    def __init__(self, xml_fn, title):
        self.run_flag = True
        self.xml_fn = xml_fn
        self.title = title

        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.lastx = 0
        self.lasty = 0

        self.plt_objs = [None] * 100

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
        glfw.set_mouse_button_callback(self.window, self.mouse_button_cb)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move_cb)
        glfw.set_scroll_callback(self.window, self.scroll_cb)

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
        self.cam.lookat = np.array([0.0, 0.0, 0.0])
        # init second cam
        self.cam2 = mujoco.MjvCamera()
        self.cam2.fixedcamid = self.data.cam("wrist").id
        self.cam2.type = mujoco.mjtCamera.mjCAMERA_FIXED

    # 키 입력 처리
    def keyboard_cb(self, window, key, scancode, act, mods):
        if act == glfw.PRESS and key == glfw.KEY_Q:  # q: 종료
            self.run_flag = False

    # 마우스 버튼 클릭 처리
    def mouse_button_cb(self, window, button, act, mods):
        self.button_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self.button_middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        self.button_right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        # update mouse position
        glfw.get_cursor_pos(window)

    # 마우스 이동 처리
    def mouse_move_cb(self, window, xpos, ypos):
        # no buttons down: nothing to do
        if (not self.button_left) and (not self.button_middle) and (not self.button_right):
            return

        dx = xpos - self.lastx
        dy = ypos - self.lasty
        self.lastx = xpos
        self.lasty = ypos

        # get current window size
        width, height = glfw.get_window_size(window)

        # get shift key state
        PRESS_LEFT_SHIFT = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        PRESS_RIGHT_SHIFT = glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        mod_shift = PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT

        # determine action based on mouse button
        if self.button_right:
            if mod_shift:
                action = mujoco.mjtMouse.mjMOUSE_MOVE_H
            else:
                action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self.button_left:
            if mod_shift:
                action = mujoco.mjtMouse.mjMOUSE_ROTATE_H
            else:
                action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        mujoco.mjv_moveCamera(self.model, action, dx / height, dy / height, self.scene, self.cam)

    # 마우스 스코롤 처리
    def scroll_cb(self, window, xoffset, yoffset):
        action = mujoco.mjtMouse.mjMOUSE_ZOOM
        mujoco.mjv_moveCamera(self.model, action, 0.0, -0.05 * yoffset, self.scene, self.cam)

    # 초기 MuJoCo 제어 정보 입력
    def init_controller(self, model, data):
        pass

    # 스텝별 MuJoCo 센서 정보 조회
    def read_cb(self, camera_flag=False):
        if camera_flag:
            cam_with, cam_height = 480, 480
            viewport2 = self.render_camera(self.cam2, cam_with, cam_height)
            mujoco.mjr_render(viewport2, self.scene, self.context)
            img = np.zeros((cam_with, cam_height, 3), dtype=np.uint8)
            mujoco.mjr_readPixels(img, None, viewport2, self.context)
            img = np.flipud(img)  # flip image
        else:
            img = None
        return (img,)

    # 스텝별 MuJoCo 제어 정보 입력 (제어할 내용이 있으면 True 리턴)
    def control_cb(self, model, data, read_data):
        # print(model.camera("customview"))
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
        data.ctrl[10] += 0.5  # fingers_actuator
        return True

    def visualize(self, read_data):
        # remove visual objects
        for i in range(len(self.plt_objs)):
            if self.plt_objs[i] is None:
                break
            self.plt_objs[i].remove()
            self.plt_objs[i] = None

        self.plt_objs[0] = plt.imshow(read_data[0])
        plt.pause(0.001)

    def render_camera(self, cam, width, height):
        viewport = mujoco.MjrRect(0, 0, width, height)
        # Update scene and render
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            None,
            cam,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scene,
        )
        mujoco.mjr_render(viewport, self.scene, self.context)
        return viewport

    def run_mujoco(self, ft=0.02):
        # renderer = mujoco.Renderer(self.model)
        while self.run_flag and not glfw.window_should_close(self.window):
            time_prev = self.data.time
            while self.data.time - time_prev < ft:
                mujoco.mj_step(self.model, self.data)

            read_data = self.read_cb(camera_flag=True)
            if self.control_cb(self.model, self.data, read_data):
                mujoco.mj_forward(self.model, self.data)
            self.visualize(read_data)

            # get framebuffer viewport
            viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
            self.render_camera(self.cam, viewport_width, viewport_height)
            # swap OpenGL buffers (blocking call due to v-sync)
            glfw.swap_buffers(self.window)
            # process pending GUI events, call GLFW callbacks
            glfw.poll_events()

        glfw.terminate()


if __name__ == "__main__":
    simulator = TidyBot("./model/stanford_tidybot/scene.xml", "TidyBot")
    # simulator = TidyBot("./model.xml", "TidyBot")
    simulator.init_mujoco()
    simulator.run_mujoco()

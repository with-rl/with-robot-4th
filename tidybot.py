# Copyright 2025 @with-RL
# Reference from
#    - https://pab47.github.io/legs.html
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

import time
import io

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize

import matplotlib.pyplot as plt
from PIL import Image

import mujoco
from mujoco.glfw import glfw

from flask import Flask, jsonify, request, Response
import threading

simulator = None
app = Flask(__name__)


PI_HALF = np.pi / 2


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

        self.joints = np.array([0, 0, 0, 0, np.pi / 3, 0, np.pi / 3, 0, np.pi / 3, np.pi / 2, 50], np.float32)

        self.plt_objs = [None] * 100

        self.capture_flag = False
        self.capture_img = None

    # MuJoCo 환경 초기화
    def init_mujoco(self):
        # MuJoCo data structures
        self.model = mujoco.MjModel.from_xml_path(self.xml_fn)  # MuJoCo model
        # self.model.light_castshadow[:] = 0  # 0 = 그림자 비활성화, 1 = 활성화
        self.data = mujoco.MjData(self.model)  # MuJoCo data
        self.data_fk = mujoco.MjData(self.model)  # MuJoCo data for KF
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
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False  # disable Rangefinder rendering
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
        self.cam.azimuth = 180
        self.cam.elevation = -75
        self.cam.distance = 1.0
        self.cam.lookat = np.array([0.5, 0.0, 0.5])
        # init second cam
        wrist_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
        if wrist_id >= 0:
            self.cam2 = mujoco.MjvCamera()
            self.cam2.fixedcamid = wrist_id
            self.cam2.type = mujoco.mjtCamera.mjCAMERA_FIXED
        else:
            self.cam2 = None

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

        mujoco.mjv_moveCamera(self.model, action, dx / width, dy / height, self.scene, self.cam)

    # 마우스 스코롤 처리
    def scroll_cb(self, window, xoffset, yoffset):
        action = mujoco.mjtMouse.mjMOUSE_ZOOM
        mujoco.mjv_moveCamera(self.model, action, 0.0, -0.05 * yoffset, self.scene, self.cam)

    # 키 입력 처리
    def keyboard_cb(self, window, key, scancode, act, mods):
        if act == glfw.PRESS and key == glfw.KEY_Q:  # q: 종료
            self.run_flag = False

    # 초기 MuJoCo 제어 정보 입력
    def init_controller(self, model, data):
        pass

    def to_position(self, element):
        pos = element.xpos
        rot = R.from_matrix(element.xmat.reshape(3, 3)).as_euler("xyz")
        return np.concatenate((pos, rot))

    def get_ee_position(self):
        mujoco.mj_kinematics(self.model, self.data)
        return self.to_position(self.data.site("pinch_site"))

    # 스텝별 MuJoCo 센서 정보 조회
    def read_cb(self, camera_flag=False):
        if camera_flag and self.cam2:
            cam_with, cam_height = 480, 480
            viewport2 = self.render_camera(self.cam2, cam_with, cam_height)
            mujoco.mjr_render(viewport2, self.scene, self.context)
            img = np.zeros((cam_with, cam_height, 3), dtype=np.uint8)
            mujoco.mjr_readPixels(img, None, viewport2, self.context)
            img = np.flipud(img)  # flip image
        else:
            img = None
        return (img,)

    def fk(self, joints):
        self.data_fk.qpos[3:10] = joints
        mujoco.mj_kinematics(self.model, self.data_fk)
        pos = self.data_fk.site("pinch_site").xpos
        rot = R.from_matrix(self.data_fk.site("pinch_site").xmat.reshape(3, 3)).as_euler("xyz")
        return np.concatenate((pos, rot))

    def ik_cost(self, thetas, position):
        position_hat = self.fk(thetas)
        p_error = np.linalg.norm(position[:3] - position_hat[:3])
        t_quat = R.from_euler("xyz", position[3:]).as_quat()
        h_quat = R.from_euler("xyz", position_hat[3:]).as_quat()
        r_error = 1 - np.dot(h_quat, t_quat) ** 2
        return p_error + r_error * 0.1

    def solve_ik(self, position):
        initial_thetas = np.copy(self.data.ctrl[3:10])
        theta_bounds = [
            (np.deg2rad(-180), np.deg2rad(180)),
            (np.deg2rad(-180), np.deg2rad(180)),
            (np.deg2rad(-180), np.deg2rad(180)),
            (np.deg2rad(-180), np.deg2rad(180)),
            (np.deg2rad(-180), np.deg2rad(180)),
            (np.deg2rad(-180), np.deg2rad(180)),
            (np.deg2rad(-180), np.deg2rad(180)),
        ]

        result = minimize(
            self.ik_cost,  # 목적 함수
            initial_thetas,  # 초기값
            args=(position,),  # 추가 매개변수
            bounds=theta_bounds,  # 범위 제한
            method="SLSQP",  # 제약 조건을 지원하는 최적화 알고리즘
            options={"ftol": 1e-6, "maxiter": 500},  # 수렴 기준
        )
        return result.x

    # 스텝별 MuJoCo 제어 정보 입력
    def control_cb(self, model, data, read_data):
        # 0: x, 1: y, 3: theta
        # 3: joint_1, 4: joint_2, 5: joint_3, 6: joint_4,
        # 7: joint_5, 8: joint_6, 9: joint_7, 10: fingers_actuator
        if data.ctrl.shape == self.joints.shape:
            for i in range(len(self.joints)):
                delta = self.joints[i] - data.ctrl[i]
                if i == 10:  # ee control
                    delta = np.clip(delta, -2, 2)
                else:
                    delta = np.clip(delta, -0.005, 0.005)

                data.ctrl[i] += delta

    def visualize(self, read_data):
        # remove visual objects
        for i in range(len(self.plt_objs)):
            if self.plt_objs[i] is None:
                break
            self.plt_objs[i].remove()
            self.plt_objs[i] = None

        if read_data[0] is not None:
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
        if cam == self.cam and self.capture_flag:
            self.capture_flag = False
            # viewport to image
            img = np.zeros((height, width, 3), dtype=np.uint8)
            depth = np.zeros((height, width, 1), dtype=np.float32)
            mujoco.mjr_readPixels(rgb=img, depth=depth, viewport=viewport, con=self.context)
            depth = np.flipud(depth)
            depth -= depth.min()
            depth /= 2 * depth[depth <= 1].mean()
            depth = 255 * np.clip(depth, 0, 1)
            depth = depth.astype(np.uint8).reshape(height, width)
            self.capture_img = Image.fromarray(depth)
        return viewport

    def run_mujoco(self, ft=0.02):
        # renderer = mujoco.Renderer(self.model)
        while self.run_flag and not glfw.window_should_close(self.window):
            time_prev = self.data.time
            while self.data.time - time_prev < ft:
                mujoco.mj_step(self.model, self.data)

            read_data = self.read_cb(camera_flag=True)
            self.control_cb(self.model, self.data, read_data)
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


@app.route("/stop", methods=["GET"])
def stop():
    simulator.run_flag = False
    return ""


@app.route("/get_ee", methods=["GET"])
def get_ee_position():
    ee_position = simulator.get_ee_position()
    return jsonify({"ee_position": list(ee_position)})


@app.route("/set_ee", methods=["POST"])
def set_ee_position():
    data = request.json
    joints = simulator.solve_ik(data["ee_position"])
    simulator.joints[3:10] = joints
    for i in range(100):
        if np.linalg.norm(simulator.joints - simulator.data.ctrl) < 0.1:
            break
        time.sleep(0.1)
    return ""


@app.route("/get_site", methods=["POST"])
def get_site_position():
    data = request.json
    position = simulator.to_position(simulator.data.site(data["site_name"]))
    return jsonify({"site_position": list(position)})


@app.route("/get_body", methods=["POST"])
def get_body_position():
    data = request.json
    position = simulator.to_position(simulator.data.body(data["body_name"]))
    return jsonify({"body_position": list(position)})


@app.route("/set_joints", methods=["POST"])
def set_joints():
    data = request.json
    simulator.joints[:] += np.array(data["joints"])
    for i in range(100):
        if np.linalg.norm(simulator.joints - simulator.data.ctrl) < 0.1:
            break
        time.sleep(0.1)
    return ""


@app.route("/set_gripper", methods=["POST"])
def set_gripper():
    data = request.json
    # simulator.joints[10] = 250 if data["gripper"] == 1 else 50
    for i in range(100):
        if np.linalg.norm(simulator.joints - simulator.data.ctrl) < 0.1:
            break
        time.sleep(0.1)
    return ""


@app.route("/capture", methods=["GET"])
def capture_camera():
    simulator.capture_flag = True
    for i in range(10):
        if simulator.capture_img is not None:
            break
        time.sleep(0.02)
    img_io = io.BytesIO()
    if simulator.capture_img is not None:
        img = simulator.capture_img
        simulator.capture_img = None
        img.save(img_io, format="PNG")  # PNG 형식으로 저장
    img_io.seek(0)
    return Response(img_io, mimetype="image/png")


if __name__ == "__main__":
    simulator = TidyBot("./model/stanford_tidybot/scene.xml", "TidyBot")
    # start flask web server
    threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=5555, debug=False, use_reloader=False),
        daemon=True,
    ).start()
    simulator.init_mujoco()
    simulator.run_mujoco()

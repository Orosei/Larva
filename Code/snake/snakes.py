import os

import cv2
import numpy as np
import torch

from .snake import Snake
from ..my_addition import draw_dir, add_args, draw_dir_v2
# from ..my_addition.Model.DirectionNet.direction import Direction
from ..my_addition.Model.crack.direction import Direction
from ..my_addition.Model.get_direction import get_pred_direction_v2

cur_path = os.path.abspath(__file__)
root_path = cur_path
for _ in range(3):
    root_path = os.path.dirname(root_path)


class Snakes:
    def __init__(self, img, label, seg_map, curve_map, ske_map, ext_eng_map, img_name):
        self.img = img
        self.label = label
        self.seg_map = seg_map
        self.curve_map = curve_map
        self.ske_map = ske_map
        self.ext_eng_map = ext_eng_map
        self.img_name = img_name

        self.refine_map = np.zeros(img.shape[:2], dtype=np.int32)
        self.is_connect = []

        self.snake_list: list[Snake] = []

        self.snake_valid = []
        self.refine_valid = []
        self.num = 0

        self.iter = 0
        self.max_iter = 0
        self.win = None

        # self.predictor = Direction(self.img)

    # ---------------- 迭代器 ----------------
    def valid_snakes(self):
        return (i for i in range(self.num) if self.snake_valid[i])

    def refine_snakes(self):
        return (i for i in range(self.num) if self.refine_valid[i])

    # ---------------- 操作 ----------------
    def Add(self, ids, pts, dsts):
        for id, pt, dst in zip(ids, pts, dsts):
            coords = np.where(self.ske_map == id)
            self.snake_list.append(Snake(coords, pt, dst, self.img, self.label, self.curve_map, self.ext_eng_map))
            self.snake_valid.append(True)
            self.refine_valid.append(False)
        self.num = len(self.snake_list)

    def clear(self):
        self.snake_list = []
        self.snake_valid = []
        self.refine_valid = []
        self.num = 0

        self.iter = 0
        self.max_iter = 0
        self.win = None

    def Process(self, iter_num, win):
        for i in range(iter_num):
            if np.any(self.snake_valid):
                count = np.count_nonzero(self.snake_valid)
                status = np.where(self.snake_valid, 1, 0)
                print(f"-->iter:{i:02d}/{iter_num}: live_snake = {count:02d} status = {status}")
                self._process_step(i, win)
            self.iter += 1
            # self.visualize_snakes()
            # self.visualize_snakes_v3()
        return self.refine_map

    def process_v2(self, max_iter, win):
        self.max_iter = max_iter
        self.win = win

        for i in range(self.max_iter):
            valid_states = np.array([s.valid for s in self.snake_list])

            if np.any(valid_states):
                count = np.count_nonzero(valid_states)
                status = valid_states.astype(int)
                print(f"-->iter:{self.iter + 1:02d}/{self.max_iter}: live_snake = {count:02d} status = {status}")
                self._process_step_v2()

            # self.visualize_snakes_v2()
            self.visualize_snakes_v2_3()
            self.iter += 1

        return self.refine_map

    def get_foreground_stats(self):
        true_count = sum(snake.in_foreground for snake in self.snake_list)
        false_count = len(self.snake_list) - true_count
        return true_count, false_count

    # ---------------- 计算 ----------------
    def _process_step(self, iter_num, win):
        self._move_snake_head()
        self._move_snake_body()
        self._update_snake(iter_num, win)
        self._update_maps()

    def _process_step_v2(self):
        self._move_snake_head_v2()
        self._move_snake_body_v2()
        self._update_snake_v2()
        # self._update_maps_v2()

        # self._update_maps_v3()
        self._update_maps_v4()

    def _update_snake(self, iter, win):
        [self.snake_list[i].update_snake(iter, win) for i in self.valid_snakes()]

    def _update_snake_v2(self):
        for snake in self.snake_list:
            if snake.valid:
                snake.update_snake_v2(self.iter, self.win)

    def _move_snake_head(self):
        for i in self.valid_snakes():
            valid, refine = self.snake_list[i].move_head()
            if not valid:
                self.snake_valid[i] = False
                if refine:
                    self._check_connect(i)
                    self.refine_valid[i] = True

    def _move_snake_head_v2(self):
        for snake in self.snake_list:
            if snake.valid:
                snake.move_head_v2()

    def _move_snake_body(self):
        for i in self.valid_snakes():
            valid = self.snake_list[i].move_body()
            self.snake_valid[i] = valid

    def _move_snake_body_v2(self):
        for snake in self.snake_list:
            if snake.valid:
                snake.move_body_v2()

    def _check_connect(self, idx):
        c = self.snake_list[idx].get_connect()
        if c in self.is_connect:
            # print(f"❌ connection failed！snake {idx}: {c[0]}-->{c[1]} already exists.")
            return False
        else:
            self.is_connect.append(c)
            # print(f"✅ connection success！ snake{idx}: {c[0]}-->{c[1]}")
            return True

    def _update_maps(self):
        # self.refine_map[:] = 0
        for i in self.refine_snakes():
            self.refine_map[self.snake_list[i].refine_map == 1] = 1

    def _update_maps_v2(self):
        for snake in self.snake_list:
            if snake.refine:
                self.refine_map[snake.refine_map == 1] = 1

    def _update_maps_v3(self):
        for snake in self.snake_list:
            self.refine_map[snake.refine_map == 1] = 1

    def _update_maps_v4(self):
        connection_registry = {}
        self.refine_map[:] = 0

        for snake in self.snake_list:
            this_idx = snake.curve_idx
            target_idx = getattr(snake, 'head_nb_curve_idx', 0)

            allow_refine = True

            if target_idx != 0:
                connection_pair = tuple(sorted([this_idx, target_idx]))

                if connection_pair not in connection_registry:
                    connection_registry[connection_pair] = this_idx
                    allow_refine = True
                else:
                    if connection_registry[connection_pair] == this_idx:
                        allow_refine = True
                    else:
                        allow_refine = False

            if allow_refine:
                self.refine_map[snake.refine_map == 1] = 1

    ################可视化########################################################
    def _generate_snake_info(self, snake_range):
        iter_num = getattr(self, "iter", 0)
        total_iter = getattr(self, "total_iter", 20)
        live_snake = sum(self.snake_valid[: len(snake_range)])
        status_list1 = self.snake_valid[: len(snake_range)]
        status_list2 = self.refine_valid[: len(snake_range)]
        status_str1 = " ".join([str(int(s)) for s in status_list1])
        status_str2 = " ".join([str(int(s)) for s in status_list2])
        info_texts = [f"iter:{iter_num}/{total_iter}: snake = {live_snake:02d} live = [{status_str1}] refine = [{status_str2}]"]
        return info_texts

    def _generate_snake_info_v2(self):
        status_list1 = [s.valid for s in self.snake_list]
        status_list2 = [s.refine for s in self.snake_list]

        live_snake = sum(status_list1)

        status_str1 = " ".join([str(int(s)) for s in status_list1])
        status_str2 = " ".join([str(int(s)) for s in status_list2])

        info_texts = [
            f"iter:{self.iter + 1:02d}/{self.max_iter}: snake = {live_snake:02d} live = [{status_str1}] refine = [{status_str2}]"]
        return info_texts

    def _create_info_bar(self, info_texts, img_width):
        bar_height = 15 * len(info_texts) if info_texts else 15
        bar_img = np.zeros((bar_height, img_width, 3), dtype=np.uint8)
        for idx, text in enumerate(info_texts):
            cv2.putText(bar_img, text, (10, 12 * (idx + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        return bar_img

    def visualize_snakes(self, is_last=False, topk=100):
        window_name = "Snakes (Image Follows Window)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)

        img_left = cv2.cvtColor(self.img.astype(np.uint8), cv2.COLOR_GRAY2BGR) if len(self.img.shape) == 2 else self.img.copy()
        # img_right = cv2.cvtColor(self.curve_map.astype(np.uint8), cv2.COLOR_GRAY2BGR) if len(self.curve_map.shape) == 2 else self.curve_map.copy()
        img_right = (
            cv2.cvtColor((self.ext_eng_map * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) if len(self.curve_map.shape) == 2 else self.curve_map.copy()
        )

        snake_range = range(min(topk, self.num)) if topk is not None and topk > 0 else range(self.num)
        info_texts = self._generate_snake_info(snake_range)

        for i in snake_range:
            snake = self.snake_list[i]
            is_alive = self.snake_valid[i]
            nodes = snake.get_nodes()
            dst = snake.dst

            if len(nodes) > 0:
                head_node = nodes[0]
                head_color = (0, 0, 255)
                y_head, x_head = int(head_node[0]), int(head_node[1])
                cv2.circle(img_left, (x_head, y_head), 1, head_color, -1)
                cv2.circle(img_right, (x_head, y_head), 1, head_color, -1)

                body_nodes = nodes[1:]
                if len(body_nodes) > 0:
                    body_color = (255, 0, 0) if is_alive else (0, 255, 0)
                    for body_node in body_nodes:
                        y_body, x_body = int(body_node[0]), int(body_node[1])
                        if 0 <= y_body < img_left.shape[0] and 0 <= x_body < img_left.shape[1]:
                            img_left[y_body, x_body] = body_color
                            img_right[y_body, x_body] = body_color

            if dst is not None and len(dst) == 2:
                dst_color = (0, 255, 255)
                y_dst, x_dst = int(dst[0]), int(dst[1])
                if 0 <= y_dst < img_left.shape[0] and 0 <= x_dst < img_left.shape[1]:
                    cv2.circle(img_left, (x_dst, y_dst), 1, dst_color, -1)
                    cv2.circle(img_right, (x_dst, y_dst), 1, dst_color, -1)

        combined_img = np.hstack((img_left, img_right))
        info_bar = self._create_info_bar(info_texts, combined_img.shape[1])
        combined_img = np.vstack((combined_img, info_bar))

        cv2.imshow(window_name, combined_img)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                cv2.destroyAllWindows()
                return False
            elif key == 32 or key == 13:
                return True
            elif is_last and key != -1:
                cv2.destroyAllWindows()
                return False

    def visualize_snakes_v2(self):
        img_left = cv2.cvtColor(self.img.astype(np.uint8), cv2.COLOR_GRAY2BGR) if len(
            self.img.shape) == 2 else self.img.copy()
        img_right = cv2.cvtColor(self.curve_map.astype(np.uint8), cv2.COLOR_GRAY2BGR) if len(
            self.curve_map.shape) == 2 else self.curve_map.copy()

        info_texts = self._generate_snake_info_v2()

        for snake in self.snake_list:
            is_alive = snake.valid
            nodes = snake.get_nodes()
            dst = snake.dst

            if len(nodes) > 0:
                head_node = nodes[0]
                y_head, x_head = int(head_node[0]), int(head_node[1])
                head_color = (0, 0, 255)

                img_right = draw_dir(head_node, img_right, snake.dir_pred_16, snake.orig_dir_16)

                cv2.circle(img_left, (x_head, y_head), 1, head_color, -1)
                cv2.circle(img_right, (x_head, y_head), 1, head_color, -1)

                body_nodes = nodes[1:]
                if len(body_nodes) > 0:
                    body_color = (255, 0, 0) if is_alive else (0, 255, 0)

                    for body_node in body_nodes:
                        y_body, x_body = int(body_node[0]), int(body_node[1])

                        if 0 <= y_body < img_left.shape[0] and 0 <= x_body < img_left.shape[1]:
                            img_left[y_body, x_body] = body_color
                            img_right[y_body, x_body] = body_color

            if dst is not None and len(dst) == 2:
                dst_color = (0, 255, 255)
                y_dst, x_dst = int(dst[0]), int(dst[1])

                if 0 <= y_dst < img_left.shape[0] and 0 <= x_dst < img_left.shape[1]:
                    cv2.circle(img_left, (x_dst, y_dst), 1, dst_color, -1)
                    cv2.circle(img_right, (x_dst, y_dst), 1, dst_color, -1)

        combined_img = np.hstack((img_left, img_right))
        info_bar = self._create_info_bar(info_texts, combined_img.shape[1])
        combined_img = np.vstack((combined_img, info_bar))

        cv2.imwrite(os.path.join(add_args.video_path, f"{self.img_name}_{self.iter:03d}.jpg"), combined_img)

    def visualize_snakes_v2_3(self):
        img_top_left = cv2.cvtColor(self.img.astype(np.uint8), cv2.COLOR_GRAY2BGR) if len(
            self.img.shape) == 2 else self.img.copy()
        img_top_right = cv2.cvtColor(self.label.astype(np.uint8), cv2.COLOR_GRAY2BGR) if len(
            self.label.shape) == 2 else self.label.copy()
        img_bottom_left = cv2.cvtColor(self.curve_map.astype(np.uint8), cv2.COLOR_GRAY2BGR) if len(
            self.curve_map.shape) == 2 else self.curve_map.copy()
        img_bottom_right = np.zeros_like(img_top_left)

        info_texts = self._generate_snake_info_v2()

        for snake in self.snake_list:
            nodes = snake.get_nodes()
            dst = snake.dst
            is_alive = snake.valid
            in_foreground = snake.in_foreground

            if len(nodes) > 0:
                head_node = nodes[0]
                y_head, x_head = int(head_node[0]), int(head_node[1])
                head_color = (0, 0, 255)

                # img_bottom_left = draw_dir(head_node, img_bottom_left, snake.prob_pred_16, snake.orig_dir_16)  # fundus
                img_bottom_left = draw_dir_v2(head_node, img_bottom_left, snake.dir_pred_16,
                                              snake.orig_dir_16)  # crack

                cv2.circle(img_top_left, (x_head, y_head), 1, head_color, -1)
                cv2.circle(img_top_right, (x_head, y_head), 1, head_color, -1)
                cv2.circle(img_bottom_left, (x_head, y_head), 1, head_color, -1)

                body_nodes = nodes[1:]
                if len(body_nodes) > 0:
                    color_alive_state = (255, 0, 0) if is_alive else (0, 255, 0)
                    color_fg_state = (255, 255, 0) if in_foreground else (255, 0, 255)

                    for body_node in body_nodes:
                        y_body, x_body = int(body_node[0]), int(body_node[1])

                        if 0 <= y_body < img_top_left.shape[0] and 0 <= x_body < img_top_left.shape[1]:
                            cv2.circle(img_top_left, (x_body, y_body), 1, color_alive_state, -1)
                            cv2.circle(img_top_right, (x_body, y_body), 1, color_fg_state, -1)
                            cv2.circle(img_bottom_left, (x_body, y_body), 1, color_alive_state, -1)

            if dst is not None and len(dst) == 2:
                dst_color = (0, 255, 255)
                y_dst, x_dst = int(dst[0]), int(dst[1])

                if 0 <= y_dst < img_top_left.shape[0] and 0 <= x_dst < img_top_left.shape[1]:
                    cv2.circle(img_top_left, (x_dst, y_dst), 1, dst_color, -1)
                    cv2.circle(img_top_right, (x_dst, y_dst), 1, dst_color, -1)
                    cv2.circle(img_bottom_left, (x_dst, y_dst), 1, dst_color, -1)

        row_top = np.hstack((img_top_left, img_top_right))
        row_bottom = np.hstack((img_bottom_left, img_bottom_right))
        grid_img = np.vstack((row_top, row_bottom))

        info_bar = self._create_info_bar(info_texts, grid_img.shape[1])
        combined_img = np.vstack((grid_img, info_bar))

        save_dir = os.path.join(add_args.video_path, self.img_name)
        os.makedirs(save_dir, exist_ok=True)

        file_name = f"{self.img_name}_{self.iter:03d}.jpg"
        save_path = os.path.join(save_dir, file_name)

        cv2.imwrite(save_path, combined_img)

    def visualize_snakes_v3(self, topk=None):
        img_left = cv2.cvtColor(self.img.astype(np.uint8), cv2.COLOR_GRAY2BGR) if len(
            self.img.shape) == 2 else self.img.copy()
        img_right = cv2.cvtColor(self.curve_map.astype(np.uint8), cv2.COLOR_GRAY2BGR) if len(
            self.curve_map.shape) == 2 else self.curve_map.copy()

        snake_range = range(min(topk, self.num)) if topk is not None and topk > 0 else range(self.num)
        info_texts = self._generate_snake_info(snake_range)

        for i in snake_range:
            snake = self.snake_list[i]
            is_alive = self.snake_valid[i]
            nodes = snake.get_nodes()
            dst = snake.dst

            if len(nodes) > 0:
                head_node = nodes[0]

                # img_left = draw_dir(self.predictor, head_node, img_left)
                img_left = draw_dir(head_node, img_left, snake.dir_pred_16, snake.orig_dir_16)

                head_color = (0, 0, 255)
                y_head, x_head = int(head_node[0]), int(head_node[1])
                cv2.circle(img_left, (x_head, y_head), 1, head_color, -1)
                cv2.circle(img_right, (x_head, y_head), 1, head_color, -1)

                body_nodes = nodes[1:]
                if len(body_nodes) > 0:
                    body_color = (255, 0, 0) if is_alive else (0, 255, 0)
                    for body_node in body_nodes:
                        y_body, x_body = int(body_node[0]), int(body_node[1])
                        if 0 <= y_body < img_left.shape[0] and 0 <= x_body < img_left.shape[1]:
                            img_left[y_body, x_body] = body_color
                            img_right[y_body, x_body] = body_color

            if dst is not None and len(dst) == 2:
                dst_color = (0, 255, 255)
                y_dst, x_dst = int(dst[0]), int(dst[1])
                if 0 <= y_dst < img_left.shape[0] and 0 <= x_dst < img_left.shape[1]:
                    cv2.circle(img_left, (x_dst, y_dst), 1, dst_color, -1)
                    cv2.circle(img_right, (x_dst, y_dst), 1, dst_color, -1)

        combined_img = np.hstack((img_left, img_right))
        info_bar = self._create_info_bar(info_texts, combined_img.shape[1])
        combined_img = np.vstack((combined_img, info_bar))

        cv2.imwrite(os.path.join(root_path, "Output\\video", f"image_{self.iter:03d}.jpg"), combined_img)

import numpy as np
import copy

from .curve_dst import get_circle
from .setting import args
from ..my_addition import (add_args, cal_head_angle_v2, get_optimal_angle_idx_v2, get_optimal_angle_idx_v3,
                           get_optimal_angle_idx_v4, get_optimal_angle_idx_v5, get_optimal_angle_idx_v6,
                           get_optimal_angle_idx_v7)
from ..my_addition.Model import get_pred_direction_v2, get_pred_direction_v3, agg_16_to_8


class Snake:
    def __init__(self, coords, nodes, dst, img, label, curve_map, ext_eng_map):
        self.img = img
        self.label = label
        self.curve_map = curve_map
        self.ext_eng_map = ext_eng_map

        self.dst = tuple(dst)
        self.snake_coords = coords
        self.curve_idx = self.curve_map[tuple(nodes[0, :])]
        self.dst_curve_idx = self.curve_map[self.dst]

        self.row_diff = [0, -1, -1, -1, 0, 1, 1, 1]
        self.col_diff = [1, 1, 0, -1, -1, -1, 0, 1]
        self.diff = np.stack([self.row_diff, self.col_diff], axis=1)

        self.max_length = 512
        self.nodes = np.zeros((self.max_length, 2), dtype=np.int32)
        self.new_nodes = np.zeros((self.max_length, 2), dtype=np.int32)
        self.nb = np.zeros((self.max_length, 8, 2), dtype=np.int32)
        self.num = min(nodes.shape[0], self.max_length)
        self.nodes[: self.num] = nodes[: self.num]
        self.nb = self.nodes[:, np.newaxis, :] + self.diff

        self.dist = np.linalg.norm(self.nodes[0] - self.dst)

        self.refine_map = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)
        self.refine_prob_map = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

        self.similarity = self._cal_similarity()

        self.valid = True
        self.refine = True
        self.in_foreground = True

        self.dir_pred_16 = np.zeros(16, dtype=np.int32)
        self.orig_dir_16 = None
        self.head_nb_curve_idx = 0

    def get_nodes(self):
        return self.nodes[: self.num]

    def update_snake(self, iteration, win):
        if self.num < self.max_length and (iteration + 1) % win == 0:
            self.nodes[self.num, :] = self.nodes[self.num - 1, :]  # 强制变多
            # print(f"迭代{iteration}：snake节点数从{self.num}增加到{self.num + 1}")
            self.num += 1

        self._clip_snake()
        self.nb = self.nodes[:, np.newaxis, :] + self.diff  # need clip?
        self._update_map()

    def update_snake_v2(self, iteration, win):
        self.nodes[:self.num - 1] = self.new_nodes[:self.num - 1]
        self.new_nodes[:] = 0
        self._update_map()

        """new iteration"""
        if self.num < self.max_length and (iteration + 1) % win == 0:
            self.nodes[self.num, :] = self.nodes[self.num - 1, :]
            self.num += 1

        self.nb = self.nodes[:, np.newaxis, :] + self.diff
        # self._update_status()
        self._update_status_v2()

    def get_connect(self):
        connect = tuple(sorted((self.curve_idx, self.dst_curve_idx)))
        return connect

    def _cal_eng(self, idx, dirs=None, gra_valid=True):
        int_eng = self._cal_int_eng(idx)
        ext1_eng = self._cal_ext1_eng(idx)
        ext2_eng = self._cal_ext2_eng(idx)
        rep_eng = -self._cal_rep_eng(idx)
        eng = int_eng + ext1_eng + ext2_eng + rep_eng

        if idx == 0 and gra_valid:  # head node
            gra_eng = self._cal_gra_eng(idx)
            eng += gra_eng

        if dirs is not None:  # only some directions
            eng = eng[dirs]

        return eng

    def _cal_eng_v2(self, idx, dirs=None):
        int_eng = self._cal_int_eng_v2(idx)
        ext1_eng = self._cal_ext1_eng(idx)

        if add_args.ext_2_valid:
            ext2_eng = self._cal_ext2_eng(idx)
        else:
            ext2_eng = np.zeros(8, dtype=np.float32)

        if add_args.repel_valid and idx == 0:
            rep_eng = -self._cal_rep_eng_v2()
        else:
            rep_eng = np.zeros(8, dtype=np.float32)

        if add_args.gra_valid and idx == 0:
            gra_eng = self._cal_gra_eng(idx)
        else:
            gra_eng = np.zeros(8, dtype=np.float32)

        eng = int_eng + ext1_eng + ext2_eng + rep_eng + gra_eng

        if dirs is not None:
            eng = eng[dirs]

        return eng

    def _cal_int_eng(self, idx):
        num = self.num
        y_nb, x_nb = self.nb[idx].T
        p2, p1, n1, n2 = [(idx + i) % num for i in (-2, -1, 1, 2)]
        y_p2, x_p2 = self.nodes[p2]
        y_p1, x_p1 = self.nodes[p1]
        y_n1, x_n1 = self.nodes[n1]
        y_n2, x_n2 = self.nodes[n2]

        d1_p = (y_p1 - y_nb) ** 2 + (x_p1 - x_nb) ** 2
        d1_n = (y_nb - y_n1) ** 2 + (x_nb - x_n1) ** 2

        if idx == 0:
            part_1st = d1_n
        elif 0 < idx < num - 1:
            part_1st = d1_n + d1_p
        else:
            part_1st = d1_p

        d2_p = (y_p2 - 2 * y_p1 + y_nb) ** 2 + (x_p2 - 2 * x_p1 + x_nb) ** 2
        d2_c = (y_p1 - 2 * y_nb + y_n1) ** 2 + (x_p1 - 2 * x_nb + x_n1) ** 2
        d2_n = (y_nb - 2 * y_n1 + y_n2) ** 2 + (x_nb - 2 * x_n1 + x_n2) ** 2

        if idx == 0:
            part_2nd = d2_n
        elif idx == 1:
            part_2nd = d2_c + d2_n
        elif 1 < idx < num - 2:
            part_2nd = d2_p + d2_c + d2_n
        elif idx == num - 2:
            part_2nd = d2_p + d2_c
        else:
            part_2nd = d2_p
        int_eng = args.alpha * part_1st + args.beta * part_2nd
        return int_eng

    def _cal_int_eng_v2(self, idx):
        num = self.num

        if not (0 < idx < num - 1):
            return np.zeros(8, dtype=np.float32)

        nb = self.nb[idx, :, :]
        y_nb, x_nb = nb[:, 0], nb[:, 1]
        p2, p1, n1, n2 = [(idx + i) % num for i in (-2, -1, 1, 2)]

        y_p2, x_p2 = self.nodes[p2]
        y_p1, x_p1 = self.nodes[p1]
        y_n1, x_n1 = self.nodes[n1]
        y_n2, x_n2 = self.nodes[n2]

        """first part"""
        d1_p = (y_p1 - y_nb) ** 2 + (x_p1 - x_nb) ** 2
        d1_n = (y_nb - y_n1) ** 2 + (x_nb - x_n1) ** 2
        part_1st = d1_n + d1_p

        """second part"""
        d2_p = (y_p2 - 2 * y_p1 + y_nb) ** 2 + (x_p2 - 2 * x_p1 + x_nb) ** 2
        d2_c = (y_p1 - 2 * y_nb + y_n1) ** 2 + (x_p1 - 2 * x_nb + x_n1) ** 2
        d2_n = (y_nb - 2 * y_n1 + y_n2) ** 2 + (x_nb - 2 * x_n1 + x_n2) ** 2

        if idx == 1:
            part_2nd = d2_c + d2_n
        elif idx == num - 2:
            part_2nd = d2_p + d2_c
        else:
            part_2nd = d2_p + d2_c + d2_n

        """total"""
        int_eng = args.alpha * part_1st + args.beta * part_2nd

        return int_eng

    def _cal_ext1_eng(self, idx):
        nb_r, nb_c = self.nb[idx, :, [0, 1]]
        ext_eng = args.gamma * self.ext_eng_map[nb_r, nb_c]
        return ext_eng

    def _cal_ext2_eng(self, idx, eps=1e-8):
        gap = 45
        sec_ang = np.arange(0, 360, gap)
        end_ang = np.arange(0 + gap, 360 + gap, gap)
        sec_num = sec_ang.size

        row, col = self.nodes[idx]
        area = self._get_circle(row, col, args.radius)
        row_diff, col_diff = area[:, 0] - row, area[:, 1] - col

        angles = np.degrees(np.arctan2(-row_diff, col_diff)) % 360.0
        ds, de = angles - sec_ang[:, np.newaxis], end_ang[:, np.newaxis] - angles
        inv_dist = 1.0 / np.sqrt(row_diff ** 2 + col_diff ** 2 + eps)

        ang_mask = np.logical_and(ds >= 0, de > 0)
        cos_ds = np.cos(np.radians(ds))
        cos_de = np.cos(np.radians(de))
        w1 = cos_ds * inv_dist * ang_mask
        w2 = cos_de * inv_dist * ang_mask

        count = np.sum(w1, axis=1)
        count[(np.arange(sec_num) + 1) % sec_num] += np.sum(w2, axis=1)

        sim_vals = self.similarity[area[:, 0], area[:, 1]]
        sim_sum = np.sum(w1 * sim_vals[np.newaxis, :], axis=1)
        sim_sum[(np.arange(sec_num) + 1) % sec_num] += np.sum(w2 * sim_vals[np.newaxis, :], axis=1)
        sim_avg = args.zeta * sim_sum / (count + eps)

        return sim_avg

    def _cal_rep_eng(self, idx):
        num = self.num
        y_nb, x_nb = self.nb[idx].T
        p1, n1 = [(idx + i) % num for i in (-1, 1)]
        y_p1, x_p1 = self.nodes[p1]
        y_n1, x_n1 = self.nodes[n1]
        d1_p = np.hypot(y_p1 - y_nb, x_p1 - x_nb)
        d1_n = np.hypot(y_nb - y_n1, x_nb - x_n1)

        if idx == 0:
            part_1st = d1_n
        elif 0 < idx < num - 1:
            part_1st = d1_n + d1_p
        else:
            part_1st = d1_p

        rep_eng = args.delta * part_1st
        return rep_eng

    def _cal_rep_eng_v2(self):
        """only head"""
        nb = self.nb[0, :, :]
        y_nb, x_nb = nb[:, 0], nb[:, 1]
        y_n, x_n = self.nodes[1]

        d1_n = np.hypot(y_n - y_nb, x_n - x_nb)
        rep_eng = args.delta * d1_n

        return rep_eng

    def _cal_gra_eng(self, idx):
        diff = self.nb[idx, :, :] - self.dst
        dist = np.linalg.norm(diff, axis=1)
        return args.eta * dist

    def _cal_similarity(self):
        avg_clr = np.mean(self.img[self.snake_coords], axis=0)
        diff = self.img - avg_clr
        similarity = np.linalg.norm(diff, axis=2)
        return similarity

    def move_head(self):
        new_dir, refine_valid = self._cal_head_dir(0)
        if new_dir >= 0:
            self._move_new(0, new_dir)
            return True, False
        else:
            return False, refine_valid

    def move_head_v2(self):
        if self._check_point_at_bound(0):
            return

        # new_dir = self._cal_head_dir_v2()
        new_dir = self._cal_head_dir_v3()

        self._buffer_new(0, new_dir)
        # self._update_status()

    def move_body(self):
        for i in range(1, self.num - 1):
            new_dir = self._move_body(i)
            if new_dir < 0:
                return False
        return True

    def move_body_v2(self):
        for i in range(1, self.num - 1):
            if not self.valid:
                return
            if self._check_point_at_bound(i):
                return

            # new_dir = self._cal_body_dir(i)
            new_dir = self._cal_body_dir_v2(i)

            self._buffer_new(i, new_dir)
        return

    def _move_body(self, idx):
        new_dir = self._cal_body_dir(idx)
        if new_dir >= 0:
            self._move_new(idx, new_dir)
        return new_dir

    def _cal_head_dir(self, idx):
        int_eng = self._cal_int_eng(idx)
        ext1_eng = self._cal_ext1_eng(idx)
        ext2_eng = self._cal_ext2_eng(idx)
        gra_eng = self._cal_gra_eng(idx)
        rep_eng = -self._cal_rep_eng(idx)
        eng = int_eng + ext1_eng + gra_eng + ext2_eng + rep_eng
        min_idx = np.argmin(eng)

        if not self._is_move_in_bounds(idx, min_idx):
            return -1, False

        snake_valid, refine_valid = self._is_snake_valid()
        return (min_idx, False) if snake_valid else (-1, refine_valid)

    def _cal_head_dir_v2(self):
        # prob_pred_16 = get_pred_direction_v2(self.predictor, [self.nodes[0, :]], use_far=False)
        # dir_pred_16, _ = get_pred_direction_v3(self.predictor, [self.nodes[0, :]], use_far=False)
        # self.dir_pred_16 = dir_pred_16.copy()
        dir_pred_16 = self.dir_pred_16

        head_angle = cal_head_angle_v2(self.nodes, self.num)

        # min_dir, orig_idx = get_optimal_angle_idx_v2(prob_pred_16, head_angle, 3)

        # dirs, orig_idx = get_optimal_angle_idx_v3(prob_pred_16, head_angle, 3)

        # min_dir, orig_idx = get_optimal_angle_idx_v4(prob_pred_16, head_angle, 3)

        # min_dir, orig_idx = get_optimal_angle_idx_v5(
        #     prob_pred_16, head_angle, 3, self.nodes[0, :], np.array(self.dst), self.dist)

        # min_dir, orig_idx = get_optimal_angle_idx_v6(
        #     prob_pred_16, head_angle, 3, self.nodes[0, :], np.array(self.dst), self.dist)

        min_dir, orig_idx = get_optimal_angle_idx_v7(
            dir_pred_16, head_angle, self.nodes[0, :], np.array(self.dst), self.dist)

        self.orig_dir_16 = copy.deepcopy(orig_idx)

        return min_dir

    def _cal_head_dir_v3(self):
        eng = self._cal_eng_v2(0)
        min_idx = np.argmin(eng)
        return min_idx

    def _cal_body_dir(self, idx):
        h, w = self.img.shape[:2]
        cur_r, cur_c = self.nodes[idx]

        if cur_r < 1 or cur_r >= h - 1 or cur_c < 1 or cur_c >= w - 1:
            return -1

        int_eng = self._cal_int_eng(idx)
        ext1_eng = self._cal_ext1_eng(idx)
        ext2_eng = self._cal_ext2_eng(idx)
        rep_eng = -self._cal_rep_eng(idx)

        """ablation study"""
        # ext2_eng[:] = 0
        # rep_eng[:] = 0

        eng = int_eng + ext1_eng + ext2_eng + rep_eng
        min_dir = np.argmin(eng)
        return min_dir

    def _cal_body_dir_v2(self, idx):
        eng = self._cal_eng_v2(idx)
        min_idx = np.argmin(eng)
        return min_idx

    def _is_snake_valid(self):
        rows, cols = self.nb[0, :, [0, 1]]
        idxs = np.unique(self.curve_map[rows, cols])

        snake_valid = not any(0 < idx != self.curve_idx for idx in idxs)
        refine_valid = self.dst_curve_idx in idxs
        return snake_valid, refine_valid

    def _update_status(self):
        head_y, head_x = self.nodes[0, 0], self.nodes[0, 1]
        rows, cols = self.nb[0, :, 0], self.nb[0, :, 1]
        h, w = self.label.shape

        if not self._check_point_at_bound(0):
            idxs = np.unique(self.curve_map[rows, cols])

            if any(0 < idx != self.curve_idx for idx in idxs):
                self.valid = False
            self.refine = not self.valid

        if 0 <= head_y < h and 0 <= head_x < w:
            self.in_foreground = self.label[head_y, head_x] > 0
        else:
            self.in_foreground = False

    def _update_status_v2(self):
        head_y, head_x = self.nodes[0, 0], self.nodes[0, 1]
        rows, cols = self.nb[0, :, 0], self.nb[0, :, 1]
        h, w = self.label.shape

        if not self._check_point_at_bound(0):
            idxs = np.unique(self.curve_map[rows, cols])
            non_zero_idxs = idxs[idxs != 0]
            if len(non_zero_idxs) > 0:
                self.head_nb_curve_idx = non_zero_idxs[0]

            if any(0 < idx != self.curve_idx for idx in idxs):
                self.valid = False
            self.refine = not self.valid

        if 0 <= head_y < h and 0 <= head_x < w:
            self.in_foreground = self.label[head_y, head_x] > 0
        else:
            self.in_foreground = False

    def _is_move_in_bounds(self, idx, dirs, margin=2):
        r, c = self.nodes[idx]
        nr, nc = r + self.row_diff[dirs], c + self.col_diff[dirs]
        h, w = self.img.shape[:2]
        return margin <= nr < h - margin and margin <= nc < w - margin

    def _check_point_at_bound(self, idx, margin=1):
        if self._is_locate_in_bounds(idx, margin):
            self.valid = False
            self.refine = False
            return True
        else:
            return False

    def _is_locate_in_bounds(self, idx, margin=1):
        r, c = self.nodes[idx]
        h, w = self.img.shape[:2]
        return r < margin or r >= h - margin or c < margin or c >= w - margin

    def _clip_snake(self, MARGIN=2):
        clip_min = [MARGIN, MARGIN]
        clip_max = [self.img.shape[0] - (MARGIN + 1), self.img.shape[1] - (MARGIN + 1)]
        self.nodes[: self.num] = np.clip(self.nodes[: self.num], clip_min, clip_max)

    def _move_new(self, idx, dir):
        self.nodes[idx, :] = self.nb[idx, dir, :]

    def _buffer_new(self, idx, dir):
        self.new_nodes[idx, :] = self.nb[idx, dir, :]

    def _update_map(self):
        row, col = self.nodes[0, :]
        self.refine_map[row, col] = 1
        self.refine_prob_map[row, col] = 1

    def _get_circle(self, y, x, r):
        return get_circle(self.curve_map, y, x, r)

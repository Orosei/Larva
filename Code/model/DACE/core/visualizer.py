import os

import cv2
import numpy as np


class TraceVisualizer:
    TRACE_WIDTH = 2
    ZOOM_TRACE_WIDTH = 1
    ZERO_WIDTH_THRESHOLD = 1
    ZOOM_SIZE = 128
    ZOOM_SCALE = 2
    BAR_HEIGHT = 80
    LINE_HEIGHT = 28

    _COLOR = {
        "color_bg": (128, 128, 128),
        "color_label": (0, 255, 0),
        "color_traj_label": (255, 0, 0),
        "color_traj_image": (255, 0, 0),
        "color_box": (0, 255, 255),
        "color_text": (0, 255, 0),
        "color_zero_width": (0, 0, 255),
    }

    # ── constructor ──────────────────────────────────────────────

    def __init__(self, image, label, save_dir, image_path, write_video=True, show_gui=True, parallel=False):
        self.save_dir = save_dir
        self.image_path = os.path.abspath(image_path)
        self.write_video = write_video
        self.show_gui = show_gui
        self.parallel = parallel
        if parallel:
            self.BAR_HEIGHT = 0
        self.window_name = f"Dace - {self.image_path}"
        self.image = image.copy()
        self.label = label.copy() if label is not None else None

        if min(self.image.shape[:2]) < 400:
            self.ZOOM_SCALE = 1

        self.video_writer = None
        if self.write_video:
            self._init_video_recorder()
        if self.show_gui:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 800, 600)

    def _init_video_recorder(self):
        os.makedirs(self.save_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(self.image_path))[0]
        video_path = os.path.abspath(os.path.join(self.save_dir, f"{stem}.mp4"))
        print(f"📹 视频保存到: {video_path}")

        h, w = self.image.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(video_path, fourcc, 10.0, (w * 2, h + self.BAR_HEIGHT))

    # ── public ────────────────────────────────────────────────────

    def update_display(self, traj, reason="", metrics=None, widths=None):
        score = metrics.last_str() if metrics else ""
        msg = f"steps:{len(traj)} | reason:{reason} | score:{score}"
        if widths and len(widths) > 0:
            msg = f"width:{widths[-1]:.2f} | " + msg

        self._last_left = self._create_left_panel(traj, widths)
        self._last_right = self._create_right_panel(traj, widths)
        final_img = self._compose_final_image(self._last_left, self._last_right)
        self._draw_text_area(final_img, msg)
        self._show_and_write(final_img)
        self.last_frame = final_img

    def close(self):
        if not self.parallel:
            self._save_last_image()
        else:
            stem = os.path.splitext(os.path.basename(self.image_path))[0]
            cv2.imwrite(os.path.join(self.save_dir, f"{stem}_final.png"), self.last_frame)
        if self.video_writer is not None:
            self.video_writer.release()
        if self.show_gui:
            cv2.destroyAllWindows()

    def _save_last_image(self):
        stem = os.path.splitext(os.path.basename(self.image_path))[0]
        cv2.imwrite(os.path.join(self.save_dir, f"{stem}_final.png"), self.last_frame)
        cv2.imwrite(os.path.join(self.save_dir, f"{stem}_panels.png"), np.hstack([self._last_left, self._last_right]))
        lz = self._last_left_zoom.copy()
        rz = self._last_right_zoom.copy()
        cv2.rectangle(lz, (0, 0), (lz.shape[1] - 1, lz.shape[0] - 1), self._COLOR["color_box"], 1)
        cv2.rectangle(rz, (0, 0), (rz.shape[1] - 1, rz.shape[0] - 1), self._COLOR["color_box"], 1)
        cv2.imwrite(os.path.join(self.save_dir, f"{stem}_zoom.png"), np.hstack([lz, rz]))
        print(f"🖼️ 图片已保存至: {self.save_dir}")

    # ── private: panels ────────────────────────────────────────────

    def _create_left_panel(self, traj, widths):
        return self._draw_panel(self.image.copy(), traj, "color_traj_image", widths)

    def _create_right_panel(self, traj, widths):
        if self.label is None:
            canvas = self.image.copy()
        else:
            h, w = self.image.shape[:2]
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            canvas[self.label > 0] = self._COLOR["color_label"]
        return self._draw_panel(canvas, traj, "color_traj_label", widths, with_label=True)

    def _draw_segments(self, canvas, traj, widths, color, offset=(0, 0)):
        """Draw traj segments with per-point widths.  width < threshold → zero-width color."""
        dx, dy = offset
        for i in range(1, len(traj)):
            pt1 = (int(traj[i - 1][0]) - dx, int(traj[i - 1][1]) - dy)
            pt2 = (int(traj[i][0]) - dx, int(traj[i][1]) - dy)
            if widths[i] < self.ZERO_WIDTH_THRESHOLD:
                cv2.line(canvas, pt1, pt2, self._COLOR["color_zero_width"], 1)
            else:
                w = max(1, int(round(widths[i])))
                cv2.line(canvas, pt1, pt2, color, w)

    def _draw_panel(self, canvas, traj, color_key, widths, with_label=False):
        if len(traj) > 1:
            self._draw_segments(canvas, traj, widths, self._COLOR[color_key])
            self._draw_zoom(canvas, traj, color_key, widths, with_label)
        return canvas

    # ── private: zoom ──────────────────────────────────────────────

    def _draw_zoom(self, img, traj, color_key, widths, with_label=False):
        if len(traj) < 1 or (with_label and self.label is None):
            return
        cx, cy = int(traj[-1][0]), int(traj[-1][1])
        r = self.ZOOM_SIZE // 2
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2 = min(self.image.shape[1], cx + r)
        y2 = min(self.image.shape[0], cy + r)

        if with_label:
            roi = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8)
            roi[self.label[y1:y2, x1:x2] > 0] = self._COLOR["color_label"]
        else:
            roi = self.image[y1:y2, x1:x2].copy()

        self._draw_segments(roi, traj, widths, self._COLOR[color_key], offset=(x1, y1))
        zoomed = cv2.resize(roi, None, fx=self.ZOOM_SCALE, fy=self.ZOOM_SCALE, interpolation=cv2.INTER_NEAREST)
        if with_label:
            self._last_right_zoom = zoomed
        else:
            self._last_left_zoom = zoomed
        self._place_zoom(img, zoomed, cx, cy)

    def _place_zoom(self, img, zoomed, cx, cy):
        zh, zw = zoomed.shape[:2]
        ih, iw = img.shape[:2]

        px = 10 if cx >= iw // 2 else iw - zw - 10
        py = 10 if cy >= ih // 2 else ih - zh - 10

        img[py : py + zh, px : px + zw] = zoomed
        cv2.rectangle(img, (px, py), (px + zw, py + zh), self._COLOR["color_box"], 1)

    # ── private: compose & output ─────────────────────────────────

    def _compose_final_image(self, left, right):
        h, w = self.image.shape[:2]
        final = np.full((h + self.BAR_HEIGHT, w * 2, 3), self._COLOR["color_bg"], dtype=np.uint8)
        final[:h, :w] = left
        final[:h, w : w * 2] = right
        return final

    def _draw_text_area(self, final, text):
        fh, fw = final.shape[:2]
        y = fh - self.BAR_HEIGHT + self.LINE_HEIGHT
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 0.8
        thick = 2
        margin = 20
        max_width = fw - 2 * margin

        text = text.replace("\n", " ")
        current_line = ""
        for word in text.split():
            test_line = current_line + (" " if current_line else "") + word
            w_test, _ = cv2.getTextSize(test_line, font, scale, thick)[0]
            if w_test <= max_width:
                current_line = test_line
            else:
                cv2.putText(final, current_line, (margin, y), font, scale, self._COLOR["color_text"], thick)
                current_line = word
                y += self.LINE_HEIGHT
        if current_line:
            cv2.putText(final, current_line, (margin, y), font, scale, self._COLOR["color_text"], thick)

    def _show_and_write(self, final):
        if self.video_writer is not None:
            self.video_writer.write(final)

        if self.show_gui:
            cv2.imshow(self.window_name, final)
            cv2.waitKey(1)

    _PALETTE = [
        (255, 0, 0),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 255),
        (255, 128, 0),
        (0, 128, 255),
        (255, 80, 80),
        (80, 80, 255),
        (255, 255, 128),
        (255, 128, 255),
    ]

    def display_all(self, trajs, widths_list):
        left = self.image.copy()
        right = np.zeros_like(self.image) if self.label is not None else self.image.copy()
        if self.label is not None:
            right[self.label > 0] = self._COLOR["color_label"]

        for i, t in enumerate(trajs):
            if len(t) < 2:
                continue
            c = self._PALETTE[i % len(self._PALETTE)]
            widths = widths_list[i] if i < len(widths_list) else None
            self._draw_segments(left, t, widths, c)
            self._draw_segments(right, t, widths, c)

        final_img = self._compose_final_image(left, right)
        self._show_and_write(final_img)
        self.last_frame = final_img

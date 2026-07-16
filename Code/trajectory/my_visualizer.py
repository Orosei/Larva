"""Three-panel point visualization that accumulates every trajectory."""

import os

import cv2
import numpy as np

from ..model.DACE.core.visualizer import TraceVisualizer as BaseTraceVisualizer


class MyTraceVisualizer(BaseTraceVisualizer):
    """Render original image, coarse segmentation, and label with all trajectory points."""

    def __init__(
        self,
        image,
        label,
        save_dir,
        image_path,
        write_video=True,
        show_gui=True,
        parallel=False,
        coarse_seg=None,
    ):
        if coarse_seg is None:
            raise ValueError("coarse_seg is required for MyTraceVisualizer")
        if np.asarray(coarse_seg).shape[:2] != image.shape[:2]:
            raise ValueError(
                "image and coarse_seg shapes must match: "
                f"image={image.shape[:2]}, coarse_seg={np.asarray(coarse_seg).shape[:2]}"
            )

        self.trajs = []
        self.coarse_seg = np.asarray(coarse_seg).copy()
        super().__init__(
            image,
            label,
            save_dir,
            image_path,
            write_video=write_video,
            show_gui=show_gui,
            parallel=parallel,
        )

    def _init_video_recorder(self):
        """Create a video writer sized for the three-panel live frame."""
        os.makedirs(self.save_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(self.image_path))[0]
        video_path = os.path.abspath(os.path.join(self.save_dir, f"{stem}.mp4"))
        print(f"📹 视频保存到: {video_path}")

        height, width = self.image.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            video_path,
            fourcc,
            10.0,
            (width * 3, height + self.BAR_HEIGHT),
        )

    @staticmethod
    def _copy_traj(traj):
        """Copy one trajectory as an ordered list of xy coordinates."""
        points = np.asarray(traj, dtype=np.float32)
        if points.size == 0:
            return []
        points = points.reshape(-1, points.shape[-1])
        return [tuple(point[:2]) for point in points]

    @staticmethod
    def _draw_traj_points(canvas, trajs, color):
        """Draw only valid trajectory coordinates, with one pixel per point."""
        height, width = canvas.shape[:2]
        for traj in trajs:
            if not traj:
                continue

            points = np.rint(np.asarray(traj, dtype=np.float32)).astype(np.int32)
            xs = points[:, 0]
            ys = points[:, 1]
            valid = (0 <= xs) & (xs < width) & (0 <= ys) & (ys < height)
            canvas[ys[valid], xs[valid]] = color
        return canvas

    def _create_all_panels(self):
        left = self._draw_traj_points(
            self.image.copy(),
            self.trajs,
            self._COLOR["color_traj_image"],
        )

        middle = np.zeros_like(self.image)
        middle[self.coarse_seg > 0] = self._COLOR["color_label"]
        middle = self._draw_traj_points(
            middle,
            self.trajs,
            self._COLOR["color_traj_label"],
        )

        right = np.zeros_like(self.image)
        if self.label is not None:
            right[self.label > 0] = self._COLOR["color_label"]
        right = self._draw_traj_points(
            right,
            self.trajs,
            self._COLOR["color_traj_label"],
        )
        return left, middle, right

    def _compose_three_panels(self, left, middle, right):
        height, width = self.image.shape[:2]
        final = np.full(
            (height + self.BAR_HEIGHT, width * 3, 3),
            self._COLOR["color_bg"],
            dtype=np.uint8,
        )
        final[:height, :width] = left
        final[:height, width : width * 2] = middle
        final[:height, width * 2 : width * 3] = right
        return final

    def _render_all(self, reason="", metrics=None):
        self._last_left, self._last_middle, self._last_right = self._create_all_panels()
        final_img = self._compose_three_panels(
            self._last_left,
            self._last_middle,
            self._last_right,
        )

        if self.BAR_HEIGHT > 0:
            score = metrics.last_str() if metrics else ""
            point_count = sum(len(traj) for traj in self.trajs)
            msg = (
                f"trajs:{len(self.trajs)} | points:{point_count} "
                f"| reason:{reason} | score:{score}"
            )
            self._draw_text_area(final_img, msg)

        self._show_and_write(final_img)
        self.last_frame = final_img

    def update_display(self, traj, reason="", metrics=None, widths=None):
        """Accumulate one trajectory and render all trajectories received so far."""
        del widths
        self.trajs.append(self._copy_traj(traj))
        self._render_all(reason, metrics)

    def display_all(self, trajs, widths_list=None):
        """Replace the current trajectories and render them together."""
        del widths_list
        self.trajs = [self._copy_traj(traj) for traj in trajs]
        self._render_all()

    def close(self):
        """Save the final coarse-seg view and three-panel frame."""
        if not hasattr(self, "last_frame"):
            self._render_all()

        os.makedirs(self.save_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(self.image_path))[0]
        cv2.imwrite(
            os.path.join(self.save_dir, f"{stem}_coarse_seg_traj.png"),
            self._last_middle,
        )
        cv2.imwrite(
            os.path.join(self.save_dir, f"{stem}_panels.png"),
            self.last_frame,
        )

        if self.video_writer is not None:
            self.video_writer.release()
        if self.show_gui:
            cv2.destroyAllWindows()

        print(f"🖼️ 图片已保存至: {self.save_dir}")


TraceVisualizer = MyTraceVisualizer

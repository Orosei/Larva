import numpy as np


class SkeletonWalker:

    _CARDINAL = np.array([(-1, 0), (0, -1), (0, 1), (1, 0)])
    _DIAGONAL = np.array([(-1, -1), (-1, 1), (1, -1), (1, 1)])

    LOOKAHEAD = 12
    NUM_PATHS = 10

    def __init__(self, skeleton, max_len=200):
        self.skeleton = skeleton
        self.max_len = max_len
        self.H, self.W = skeleton.shape

    def _gather(self, p, visited):
        x, y = int(p[0]), int(p[1])
        result = []
        for offsets in (self._CARDINAL, self._DIAGONAL):
            for dx, dy in offsets:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.W and 0 <= ny < self.H and self.skeleton[ny, nx]:
                    nxt = np.array([nx, ny])
                    if tuple(nxt) not in visited:
                        result.append(nxt)
        return result

    def _walk_path(self, start, visited, steps=5):
        path = []
        cur = start
        local_seen = set(tuple(p) for p in visited)
        for _ in range(steps):
            neighbors = self._gather(cur, local_seen)
            if not neighbors:
                # self._debug_dead_end(cur, local_seen, visited)
                break
            nxt = neighbors[np.random.choice(len(neighbors))]
            path.append(nxt)
            local_seen.add(tuple(nxt))
            cur = nxt
        return path

    def extract(self, start):
        current = np.array(start, dtype=np.int32)
        traj = [current]
        visited = {tuple(current)}

        while len(traj) < self.max_len:
            paths = [self._walk_path(current, visited, self.LOOKAHEAD) for _ in range(self.NUM_PATHS)]
            paths = [p for p in paths if p]
            if not paths:
                break

            tip = np.array(traj[-1], dtype=np.float32)
            best = max(paths, key=lambda p: np.linalg.norm(p[-1] - tip))

            # self._debug_vis_paths(self.skeleton, traj, paths, best)

            for pt in best:
                traj.append(pt)
                visited.add(tuple(pt))
                current = pt
        return traj

    def _debug_dead_end(self, cur, local_seen, visited):
        x, y = int(cur[0]), int(cur[1])
        nb = []
        for offsets in (self._CARDINAL, self._DIAGONAL):
            for dx, dy in offsets:
                nx, ny = x + dx, y + dy
                in_bounds = 0 <= nx < self.W and 0 <= ny < self.H
                on_skel = in_bounds and bool(self.skeleton[ny, nx])
                in_local = (nx, ny) in local_seen
                in_visited = (nx, ny) in visited
                nb.append((dx, dy, on_skel, in_local, in_visited))
        print(f"dead end: cur=({x},{y}) local={len(local_seen)} visited={len(visited)}")
        for dx, dy, sk, lo, vi in nb:
            print(f"  ({dx:+d},{dy:+d}) skel={int(sk)} local={int(lo)} visited={int(vi)}")

    @staticmethod
    def _debug_vis_paths(skeleton, traj, paths, best, window="paths"):
        import cv2

        h, w = skeleton.shape
        disp = np.zeros((h, w, 3), dtype=np.uint8)
        disp[skeleton > 0] = (200, 200, 200)

        def _draw(pts, color, width=1):
            pts = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(disp, [pts], False, color, width)

        _draw(traj, (255, 0, 0), 1)
        for p in paths:
            _draw(p, (100, 100, 255), 1)
        _draw(best, (0, 255, 0), 1)

        # zoomed inset: 32×32 around traj[-1] → 128×128
        if len(traj) > 1:
            cx, cy = int(traj[-1][0]), int(traj[-1][1])
            r = 16
            x1, y1 = max(0, cx - r), max(0, cy - r)
            x2, y2 = min(w, cx + r), min(h, cy + r)
            roi = disp[y1:y2, x1:x2]
            zoom = cv2.resize(roi, (128, 128), interpolation=cv2.INTER_NEAREST)
            zh, zw = zoom.shape[:2]
            px = w - zw - 4
            py = h - zh - 4
            disp[py : py + zh, px : px + zw] = zoom
            cv2.rectangle(disp, (px, py), (px + zw, py + zh), (0, 255, 255), 1)

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.imshow(window, disp)
        cv2.waitKey(0)

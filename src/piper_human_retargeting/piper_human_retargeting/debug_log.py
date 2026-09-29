# -*- coding: utf-8 -*-
"""
piper_human_retargeting.debug_log
=================================
CSV 调试记录：raw / filtered 对比。

为什么必须记录 raw 与 filtered 两列：
    只看机器人动得顺不顺，无法判断问题出在哪一级：
        raw 抖 + filtered 稳  -> 滤波有效，但可能延迟过大
        raw 稳 + filtered 抖  -> 滤波器本身有问题（bug）
        raw 与 filtered 都抖  -> 问题在感知层，不在滤波
    只有两列同时记下来，事后才能区分这三种情况。

同时记录「映射后(滤波前)」的关节值，用于区分：
    抖动是来自人体角度，还是来自映射/限幅。

写入方式：带缓冲的追加写 + 定时 flush。
    每帧 flush 会让高频写盘拖慢控制循环，因此按行数/时间批量刷。
"""

import csv
import os
import time
from typing import Dict, List, Optional


class CsvDebugLogger:
    """
    CSV 调试记录器。

    用法：
        logger = CsvDebugLogger(path, every_n=1)
        logger.log(snapshot)      # snapshot 是 DebugSnapshot
        logger.close()
    """

    def __init__(self, path: str, every_n: int = 1,
                 flush_interval_s: float = 1.0, logger=None):
        self.path = path
        self.every_n = max(1, int(every_n))
        self.flush_interval_s = flush_interval_s
        self.logger = logger

        self._fh = None
        self._writer = None
        self._header_written = False
        self._n_seen = 0
        self._n_written = 0
        self._last_flush = 0.0

        self.open()

    # ------------------------------------------------------------
    def open(self) -> bool:
        try:
            d = os.path.dirname(os.path.abspath(self.path))
            if d:
                os.makedirs(d, exist_ok=True)
            # newline='' 是 csv 模块的要求，否则 Windows 换行会出问题
            self._fh = open(self.path, "w", newline="", encoding="utf-8")
            self._writer = None            # 表头在第一次 log 时写
            self._header_written = False
            self._last_flush = time.monotonic()
            return True
        except Exception as exc:                            # noqa: BLE001
            if self.logger:
                self.logger.error(f"CSV 打开失败 {self.path}: {exc}")
            self._fh = None
            return False

    # ------------------------------------------------------------
    def log(self, snapshot) -> bool:
        """记录一帧快照（按 every_n 抽样）"""
        if self._fh is None:
            return False

        self._n_seen += 1
        if (self._n_seen - 1) % self.every_n != 0:
            return False

        row: Dict[str, object] = snapshot.to_csv_row()

        try:
            if not self._header_written:
                header: List[str] = snapshot.csv_header()
                self._writer = csv.DictWriter(self._fh, fieldnames=header,
                                              extrasaction="ignore")
                self._writer.writeheader()
                self._header_written = True

            self._writer.writerow(row)
            self._n_written += 1

            now = time.monotonic()
            if now - self._last_flush >= self.flush_interval_s:
                self._fh.flush()
                self._last_flush = now
            return True
        except Exception as exc:                            # noqa: BLE001
            if self.logger:
                self.logger.error(f"CSV 写入失败: {exc}")
            return False

    # ------------------------------------------------------------
    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:                               # noqa: BLE001
                pass
            finally:
                self._fh = None

    @property
    def rows_written(self) -> int:
        return self._n_written

    @property
    def frames_seen(self) -> int:
        return self._n_seen

    def describe(self) -> str:
        return (f"CSV -> {self.path} "
                f"(每 {self.every_n} 帧记录, 已写 {self._n_written} 行)")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def analyze_csv_delay(path: str) -> Optional[dict]:
    """
    粗略分析 CSV：量化滤波延迟。

    做法：找 raw 序列里变化最快的一段，比较 raw 与 filtered 达到
    该变化量 63%（一阶系统时间常数量级）所用的帧数。
    这只是一个**粗略**指标，用于横向比较不同 alpha 的效果，
    不是严格的系统辨识。

    Returns:
        {"alpha_equiv_frames": float, "n_rows": int} 或 None（数据不足）
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:                                       # noqa: BLE001
        return None

    if len(rows) < 10:
        return None

    def col(name):
        out = []
        for r in rows:
            try:
                out.append(float(r[name]))
            except (KeyError, ValueError):
                out.append(None)
        return out

    raw = col("raw_upper")
    flt = col("filtered_upper")
    valid = [(a, b) for a, b in zip(raw, flt) if a is not None and b is not None]
    if len(valid) < 10:
        return None

    # 找最大阶跃
    best_i, best_d = 0, 0.0
    for i in range(1, len(valid)):
        d = abs(valid[i][0] - valid[i - 1][0])
        if d > best_d:
            best_d, best_i = d, i
    if best_d < 1.0:
        return {"alpha_equiv_frames": 0.0, "n_rows": len(rows)}

    target = valid[best_i][0]
    start = valid[best_i - 1][1]                    # 阶跃前的滤波值
    span = target - start
    if abs(span) < 1e-6:
        return {"alpha_equiv_frames": 0.0, "n_rows": len(rows)}

    # 找 filtered 达到 63% 的帧数
    frames = 0
    for k in range(best_i, min(best_i + 200, len(valid))):
        frames = k - best_i + 1
        ratio = (valid[k][1] - start) / span
        if ratio >= 0.63:
            break

    return {"alpha_equiv_frames": float(frames), "n_rows": len(rows)}

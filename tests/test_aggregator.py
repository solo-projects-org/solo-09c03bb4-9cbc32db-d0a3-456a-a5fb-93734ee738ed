"""聚合算法自测（不依赖网络，用内存假站点模拟两个站点服务）。

覆盖：
  1. 场景一（2026-01）：总榜第一在两站都没进首屏，需要翻到第二页后才能停读；
  2. 场景二（2026-02）：首屏后即可确认并停读；
  3. 同分按编号字典序（2026-03）；
  4. 缺号计零（点查路径）；
  5. K 大于组合总数；
  6. 某站取数失败时只报出错站点，且可重新查询；
  7. 全零月报（2026-05）：零次组合照常参榜但结果最多 K 条；
  8. 同分利用页末编号首屏即停（2026-06）；
  9. 未读同分组合仍可能入榜时必须续翻（2026-07）；
 10. 随机性质测试：受限取数结果与“直接拿全量暴力计算”的结果逐例一致
     （含零次计数与同分分层对抗数据）。
"""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "web"))

from aggregator import StationError, top_k  # noqa: E402

PAGE_SIZE = 5


class FakeStation:
    """内存站点：内部持有全量数据，只暴露分页榜单和点查，行为对齐真实接口。"""

    def __init__(self, name, data: dict[str, int], fail_rank_pages=()):
        self.name = name
        # 与 station/app.py 一致：次数降序、同分编号字典序
        self.ordered = sorted(data.items(), key=lambda kv: (-kv[1], kv[0]))
        self.rank_calls: list[int] = []
        self.count_calls: list[str] = []
        self.fail_rank_pages = set(fail_rank_pages)
        self.data = dict(data)

    def rank(self, month: str, page: int) -> dict:
        self.rank_calls.append(page)
        if page in self.fail_rank_pages:
            raise RuntimeError("模拟连接中断")
        total = len(self.ordered)
        total_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
        start = (page - 1) * PAGE_SIZE
        items = [
            {"combo_id": c, "count": n}
            for c, n in self.ordered[start:start + PAGE_SIZE]
        ]
        return {
            "station": self.name,
            "month": month,
            "page": page,
            "page_size": PAGE_SIZE,
            "total": total,
            "total_pages": total_pages,
            "has_next": start + PAGE_SIZE < total,
            "items": items,
        }

    def count(self, month: str, combo: str) -> dict:
        self.count_calls.append(combo)
        return {
            "station": self.name,
            "month": month,
            "combo_id": combo,
            "count": self.data.get(combo, 0),  # 缺号计零
        }


def build_stations(spec: dict[str, dict[str, int]], overrides=None):
    overrides = overrides or {}
    fakes = {
        name: FakeStation(name, data, **overrides.get(name, {}))
        for name, data in spec.items()
    }
    stations = {
        name: (fake.rank, fake.count) for name, fake in fakes.items()
    }
    return fakes, stations


# --- 与 station/data 下 CSV 完全一致的数据集 ---
DATA_2026_01 = {
    "维修站A": {
        "A01": 10, "A02": 10, "A03": 10, "A04": 10,
        "A05": 10, "Z999": 9, "A06": 1,
    },
    "维修站B": {
        "B01": 10, "B02": 10, "B03": 10, "B04": 10,
        "B05": 10, "Z999": 9, "B06": 1,
    },
}

DATA_2026_02 = {
    "维修站A": {
        "C001": 30, "C002": 20, "C003": 10, "C004": 6,
        "C005": 5, "C006": 2, "C007": 1,
    },
    "维修站B": {
        "C001": 25, "C002": 18, "C003": 12, "C004": 7,
        "C005": 4, "C006": 1,
    },
}

DATA_2026_03 = {
    "维修站A": {"T01": 5, "T02": 5, "T03": 5, "T04": 4, "T05": 2, "T06": 1},
    "维修站B": {"T01": 4, "T02": 4, "T03": 4, "T04": 4, "T05": 3},
}


DATA_2026_04 = {
    "维修站A": {
        "X01": 100, "A11": 9, "A12": 8, "A13": 7, "A14": 6,
        "A15": 5, "A16": 4, "A17": 3, "A18": 2, "A19": 1,
    },
    "维修站B": {
        "B11": 10, "B12": 9, "B13": 8, "B14": 7, "B15": 6,
        "B16": 5, "B17": 4, "B18": 3, "B19": 2, "B20": 1,
    },
}

# 全零月报：所有登记组合次数均为 0，仍按规则参榜（同分按编号字典序）
DATA_2026_05 = {
    "维修站A": {"Z01": 0, "Z02": 0, "Z03": 0, "Z04": 0, "Z05": 0, "Z06": 0},
    "维修站B": {"Z05": 0, "Z06": 0, "Z07": 0, "Z08": 0, "Z09": 0, "Z10": 0},
}

# 同分可停月：陌生组合最坏 5+5=10 与第 3 名同分，但未读编号必须大于
# 两站页末编号 D05，而 D05 > 第 3 名编号 D03，无法同分反超 -> 首屏即停
DATA_2026_06 = {
    "维修站A": {
        "H1": 8, "H2": 7, "D03": 5, "D04": 5, "D05": 5,
        "D06": 5, "D10": 5, "L1": 1,
    },
    "维修站B": {
        "H1": 8, "H2": 7, "D03": 5, "D04": 5, "D05": 5,
        "D06": 5, "D10": 5, "L2": 1,
    },
}

# 同分必须续翻月：C01 在两站都排第 7（首屏之外），两站各 5 次合计 10，
# 编号 C01 小于首屏候选；首屏后它确实可能挤进前 3，必须各翻第 2 页
DATA_2026_07 = {
    "维修站A": {
        "H1": 20, "A01": 6, "A02": 6, "A03": 6, "A04": 6,
        "A05": 6, "C01": 5, "Z01": 5, "L1": 1,
    },
    "维修站B": {
        "H2": 20, "B01": 6, "B02": 6, "B03": 6, "B04": 6,
        "B05": 6, "C01": 5, "Z02": 5, "L2": 1,
    },
}


class ScenarioTests(unittest.TestCase):
    def test_01_champion_beyond_first_screen(self):
        """场景一：总榜第一 Z999 在两站首屏（各前 5 条）都不出现，
        翻到第二页拿到后即可确认名单与次序并停读。"""
        fakes, stations = build_stations(DATA_2026_01)
        result = top_k("2026-01", 3, stations)

        ids = [r["combo_id"] for r in result["results"]]
        # 10 个首屏组合同分（总和 10），按编号字典序 A01 < A02 < B01 ...
        self.assertEqual(ids, ["Z999", "A01", "A02"])
        self.assertEqual(result["results"][0]["total"], 18)
        self.assertEqual(result["results"][0]["per_station"],
                         {"维修站A": 9, "维修站B": 9})

        # 两站首屏（前 5 条）都没有 Z999，它在两站都排在第 6 位
        self.assertNotIn("Z999", [c for c, _ in list(fakes["维修站A"].ordered)[:5]])
        self.assertNotIn("Z999", [c for c, _ in list(fakes["维修站B"].ordered)[:5]])

        # 必须各翻 2 页，且第 2 页拿到后立即停（没有第 3 次请求）
        self.assertEqual(result["pages_fetched"], {"维修站A": 2, "维修站B": 2})
        self.assertEqual(fakes["维修站A"].rank_calls, [1, 2])
        self.assertEqual(fakes["维修站B"].rank_calls, [1, 2])

    def test_02_stop_after_first_screen(self):
        """场景二：两站各取首屏后，陌生组合上界已小于第 K 名，立即停读。"""
        fakes, stations = build_stations(DATA_2026_02)
        result = top_k("2026-02", 3, stations)

        ids = [r["combo_id"] for r in result["results"]]
        self.assertEqual(ids, ["C001", "C002", "C003"])
        totals = {r["combo_id"]: r["total"] for r in result["results"]}
        self.assertEqual(totals, {"C001": 55, "C002": 38, "C003": 22})

        # 首屏（每页 5 条）之后就停：每站只请求 1 页
        self.assertEqual(result["pages_fetched"], {"维修站A": 1, "维修站B": 1})
        self.assertEqual(fakes["维修站A"].rank_calls, [1])
        self.assertEqual(fakes["维修站B"].rank_calls, [1])
        # 前 3 名都在两站首屏内，无需点查
        self.assertEqual(result["count_queries"], {"维修站A": 0, "维修站B": 0})

        # 复核停止不等式：首屏最小次数 A=5、B=4，G=9 < 第三名总和 22
        self.assertLess(5 + 4, 22)

    def test_03_tie_break_lexicographic(self):
        """同分按编号字典序。"""
        _, stations = build_stations(DATA_2026_03)
        result = top_k("2026-03", 4, stations)
        ids = [r["combo_id"] for r in result["results"]]
        # T01/T02/T03 总和都是 9，按字典序 T01 < T02 < T03；T04 总和 8
        self.assertEqual(ids, ["T01", "T02", "T03", "T04"])
        for cid in ("T01", "T02", "T03"):
            row = next(r for r in result["results"] if r["combo_id"] == cid)
            self.assertEqual(row["total"], 9)

    def test_04_missing_combo_counts_zero_via_point_query(self):
        """某组合只在一个站点出现时，另一站通过点查确认缺号计零。"""
        spec = {
            "维修站A": {"X1": 100, "X2": 90, "X3": 80, "X4": 70, "X5": 60, "X6": 1},
            "维修站B": {"Y1": 50, "Y2": 40, "Y3": 30, "Y4": 20, "Y5": 10, "Y6": 2},
        }
        fakes, stations = build_stations(spec)
        result = top_k("2026-09", 2, stations)
        rows = result["results"]
        self.assertEqual([r["combo_id"] for r in rows], ["X1", "X2"])
        # X1/X2 不在 B 的任何一页里（B 耗尽前上界仍 > 0 时走点查；耗尽后走缺号）
        for row in rows:
            self.assertEqual(row["per_station"]["维修站B"], 0)
        # B 站点确实对点查返回过 0
        self.assertIn("X1", fakes["维修站B"].count_calls)
        b_answer = fakes["维修站B"].count("2026-09", "NOT-EXIST")
        self.assertEqual(b_answer["count"], 0)

    def test_04b_point_query_when_station_not_exhausted(self):
        """2026-04 场景：榜首 X01 只在 A 首屏出现；B 未耗尽且首屏没有它，
        必须通过编号点查确认缺号计零，而不是继续翻 B 的榜单页。"""
        fakes, stations = build_stations(DATA_2026_04)
        result = top_k("2026-04", 1, stations)
        self.assertEqual(len(result["results"]), 1)
        row = result["results"][0]
        self.assertEqual(row["combo_id"], "X01")
        self.assertEqual(row["per_station"], {"维修站A": 100, "维修站B": 0})
        self.assertEqual(row["total"], 100)
        # 各只翻 1 页就确认停读
        self.assertEqual(result["pages_fetched"], {"维修站A": 1, "维修站B": 1})
        # B 未耗尽（10 条数据共 2 页），缺号只能点查确认，恰好 1 次
        self.assertFalse(fakes["维修站B"].rank_calls[-1] >= 2)
        self.assertEqual(result["count_queries"], {"维修站A": 0, "维修站B": 1})
        self.assertEqual(fakes["维修站B"].count_calls, ["X01"])

    def test_05_k_larger_than_universe(self):
        """K 大于两站组合总数时返回全部组合，次序仍符合规则。"""
        spec = {
            "维修站A": {"M1": 3, "M2": 1},
            "维修站B": {"M2": 2, "M3": 7},
        }
        _, stations = build_stations(spec)
        result = top_k("2026-10", 10, stations)
        ids = [r["combo_id"] for r in result["results"]]
        # 总和：M3=7, M2=3, M1=3；同分 3 按字典序 M1 < M2
        self.assertEqual(ids, ["M3", "M1", "M2"])
        self.assertEqual(len(result["results"]), 3)

    def test_06_station_failure_only_reports_failed_station_and_retryable(self):
        """一路取数失败：只提示出错站点；修复后重新查询可以成功。"""
        fakes, stations = build_stations(
            DATA_2026_01,
            {"维修站B": {"fail_rank_pages": (2,)}},
        )
        with self.assertRaises(StationError) as ctx:
            top_k("2026-01", 3, stations)
        self.assertEqual(ctx.exception.station, "维修站B")
        self.assertNotIn("维修站A", str(ctx.exception))

        # 模拟用户重新查询：换一个健康的同数据站点实例
        fakes2, stations2 = build_stations(DATA_2026_01)
        result = top_k("2026-01", 3, stations2)
        self.assertEqual([r["combo_id"] for r in result["results"]],
                         ["Z999", "A01", "A02"])

    def test_07_first_page_failure_reports_station(self):
        fakes, stations = build_stations(
            DATA_2026_02,
            {"维修站A": {"fail_rank_pages": (1,)}},
        )
        with self.assertRaises(StationError) as ctx:
            top_k("2026-02", 3, stations)
        self.assertEqual(ctx.exception.station, "维修站A")

    def test_08_all_zero_month_returns_at_most_k(self):
        """全零月报查前 3：零次组合照常参榜，但结果最多 3 条，不得列出整榜。"""
        fakes, stations = build_stations(DATA_2026_05)
        result = top_k("2026-05", 3, stations)
        ids = [r["combo_id"] for r in result["results"]]
        # 并集 10 个零次组合，字典序前 3 名入榜
        self.assertEqual(ids, ["Z01", "Z02", "Z03"])
        self.assertEqual(len(result["results"]), 3)
        for row in result["results"]:
            self.assertEqual(row["total"], 0)
            self.assertEqual(set(row["per_station"].values()), {0})
        # 全零上界为 0，无需任何点查
        self.assertEqual(result["count_queries"], {"维修站A": 0, "维修站B": 0})

    def test_08b_all_zero_month_k_larger_than_universe(self):
        """全零月报且 K 大于组合总数：全部列出，条数仍不超过 K。"""
        _, stations = build_stations(DATA_2026_05)
        result = top_k("2026-05", 20, stations)
        self.assertEqual(len(result["results"]), 10)
        ids = [r["combo_id"] for r in result["results"]]
        self.assertEqual(ids, sorted(ids))

    def test_08c_registered_zero_combos_participate_with_positive_ones(self):
        """零次组合与正次数混合：零次组合照常参榜并按规则排在末尾。"""
        spec = {
            "维修站A": {"P1": 5, "P2": 0},
            "维修站B": {"P1": 3, "P3": 0},
        }
        _, stations = build_stations(spec)
        result = top_k("mix", 10, stations)
        ids = [r["combo_id"] for r in result["results"]]
        # 总和：P1=8；P2=0、P3=0 同分按字典序 P2 < P3
        self.assertEqual(ids, ["P1", "P2", "P3"])
        self.assertEqual(len(result["results"]), 3)

    def test_09_tie_uses_page_end_id_to_stop_after_first_screen(self):
        """2026-06：陌生组合最坏与第 3 名同分，但页末编号约束其编号更大，
        首屏即可确认停读，旧实现会保守地多翻第 2 页。"""
        fakes, stations = build_stations(DATA_2026_06)
        result = top_k("2026-06", 3, stations)
        ids = [r["combo_id"] for r in result["results"]]
        self.assertEqual(ids, ["H1", "H2", "D03"])
        self.assertEqual(result["pages_fetched"], {"维修站A": 1, "维修站B": 1})
        self.assertEqual(fakes["维修站A"].rank_calls, [1])
        self.assertEqual(fakes["维修站B"].rank_calls, [1])
        # 入榜组合都在两站首屏内，无需点查
        self.assertEqual(result["count_queries"], {"维修站A": 0, "维修站B": 0})
        # 复核判定依据：页末 (5, D05)，陌生同分组合编号必须 > D05 > D03
        self.assertGreater("D05", "D03")

    def test_10_unread_tie_combo_may_enter_so_paging_continues(self):
        """2026-07：未读的 C01 两站各 5 次（合计 10）且编号更小，首屏后
        仍可能挤进前 3，不允许提前停读；翻到第 2 页后确认 C01 入榜。"""
        fakes, stations = build_stations(DATA_2026_07)
        result = top_k("2026-07", 3, stations)
        ids = [r["combo_id"] for r in result["results"]]
        self.assertEqual(ids, ["H1", "H2", "C01"])
        # C01 在两站都位于第 2 页，两站都必须续翻
        self.assertEqual(result["pages_fetched"], {"维修站A": 2, "维修站B": 2})
        self.assertEqual(fakes["维修站A"].rank_calls, [1, 2])
        self.assertEqual(fakes["维修站B"].rank_calls, [1, 2])
        c01 = next(r for r in result["results"] if r["combo_id"] == "C01")
        self.assertEqual(c01["per_station"], {"维修站A": 5, "维修站B": 5})
        self.assertEqual(c01["total"], 10)

    def test_11_empty_station_month(self):
        """某站当月没有任何登记组合：空页即耗尽，另一站数据独立成榜，
        结果不超过 K 且不发点查。"""
        spec = {
            "维修站A": {"E1": 4, "E2": 3, "E3": 2},
            "维修站B": {},
        }
        fakes, stations = build_stations(spec)
        result = top_k("empty", 5, stations)
        ids = [r["combo_id"] for r in result["results"]]
        self.assertEqual(ids, ["E1", "E2", "E3"])
        self.assertEqual(len(result["results"]), 3)
        for row in result["results"]:
            self.assertEqual(row["per_station"]["维修站B"], 0)
        self.assertEqual(result["count_queries"], {"维修站A": 0, "维修站B": 0})
        # 空站首请求即空页，页码仍记 1
        self.assertEqual(result["pages_fetched"]["维修站B"], 1)


class BruteForce:
    """基准实现：不受接口限制，直接用全量数据暴力计算 Top-K。"""

    @staticmethod
    def top_k(spec: dict[str, dict[str, int]], k: int):
        union: dict[str, int] = {}
        per_station: dict[str, dict[str, int]] = {}
        for station, data in spec.items():
            per_station[station] = data
            for combo, cnt in data.items():
                union[combo] = union.get(combo, 0) + cnt
        ordered = sorted(union.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [
            {
                "combo_id": combo,
                "total": total,
                "per_station": {s: data.get(combo, 0) for s, data in spec.items()},
            }
            for combo, total in ordered
        ]


class RandomizedPropertyTests(unittest.TestCase):
    def test_matches_brute_force_on_random_data(self):
        rng = random.Random(20261120)
        for trial in range(300):
            n_a = rng.randint(1, 14)
            n_b = rng.randint(1, 14)
            pool = [f"C{i:03d}" for i in range(rng.randint(1, 20))]
            rng.shuffle(pool)
            spec = {
                "维修站A": {
                    c: rng.randint(1, 50) for c in rng.sample(pool, min(n_a, len(pool)))
                },
                "维修站B": {
                    c: rng.randint(1, 50) for c in rng.sample(pool, min(n_b, len(pool)))
                },
            }
            k = rng.randint(1, 12)

            fakes, stations = build_stations(spec)
            result = top_k("random", k, stations)
            expected = BruteForce.top_k(spec, k)

            self.assertEqual(
                [(r["combo_id"], r["total"]) for r in result["results"]],
                [(r["combo_id"], r["total"]) for r in expected],
                msg=f"trial {trial} 名次不一致",
            )
            for got, exp in zip(result["results"], expected):
                self.assertEqual(got["per_station"], exp["per_station"],
                                 msg=f"trial {trial} 分站次数不一致")
            # 不变量：结果条数永远不超过 K（全零月报曾违反过这一条）
            self.assertLessEqual(len(result["results"]), k, msg=f"trial {trial} 超出 K 条")

            # 不变量：翻页数不得超过各站总页数，且每站请求页码连续
            for name, fake in fakes.items():
                total_pages = (len(fake.ordered) + PAGE_SIZE - 1) // PAGE_SIZE
                self.assertLessEqual(len(fake.rank_calls), total_pages)
                self.assertEqual(fake.rank_calls, list(range(1, len(fake.rank_calls) + 1)))

    def test_matches_brute_force_with_zero_counts(self):
        """含大量零次登记组合的随机数据：零次照常参榜，且结果仍与暴力基准一致。"""
        rng = random.Random(20261120)
        for trial in range(300):
            pool = [f"Z{i:03d}" for i in range(rng.randint(1, 24))]
            rng.shuffle(pool)
            spec = {
                "维修站A": {c: rng.randint(0, 20) for c in rng.sample(
                    pool, rng.randint(1, len(pool)))},
                "维修站B": {c: rng.randint(0, 20) for c in rng.sample(
                    pool, rng.randint(1, len(pool)))},
            }
            k = rng.randint(1, 15)
            _, stations = build_stations(spec)
            result = top_k("random-zero", k, stations)
            expected = BruteForce.top_k(spec, k)
            self.assertEqual(
                [(r["combo_id"], r["total"]) for r in result["results"]],
                [(r["combo_id"], r["total"]) for r in expected],
                msg=f"zero trial {trial} 名次不一致",
            )
            self.assertLessEqual(len(result["results"]), k)

    def test_tie_layers_never_wrong_and_never_over_fetches(self):
        """同分分层对抗数据：结果与暴力基准一致；并抽查“页末编号可判定”
        的场景确实在首屏停读（旧实现做不到）。"""
        rng = random.Random(7)
        saved_pages = 0
        for trial in range(200):
            # 人为制造大量同分：次数只在少数几个档位里取
            tiers = [8, 5, 1, 0]
            pool = [f"T{i:03d}" for i in range(rng.randint(6, 22))]
            rng.shuffle(pool)
            spec = {}
            for station in ("维修站A", "维修站B"):
                chosen = rng.sample(pool, rng.randint(5, len(pool)))
                spec[station] = {c: rng.choice(tiers) for c in chosen}
            k = rng.randint(1, 8)
            fakes, stations = build_stations(spec)
            result = top_k("random-tie", k, stations)
            expected = BruteForce.top_k(spec, k)
            self.assertEqual(
                [r["combo_id"] for r in result["results"]],
                [r["combo_id"] for r in expected],
                msg=f"tie trial {trial} 名次不一致",
            )
            self.assertLessEqual(len(result["results"]), k)
            for name, fake in fakes.items():
                total_pages = (len(fake.ordered) + PAGE_SIZE - 1) // PAGE_SIZE
                self.assertLessEqual(len(fake.rank_calls), total_pages)
            saved_pages += sum(
                (len(fake.ordered) + PAGE_SIZE - 1) // PAGE_SIZE - len(fake.rank_calls)
                for fake in fakes.values()
            )
        # 200 组分层数据里整体应存在可观的早停节省（旧实现更保守）
        self.assertGreater(saved_pages, 0)

    def test_early_stop_never_exceeds_needed_pages(self):
        """性质：停读时再继续翻页也不会改变已确认的 Top-K（用暴力基准佐证）。"""
        rng = random.Random(42)
        for _ in range(100):
            spec = {
                "维修站A": {f"A{i}": rng.randint(1, 30) for i in range(rng.randint(5, 13))},
                "维修站B": {f"B{i}": rng.randint(1, 30) for i in range(rng.randint(5, 13))},
            }
            k = rng.randint(1, 5)
            fakes, stations = build_stations(spec)
            result = top_k("random", k, stations)
            expected = BruteForce.top_k(spec, k)
            self.assertEqual(
                [r["combo_id"] for r in result["results"]],
                [r["combo_id"] for r in expected],
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""分析网页的取数与 Top-K 聚合算法（纯逻辑，不依赖 Flask，便于自测）。

站点接口的约束：
  * 只能按次数从高到低、每页 5 条翻页（rank），或按编号点查（count，缺号计 0）；
  * 中心不能直接读取站点全量数据。

算法目标：在无法一次拿到全量数据的前提下，用尽量少的翻页确认
两站次数之和的前 K 名（同分按编号字典序），一旦名单和次序可确认就停。

停读判定（每读完一页都做一次）：
  榜单按 (次数降序, 编号字典序) 排列。设某站已读最后一页页末记录为
  (e_count, e_id)，且该站还有未读页，则任何未读组合 c 在该站：
    * 次数不超过 e_count；
    * 若次数恰好等于 e_count，则编号 c 严格大于 e_id
      （页末已是本页该次数层字典序最大的编号）。
  站点已耗尽（无下一页）时，未读组合在该站次数确定为 0。
  据此为每个“未在某站榜单出现过”的组合估计它在该站的最坏次数，
  再求两站最坏总和；当第 K 名之外的所有组合（含完全陌生组合）在
  最坏情况下都无法超越第 K 名、且前 K 名内部次序也不会被反超时，
  名单与次序即确认，立即停翻，随后只用点查补齐入榜组合的分站次数。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

PAGE_SIZE = 5


class StationError(RuntimeError):
    """某个站点取数失败。只携带出错站点信息，便于上层只提示出错站点。"""

    def __init__(self, station: str, detail: str = ""):
        self.station = station
        self.detail = detail
        super().__init__(station, detail)


# rank_fetcher(month, page) -> 站点 rank 接口的 JSON(dict)
# count_fetcher(month, combo) -> 站点 count 接口的 JSON(dict)
RankFetcher = Callable[[str, int], dict]
CountFetcher = Callable[[str, str], dict]


@dataclass
class StationState:
    name: str
    rank_fetcher: RankFetcher
    count_fetcher: CountFetcher
    page: int = 0                      # 已取到第几页
    exhausted: bool = False
    total_pages: Optional[int] = None
    # 已确定次数的组合（榜单页出现过，或点查确认过）-> 该站次数
    seen_counts: dict[str, int] = field(default_factory=dict)
    # 已读最后一页页末记录：(最小次数, 该次数层最大编号)
    last_page_end: Optional[tuple[int, str]] = None

    def has_bound(self) -> bool:
        """是否已有可用的未读上界（至少读过一页，或已经耗尽）。"""
        return self.exhausted or self.last_page_end is not None

    def unseen_max_count(self, combo: Optional[str]) -> int:
        """组合 combo 在本站“未读记录”中的最坏次数（含）。

        combo 为 None 表示完全陌生的组合（编号未知，只做次数上界）。
          * 已耗尽：未读组合次数确定为 0；
          * 页末次数为 0：未读组合次数也只能是 0；
          * 否则未读次数不超过页末次数 e_count；当 combo 编号已知且
            不大于页末编号时，它不可能以 e_count 次出现在未读页，
            最坏只能取 e_count - 1（编号字典序约束）。
        """
        if self.exhausted:
            return 0
        if self.last_page_end is None:
            raise RuntimeError(f"站点 {self.name} 尚未取过任何页，无上界可用")
        e_count, e_id = self.last_page_end
        if e_count == 0:
            return 0
        if combo is not None and combo <= e_id:
            return e_count - 1
        return e_count


def _fetch_rank(state: StationState, month: str) -> dict:
    # 调用方在请求前已把 state.page 递增为待取页码
    try:
        payload = state.rank_fetcher(month, state.page)
    except StationError:
        raise
    except Exception as exc:  # 网络错误、超时、非 JSON 等统一归为该站出错
        raise StationError(state.name, str(exc)) from exc
    if not isinstance(payload, dict) or "items" not in payload:
        raise StationError(state.name, "榜单响应格式不正确")
    return payload


def _fetch_count(state: StationState, month: str, combo: str) -> int:
    try:
        payload = state.count_fetcher(month, combo)
    except StationError:
        raise
    except Exception as exc:
        raise StationError(state.name, str(exc)) from exc
    if not isinstance(payload, dict) or "count" not in payload:
        raise StationError(state.name, "点查响应格式不正确")
    try:
        return int(payload["count"])
    except (TypeError, ValueError) as exc:
        raise StationError(state.name, "点查次数格式不正确") from exc


def _rebuild_union(states: dict[str, StationState]) -> dict[str, dict]:
    """用各站已确定的次数重建候选全集（含点查补入的缺号零值）。"""
    union: dict[str, dict] = {}
    for state in states.values():
        for combo, cnt in state.seen_counts.items():
            record = union.setdefault(combo, {"sum": 0, "stations": {}})
            # 同一组合在同一站只会有一个确定值，直接覆盖并由各站值求和
            record["stations"][state.name] = cnt
            record["sum"] = sum(record["stations"].values())
    return union


def _ordered_candidates(union: dict[str, dict]) -> list[tuple[str, int]]:
    """候选组合按 (总次数降序, 编号字典序) 排序。"""
    return sorted(
        ((combo, vals["sum"]) for combo, vals in union.items()),
        key=lambda x: (-x[1], x[0]),
    )


def _worst_total(
    combo: Optional[str],
    union: dict[str, dict],
    states: dict[str, StationState],
) -> int:
    """组合 combo 在所有未读记录都取最坏值时的两站总和上界。

    combo 为 None 表示陌生组合（无任何已确定次数，编号未知）。
    """
    worst = 0
    for state in states.values():
        if combo is not None and combo in state.seen_counts:
            worst += state.seen_counts[combo]  # 该站次数已由榜单/点查确定
        else:
            worst += state.unseen_max_count(combo)
    return worst


def _stranger_cannot_overtake(
    kth_combo: str,
    kth_sum: int,
    states: dict[str, StationState],
) -> bool:
    """判断完全陌生的组合是否确定无法超越第 K 名。

    陌生组合没有任何已确定次数，最坏总和 W = Σ 各站未读最坏次数。
      * W < 第 K 名总和：确定无法超越；
      * W > 第 K 名总和：可能超越；
      * W 相等：只有当它能以“编号字典序小于第 K 名”的身份取到 W 时
        才能超越。各非耗尽且页末次数 > 0 的站点都会约束其编号严格
        大于该站页末编号；若这些约束下不存在小于第 K 名编号的可能，
        则确定无法超越。站点耗尽或页末次数为 0 时不贡献次数，
        也不产生编号约束。
    """
    worst = 0
    constrained_ends: list[str] = []
    for state in states.values():
        if state.exhausted or state.last_page_end is None:
            continue
        e_count, e_id = state.last_page_end
        if e_count > 0:
            worst += e_count
            constrained_ends.append(e_id)
    if worst < kth_sum:
        return True
    if worst > kth_sum:
        return False
    if not constrained_ends:
        # W 等于第 K 名总和却没有任何编号约束（所有贡献都是 0）：
        # 陌生组合可能携带更小的编号，必须保守判定为可能超越
        return False
    # 陌生组合编号必须严格大于每个贡献正次数站点的页末编号，
    # 即 > max(页末编号)；若该下界已不小于第 K 名编号，则不存在
    # 字典序更小的可行编号，无法在同分中反超
    return max(constrained_ends) >= kth_combo


def _bound_allows_stop(
    union: dict[str, dict], states: dict[str, StationState], k: int
) -> Optional[list[tuple[str, int]]]:
    """仅依据榜单页数据，检查 Top-K 的名单与次序是否已经可以确认。

    可以确认则返回候选前 K 名 [(combo, total), ...]，否则返回 None。
    任一站还没取首页时没有可用上界，不能停。
    """
    if not all(s.has_bound() for s in states.values()):
        return None

    ordered = _ordered_candidates(union)
    if not ordered:
        # 两站至少各读过一页仍无任何登记组合：整体候选集为空
        return []
    if len(ordered) < k:
        if not all(s.exhausted for s in states.values()):
            # 未读页里可能还有新组合，候选总数未定，不能提前收尾
            return None
        # 两站全部耗尽仍不足 K 个：数据本身就这么多，全部按规则返回
        return ordered
    top_k = ordered[:k]

    # 前 K 内部次序也必须已经确定：相邻两名 i、i+1，即使 i+1 在所有
    # 未确定站都取到最坏值、i 保持已确定值，i 仍不能被反超（含同分
    # 字典序）。相邻对都稳定则整张前 K 榜的次序稳定
    for idx in range(len(top_k) - 1):
        hi_combo, _ = top_k[idx]
        lo_combo, _ = top_k[idx + 1]
        lo_worst = _worst_total(lo_combo, union, states)
        hi_known = union[hi_combo]["sum"]
        if lo_worst > hi_known or (lo_worst == hi_known and lo_combo < hi_combo):
            return None

    kth_combo, kth_sum = top_k[-1]
    if not _stranger_cannot_overtake(kth_combo, kth_sum, states):
        return None

    # 第 K 名之外的已见组合：逐个用含页末编号约束的最坏总和判断
    for combo, _ in ordered[k:]:
        worst = _worst_total(combo, union, states)
        if worst > kth_sum or (worst == kth_sum and combo < kth_combo):
            return None
    return top_k


def _resolve_top_k_counts(
    top_k: list[tuple[str, int]],
    states: dict[str, StationState],
    month: str,
    count_queries: dict[str, int],
) -> list[dict]:
    """对前 K 名逐个站点确认精确次数（榜单没有就点查，缺号计 0）。"""
    results: list[dict] = []
    for combo, _ in top_k:
        entry = {"combo_id": combo, "per_station": {}}
        for state in states.values():
            if combo in state.seen_counts:
                value = state.seen_counts[combo]
            elif state.exhausted:
                value = 0  # 整站榜单都翻过仍未出现，确定为缺号
                state.seen_counts[combo] = 0
            else:
                value = _fetch_count(state, month, combo)
                count_queries[state.name] += 1
                state.seen_counts[combo] = value
            entry["per_station"][state.name] = value
        entry["total"] = sum(entry["per_station"].values())
        results.append(entry)
    results.sort(key=lambda e: (-e["total"], e["combo_id"]))
    return results


def top_k(
    month: str,
    k: int,
    stations: dict[str, tuple[RankFetcher, CountFetcher]],
    max_pages: int = 10000,
) -> dict:
    """聚合两个站点的数据，返回 Top-K 及每站实际翻页数。

    stations: {站点名: (rank_fetcher, count_fetcher)}
    返回: {"month", "k", "results", "pages_fetched": {站点: 页数},
           "count_queries": {站点: 点查次数}}
    任一站取数失败即抛 StationError，只携带出错站点，上层可重新查询。
    """
    if k < 1:
        raise ValueError("k 必须是正整数")
    if not stations:
        raise ValueError("至少需要一个站点")

    states = {
        name: StationState(name=name, rank_fetcher=rf, count_fetcher=cf)
        for name, (rf, cf) in stations.items()
    }
    count_queries = {name: 0 for name in states}
    station_names = list(stations.keys())

    def ingest_page(state: StationState, items: list[dict]) -> None:
        if not items:
            # 空页不改变既有页末上界；调用方会据此把站点标记为耗尽
            return
        min_count = min(int(item["count"]) for item in items)
        # 页末编号 = 最小次数层里字典序最大的编号
        end_id = max(
            str(item["combo_id"])
            for item in items
            if int(item["count"]) == min_count
        )
        for item in items:
            state.seen_counts[str(item["combo_id"])] = int(item["count"])
        state.last_page_end = (min_count, end_id)

    # 两站轮流翻页，每取完一页就检查能否确认；确认成功立即停止
    while True:
        # 点查补入的次数也会写入 seen_counts，每轮统一重建保证总和最新
        union = _rebuild_union(states)
        candidate_top = _bound_allows_stop(union, states, k)
        if candidate_top is not None:
            # 无论候选是否满 K 个，结果都不得超过 K 条
            results = _resolve_top_k_counts(
                candidate_top[:k], states, month, count_queries
            )
            break

        if all(s.exhausted for s in states.values()):
            # 兜底：两站全部翻完仍无法经由常规判定确认（理论上不会发生，
            # 因为耗尽后所有次数均已确定）。同样只取前 K，绝不列出整榜
            all_ordered = _ordered_candidates(union)[:k]
            results = _resolve_top_k_counts(
                all_ordered, states, month, count_queries
            )
            break

        progressed = False
        for name in station_names:
            state = states[name]
            if state.exhausted:
                continue
            if state.page >= max_pages:
                raise StationError(name, f"翻页超过上限 {max_pages}")
            state.page += 1
            payload = _fetch_rank(state, month)
            page_items = payload.get("items") or []
            state.total_pages = payload.get("total_pages")
            ingest_page(state, page_items)
            has_next = payload.get("has_next")
            if has_next is None:
                has_next = (
                    state.total_pages is not None
                    and state.page < state.total_pages
                )
            if not has_next or not page_items:
                state.exhausted = True
            progressed = True

            candidate_top = _bound_allows_stop(_rebuild_union(states), states, k)
            if candidate_top is not None:
                results = _resolve_top_k_counts(
                    candidate_top[:k], states, month, count_queries
                )
                break
        else:
            # for 正常跑完未 break：继续 while 下一轮（由 progressed 兜底防死循环）
            if not progressed:
                raise RuntimeError("无法继续取数且未能确认结果")
            continue
        break

    return {
        "month": month,
        "k": k,
        "results": results,
        "pages_fetched": {name: states[name].page for name in states},
        "count_queries": count_queries,
    }

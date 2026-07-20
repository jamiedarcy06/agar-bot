import math
from collections import defaultdict

from helper.game import Game
from lib.config.arena import MAX_BLOB_COUNT, VISION_REFERENCE_SUM_OF_RADII
from lib.config.player import (
    BASE_PLAYER_SPEED,
    EAT_SIZE_RATIO,
    MERGE_ATTRACTION_SPEED,
    MIN_PLAYER_SPEED,
    PLAYER_SPEED_RADIUS_FACTOR,
    SAME_PLAYER_OVERLAP_EPSILON,
    SPLIT_EJECT_DRAG,
    SPLIT_EJECT_SPEED,
    SPLIT_MIN_MASS,
)
from lib.interface.events.moves.move_player import MovePlayer
from lib.interface.queries.query_move import QueryMovePlayer
from lib.models.penguin_model import DirectionModel

P_FOOD = 2.6
C_FOOD = 1.0

VIRUS_BAIT_RATIO = EAT_SIZE_RATIO
VIRUS_DANGER_MARGIN = 0.1
VIRUS_DANGER_RATIO = VIRUS_BAIT_RATIO - VIRUS_DANGER_MARGIN
P_VIRUS = 2.5
C_VIRUS = 1.0

LUNGE_PATH_DRIFT_MARGIN = 1.0

EAT_RATIO = 1.11
PREY_MAX_RATIO = 1.0 / EAT_RATIO
THREAT_MIN_RATIO = EAT_RATIO
P_PREY_THREAT = 2.0
C_PREY = 0.764
C_THREAT = 0.332

REAL_EAT_RATIO = math.sqrt(EAT_SIZE_RATIO)

THREAT_DETECTION_RANGE = 15.0

THREAT_DETECTION_RANGE_THREAT = 22.0

MIN_SWARM_SIZE = 3
SPLIT_EAT_RATIO = math.sqrt(2.0 * EAT_SIZE_RATIO)

VISION_SPLIT_MIN_RADIUS = VISION_REFERENCE_SUM_OF_RADII / math.sqrt(2.0)
EJECT_MAX = SPLIT_EJECT_SPEED / (1.0 - SPLIT_EJECT_DRAG)
PROXIMITY_HORIZON = 3
ATTACK_PROXIMITY_HORIZON = 4

THREAT_IMMEDIATE_HORIZON = 3

MULTI_THREAT_LOOKAHEAD_ROUNDS = 15
MULTI_THREAT_CANDIDATE_HEADINGS = 32

MULTI_THREAT_CANCELLATION_RATIO = 0.5


def _eject_partial(t: int) -> float:
    return SPLIT_EJECT_SPEED * (1.0 - SPLIT_EJECT_DRAG**t) / (1.0 - SPLIT_EJECT_DRAG)


def _is_food_reachable(fx: float, fy: float, radius: float, size: float) -> bool:
    lo, hi = radius, size - radius
    closest_x = min(max(fx, lo), hi)
    closest_y = min(max(fy, lo), hi)
    return math.hypot(fx - closest_x, fy - closest_y) <= radius


def _movement_speed(radius: float) -> float:
    return max(MIN_PLAYER_SPEED, BASE_PLAYER_SPEED / (1.0 + radius * PLAYER_SPEED_RADIUS_FACTOR))


def _nearest_own_dist(px: float, py: float, my_own_positions) -> float:
    best_sq = math.inf
    for bx, by in my_own_positions:
        dx = px - bx
        dy = py - by
        d_sq = dx * dx + dy * dy
        if d_sq < best_sq:
            best_sq = d_sq
    return math.sqrt(best_sq)


WALL_MARGIN_MIN = 4.0
WALL_MARGIN_FRACTION = 0.1
WALL_MARGIN_HORIZON = 3
WALL_MARGIN_EXIT_MARGIN = 1.3
_wall_near_x = False
_wall_near_y = False


def _wall_repulsion(x: float, y: float, size: float, margin_x: float, margin_y: float) -> tuple[float, float]:
    fx = fy = 0.0
    if x < margin_x:
        fx += (margin_x - x) / margin_x
    elif x > size - margin_x:
        fx -= (x - (size - margin_x)) / margin_x
    if y < margin_y:
        fy += (margin_y - y) / margin_y
    elif y > size - margin_y:
        fy -= (y - (size - margin_y)) / margin_y
    return (fx, fy)


def _update_wall_proximity(mx: float, my: float, my_r: float, size: float) -> tuple[float, float]:
    global _wall_near_x, _wall_near_y
    flat_margin = max(WALL_MARGIN_MIN, size * WALL_MARGIN_FRACTION)
    reach_margin = max(WALL_MARGIN_MIN, _movement_speed(my_r) * WALL_MARGIN_HORIZON + my_r)
    base_margin = min(flat_margin, reach_margin)
    margin_x = base_margin * WALL_MARGIN_EXIT_MARGIN if _wall_near_x else base_margin
    margin_y = base_margin * WALL_MARGIN_EXIT_MARGIN if _wall_near_y else base_margin
    wx, wy = _wall_repulsion(mx, my, size, margin_x, margin_y)
    _wall_near_x = wx != 0.0
    _wall_near_y = wy != 0.0
    return wx, wy


def _redirect_along_wall(dx: float, dy: float, mx: float, my: float, my_r: float, size: float) -> tuple[float, float]:
    wx, wy = _update_wall_proximity(mx, my, my_r, size)
    return _apply_wall_redirect(dx, dy, wx, wy)


def _apply_wall_redirect(dx: float, dy: float, wx: float, wy: float) -> tuple[float, float]:
    if wx == 0.0 and wy == 0.0:
        return dx, dy

    clipped_x = wx != 0.0 and dx * wx < 0.0
    clipped_y = wy != 0.0 and dy * wy < 0.0

    if clipped_x and clipped_y:
        score_x = dx * math.copysign(1.0, wx)
        score_y = dy * math.copysign(1.0, wy)
        return (wx, 0.0) if score_x >= score_y else (0.0, wy)

    rdx = 0.0 if clipped_x else dx
    rdy = 0.0 if clipped_y else dy
    return rdx, rdy


def _predicted_own_blob_count(my_own_blobs) -> int:
    blobs = list(my_own_blobs)
    n = len(blobs)
    if n <= 1:
        return n
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    slack = SAME_PLAYER_OVERLAP_EPSILON + 2.0 * MERGE_ATTRACTION_SPEED
    for i in range(n):
        bi = blobs[i]
        if bi.merge_cooldown > 1:
            continue
        for j in range(i + 1, n):
            bj = blobs[j]
            if bj.merge_cooldown > 1:
                continue
            dist = math.hypot(bi.pos[0] - bj.pos[0], bi.pos[1] - bj.pos[1])
            if dist <= bi.radius + bj.radius + slack:
                union(i, j)

    return len({find(i) for i in range(n)})


def _virus_pop_is_dangerous(my_r: float, my_blob_count: int, virus_radius: float, visible_blobs) -> bool:
    piece_count = max(1, MAX_BLOB_COUNT - my_blob_count + 1)
    total_mass_after = my_r * my_r + virus_radius * virus_radius
    piece_radius = math.sqrt(total_mass_after / piece_count)
    eat_threshold = piece_radius * EAT_SIZE_RATIO
    return any(blob.radius >= eat_threshold for blob in visible_blobs)


def _virus_pop_would_be_dangerous(child_radius: float, blob_count_after_split: int, virus, visible_blobs) -> bool:
    piece_count = max(1, MAX_BLOB_COUNT - blob_count_after_split + 1)
    total_mass_after = child_radius * child_radius + virus.radius * virus.radius
    piece_radius = math.sqrt(total_mass_after / piece_count)
    eat_threshold = piece_radius * EAT_SIZE_RATIO
    return any(blob.radius >= eat_threshold for blob in visible_blobs)


def _vision_split_is_safe(
    my_own_positions, blob_count_after_split: int, child_radius: float, visible_blobs, visible_viruses
) -> bool:
    for blob in visible_blobs:
        if blob.radius / child_radius < THREAT_MIN_RATIO:
            continue
        nearest_own_dist = _nearest_own_dist(blob.pos[0], blob.pos[1], my_own_positions)
        if nearest_own_dist <= THREAT_DETECTION_RANGE:
            return False

    child_reach = child_radius + _movement_speed(child_radius)
    for virus in visible_viruses:
        if child_radius / virus.radius < VIRUS_DANGER_RATIO:
            continue
        nearest_own_dist = _nearest_own_dist(virus.pos[0], virus.pos[1], my_own_positions)
        if nearest_own_dist > child_reach:
            continue
        if _virus_pop_would_be_dangerous(child_radius, blob_count_after_split, virus, visible_blobs):
            return False

    return True


def _point_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len_sq))
    projx, projy = ax + t * dx, ay + t * dy
    return math.hypot(px - projx, py - projy)


def _split_landing_point(mx: float, my: float, tx: float, ty: float, child_radius: float) -> tuple[float, float]:
    dx, dy = tx - mx, ty - my
    d = math.hypot(dx, dy)
    if d < 1e-9:
        return mx, my
    ux, uy = dx / d, dy / d
    jump = 2.0 * child_radius + _movement_speed(child_radius) + SPLIT_EJECT_SPEED
    return mx + ux * jump, my + uy * jump


def _virus_blocks_lunge(
    mx: float, my: float, tx: float, ty: float, child_radius: float, blob_count_after_split: int, visible_blobs, visible_viruses
) -> bool:
    lx = ly = None
    for virus in visible_viruses:
        if child_radius < virus.radius * VIRUS_BAIT_RATIO:
            continue
        if lx is None:
            lx, ly = _split_landing_point(mx, my, tx, ty, child_radius)
        if _point_segment_distance(virus.pos[0], virus.pos[1], lx, ly, tx, ty) > child_radius + LUNGE_PATH_DRIFT_MARGIN:
            continue
        if _virus_pop_would_be_dangerous(child_radius, blob_count_after_split, virus, visible_blobs):
            return True
    return False


def _resolve_swarm_threats(mx, my, my_mass, my_smallest, my_largest, my_blob_count, my_own_positions, visible_blobs, visible_viruses):
    by_player = defaultdict(list)
    for blob in visible_blobs:
        by_player[blob.player_id].append(blob)

    flee_x = flee_y = 0.0
    any_flee = False
    eat_candidate = None

    for frags in by_player.values():
        if len(frags) < MIN_SWARM_SIZE:
            continue
        total_mass = sum(f.radius * f.radius for f in frags)
        if total_mass <= 0:
            continue
        projected_radius = math.sqrt(total_mass)
        if projected_radius / my_smallest < THREAT_MIN_RATIO:
            continue

        cx = sum(f.pos[0] * f.radius * f.radius for f in frags) / total_mass
        cy = sum(f.pos[1] * f.radius * f.radius for f in frags) / total_mass

        swarm_eat = None
        swarm_has_dangerous_fragment = False
        resplit_declined = False
        for frag in frags:
            fdx, fdy = frag.pos[0] - mx, frag.pos[1] - my
            fdist = math.hypot(fdx, fdy)
            if fdist < 1e-6:
                continue

            frag_mass = frag.radius * frag.radius
            frag_speed = _movement_speed(frag.radius)

            if frag.radius / my_smallest >= THREAT_MIN_RATIO:
                frag_reach = _movement_speed(frag.radius) * THREAT_IMMEDIATE_HORIZON + frag.radius
                nearest_own_dist = _nearest_own_dist(frag.pos[0], frag.pos[1], my_own_positions)
                if nearest_own_dist <= frag_reach:
                    swarm_has_dangerous_fragment = True

            if my_mass >= frag_mass * EAT_SIZE_RATIO:
                net_speed = _movement_speed(my_largest) - frag_speed
                direct_reach = max(0.0, net_speed) * PROXIMITY_HORIZON + my_largest
                if fdist <= direct_reach:
                    candidate = (2, fdist, fdx, fdy, False)
                    if swarm_eat is None or candidate[0] > swarm_eat[0] or (
                        candidate[0] == swarm_eat[0] and candidate[1] < swarm_eat[1]
                    ):
                        swarm_eat = candidate

            my_largest_mass = my_largest * my_largest
            if my_largest >= frag.radius * SPLIT_EAT_RATIO and my_largest_mass >= SPLIT_MIN_MASS:
                child_radius = math.sqrt(my_largest_mass / 2.0)
                child_speed = _movement_speed(child_radius)
                net_speed = child_speed - frag_speed
                split_reach = (
                    2 * child_radius + _eject_partial(PROXIMITY_HORIZON) + net_speed * PROXIMITY_HORIZON + child_radius
                )
                if fdist <= split_reach and not _virus_blocks_lunge(
                    mx, my, frag.pos[0], frag.pos[1], child_radius, my_blob_count + 1, visible_blobs, visible_viruses
                ):
                    if my_blob_count == 1:
                        candidate = (1, fdist, fdx, fdy, True)
                        if swarm_eat is None or candidate[0] > swarm_eat[0] or (
                            candidate[0] == swarm_eat[0] and candidate[1] < swarm_eat[1]
                        ):
                            swarm_eat = candidate
                    else:
                        resplit_declined = True

        if resplit_declined and swarm_eat is None and not swarm_has_dangerous_fragment:
            continue

        if swarm_eat is not None and not swarm_has_dangerous_fragment:
            if eat_candidate is None or swarm_eat[0] > eat_candidate[0]:
                eat_candidate = swarm_eat
            continue

        if swarm_has_dangerous_fragment:
            continue

        dx, dy = cx - mx, cy - my
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            continue
        typical_r = projected_radius / math.sqrt(len(frags))
        worst_case_reach = _movement_speed(typical_r) * THREAT_IMMEDIATE_HORIZON + EJECT_MAX
        if dist <= worst_case_reach:
            w = 1.0 / dist
            flee_x -= dx * w
            flee_y -= dy * w
            any_flee = True

    if any_flee:
        if flee_x == 0.0 and flee_y == 0.0:
            flee_x, flee_y = 1.0, 0.0
        return ("flee", flee_x, flee_y, False)

    if eat_candidate is not None:
        _, _, dx, dy, split = eat_candidate
        return ("eat", dx, dy, split)

    return (None, 0.0, 0.0, False)


def _pursue_step(px: float, py: float, pr: float, tx: float, ty: float) -> tuple[float, float]:
    dx, dy = tx - px, ty - py
    dist = math.hypot(dx, dy)
    if dist < 1e-9:
        return px, py
    speed = _movement_speed(pr)
    return px + dx / dist * speed, py + dy / dist * speed


def _score_escape_heading(
    mx: float, my: float, my_r: float, hx: float, hy: float,
    threats: list[tuple[float, float, float]], map_size: float, rounds: int,
) -> float:
    speed_me = _movement_speed(my_r)
    x, y = mx, my
    positions = list(threats)
    worst_margin = math.inf
    for _ in range(rounds):
        x = min(max(x + hx * speed_me, my_r), map_size - my_r)
        y = min(max(y + hy * speed_me, my_r), map_size - my_r)
        positions = [_pursue_step(tx, ty, tr, x, y) + (tr,) for (tx, ty, tr) in positions]
        margin = min(math.hypot(x - tx, y - ty) - tr for (tx, ty, tr) in positions)
        worst_margin = min(worst_margin, margin)
    return worst_margin


_multi_threat_heading: tuple[float, float] | None = None

MULTI_THREAT_COMMIT_MARGIN = 0.5

def _pick_escape_heading(mx: float, my: float, my_r: float, threats: list[tuple[float, float, float]], map_size: float):
    global _multi_threat_heading

    candidates = [
        (math.cos(2.0 * math.pi * i / MULTI_THREAT_CANDIDATE_HEADINGS), math.sin(2.0 * math.pi * i / MULTI_THREAT_CANDIDATE_HEADINGS))
        for i in range(MULTI_THREAT_CANDIDATE_HEADINGS)
    ]

    prev_score = None
    if _multi_threat_heading is not None:
        prev_score = _score_escape_heading(mx, my, my_r, *_multi_threat_heading, threats, map_size, MULTI_THREAT_LOOKAHEAD_ROUNDS)

    best_heading = None
    best_score = -math.inf
    for hx, hy in candidates:
        score = _score_escape_heading(mx, my, my_r, hx, hy, threats, map_size, MULTI_THREAT_LOOKAHEAD_ROUNDS)
        if score > best_score:
            best_score = score
            best_heading = (hx, hy)

    if prev_score is not None and prev_score >= best_score - MULTI_THREAT_COMMIT_MARGIN:
        chosen = _multi_threat_heading
    else:
        chosen = best_heading

    _multi_threat_heading = chosen
    return chosen


def _find_imminent_threat(mx, my, my_smallest, my_own_positions, visible_blobs, exit_margin=1.0):
    threats = _find_imminent_threats(mx, my, my_smallest, my_own_positions, visible_blobs, exit_margin)
    return _sum_flee_vector(threats, mx, my)


def _find_imminent_threats(mx, my, my_smallest, my_own_positions, visible_blobs, exit_margin=1.0):
    my_smallest_mass = my_smallest * my_smallest
    gated: list[tuple[float, float, float]] = []
    for blob in visible_blobs:
        nearest_own_dist = _nearest_own_dist(blob.pos[0], blob.pos[1], my_own_positions)

        gated_in = False
        ratio = blob.radius / my_smallest
        if ratio >= REAL_EAT_RATIO:
            close_reach = blob.radius * CLOSE_RANGE_FACTOR
            full_reach = _movement_speed(blob.radius) * THREAT_IMMEDIATE_HORIZON + blob.radius
            if ratio >= THREAT_MIN_RATIO:
                reach = full_reach * exit_margin
            else:
                t = (ratio - REAL_EAT_RATIO) / (THREAT_MIN_RATIO - REAL_EAT_RATIO)
                reach = ((1.0 - t) * close_reach + t * full_reach) * exit_margin
            gated_in = nearest_own_dist <= reach
        if not gated_in and (blob.radius * blob.radius) / my_smallest_mass >= NEAR_EQUAL_MASS_RATIO:
            close_reach = blob.radius * CLOSE_RANGE_FACTOR * exit_margin
            gated_in = nearest_own_dist <= close_reach
        if not gated_in and ratio >= SPLIT_EAT_RATIO and blob.radius * blob.radius >= SPLIT_MIN_MASS:
            child_radius = math.sqrt(blob.radius * blob.radius / 2.0)
            net_speed = _movement_speed(child_radius) - _movement_speed(my_smallest)
            split_reach = (
                2 * child_radius
                + _eject_partial(THREAT_IMMEDIATE_HORIZON)
                + net_speed * THREAT_IMMEDIATE_HORIZON
                + child_radius
            ) * exit_margin
            gated_in = nearest_own_dist <= split_reach
        if not gated_in:
            continue
        gated.append((blob.pos[0], blob.pos[1], blob.radius))
    return gated


def _sum_flee_vector(threats: list[tuple[float, float, float]], mx: float, my: float):
    if not threats:
        return None
    flee_x = flee_y = 0.0
    for tx, ty, _ in threats:
        dx, dy = tx - mx, ty - my
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            continue
        w = 1.0 / dist
        flee_x -= dx * w
        flee_y -= dy * w
    if flee_x == 0.0 and flee_y == 0.0:
        flee_x, flee_y = 1.0, 0.0
    return flee_x, flee_y


def _find_split_attack_target(mx, my, my_r, my_mass, my_blob_count, visible_blobs, visible_viruses):
    if my_blob_count != 1 or my_mass < SPLIT_MIN_MASS:
        return None

    child_radius = math.sqrt(my_mass / 2.0)
    child_speed = _movement_speed(child_radius)

    best = None
    for blob in visible_blobs:
        if my_r < blob.radius * SPLIT_EAT_RATIO:
            continue
        dx, dy = blob.pos[0] - mx, blob.pos[1] - my
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            continue

        frag_speed = _movement_speed(blob.radius)
        net_speed = child_speed - frag_speed
        reach = (
            2 * child_radius + _eject_partial(ATTACK_PROXIMITY_HORIZON) + net_speed * ATTACK_PROXIMITY_HORIZON + child_radius
        )
        if dist > reach:
            continue

        safe = True
        for other in visible_blobs:
            if other is blob:
                continue
            if other.radius / child_radius < THREAT_MIN_RATIO:
                continue
            odist = math.hypot(other.pos[0] - mx, other.pos[1] - my)
            if odist <= THREAT_DETECTION_RANGE:
                safe = False
                break
        if not safe:
            continue

        if _virus_blocks_lunge(mx, my, blob.pos[0], blob.pos[1], child_radius, my_blob_count + 1, visible_blobs, visible_viruses):
            continue

        if best is None or dist < best[0]:
            best = (dist, dx, dy)

    return best


def _find_multi_blob_split_attack_target(me, my_smallest, my_blob_count, visible_blobs, visible_viruses):
    if my_blob_count <= 1 or my_blob_count >= MAX_BLOB_COUNT:
        return None

    largest_blob = max(me.blobs.values(), key=lambda b: b.radius)
    my_largest = largest_blob.radius
    largest_mass = my_largest * my_largest
    if largest_mass < SPLIT_MIN_MASS:
        return None

    my_own_positions = [b.pos for b in me.blobs.values()]
    post_split_blob_count = min(2 * my_blob_count, MAX_BLOB_COUNT)
    worst_case_child_radius = math.sqrt(my_smallest * my_smallest / 2.0)

    child_radius = math.sqrt(largest_mass / 2.0)
    child_speed = _movement_speed(child_radius)
    ax, ay = largest_blob.pos

    best = None
    for blob in visible_blobs:
        if my_largest < blob.radius * SPLIT_EAT_RATIO:
            continue
        dx, dy = blob.pos[0] - ax, blob.pos[1] - ay
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            continue

        frag_speed = _movement_speed(blob.radius)
        net_speed = child_speed - frag_speed
        reach = (
            2 * child_radius + _eject_partial(ATTACK_PROXIMITY_HORIZON) + net_speed * ATTACK_PROXIMITY_HORIZON + child_radius
        )
        if dist > reach:
            continue

        safe = True
        for other in visible_blobs:
            if other is blob:
                continue
            if other.radius / worst_case_child_radius < THREAT_MIN_RATIO:
                continue
            other_nearest_own_dist = _nearest_own_dist(other.pos[0], other.pos[1], my_own_positions)
            if other_nearest_own_dist <= THREAT_DETECTION_RANGE:
                safe = False
                break
        if not safe:
            continue

        if _virus_blocks_lunge(ax, ay, blob.pos[0], blob.pos[1], child_radius, post_split_blob_count, visible_blobs, visible_viruses):
            continue

        if best is None or dist < best[0]:
            best = (dist, dx, dy)

    return best


WALL_CORNER_EAT_RATIO = 2.0 + math.sqrt(2.0)
CORNER_WALL_SLACK = 0.5


def _is_wall_cornered(x: float, y: float, r: float, size: float) -> bool:
    near_x = x <= r + CORNER_WALL_SLACK or x >= size - r - CORNER_WALL_SLACK
    near_y = y <= r + CORNER_WALL_SLACK or y >= size - r - CORNER_WALL_SLACK
    return near_x and near_y


def _find_corner_split_target(me, visible_blobs, visible_viruses, size, my_blob_count):
    best = None
    for blob_obj in me.blobs.values():
        br = blob_obj.radius
        b_mass = br * br
        if b_mass < SPLIT_MIN_MASS:
            continue
        bx, by = blob_obj.pos
        child_radius = math.sqrt(b_mass / 2.0)

        for target in visible_blobs:
            tr = target.radius
            if br <= tr * WALL_CORNER_EAT_RATIO:
                continue
            if child_radius <= tr * WALL_CORNER_EAT_RATIO:
                pass
            else:
                continue
            if not _is_wall_cornered(target.pos[0], target.pos[1], tr, size):
                continue

            dx, dy = target.pos[0] - bx, target.pos[1] - by
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                continue

            safe = True
            for other in visible_blobs:
                if other is target:
                    continue
                if other.radius / child_radius < THREAT_MIN_RATIO:
                    continue
                odist = math.hypot(other.pos[0] - bx, other.pos[1] - by)
                if odist <= THREAT_DETECTION_RANGE:
                    safe = False
                    break
            if not safe:
                continue

            if _virus_blocks_lunge(bx, by, target.pos[0], target.pos[1], child_radius, my_blob_count + 1, visible_blobs, visible_viruses):
                continue

            if best is None or dist < best[0]:
                best = (dist, dx, dy)

    return best


VIRUS_FLEE_EXIT_MARGIN = 1.3
_virus_flee_active = False

THREAT_IMMEDIATE_EXIT_MARGIN = 1.3
_imminent_flee_active = False

NEAR_EQUAL_MASS_RATIO = 1.0
CLOSE_RANGE_FACTOR = 1.3

PREY_STUCK_WINDOW = 20
PREY_STUCK_MIN_IMPROVEMENT = 0.3
_prey_dist_history: list[float] = []


def _prey_chase_is_stalled(nearest_prey_dist: float) -> bool:
    _prey_dist_history.append(nearest_prey_dist)
    if len(_prey_dist_history) > PREY_STUCK_WINDOW:
        _prey_dist_history.pop(0)
    if len(_prey_dist_history) < PREY_STUCK_WINDOW:
        return False
    improvement = _prey_dist_history[0] - min(_prey_dist_history)
    return improvement < PREY_STUCK_MIN_IMPROVEMENT


def compute_steering(me, visible_food, visible_blobs, visible_viruses, map_size, am_leading):
    global _virus_flee_active, _imminent_flee_active, _multi_threat_heading

    mx, my = me.x, me.y
    my_r = me.radius
    my_mass = my_r * my_r
    my_blob_count = len(me.blobs)
    my_virus_blob_count = _predicted_own_blob_count(me.blobs.values())

    reach = my_r + _movement_speed(my_r)
    imminent_x = imminent_y = 0.0
    any_imminent = False
    exit_margin = VIRUS_FLEE_EXIT_MARGIN if _virus_flee_active else 1.0
    for virus in visible_viruses:
        if my_r / virus.radius < VIRUS_DANGER_RATIO:
            continue
        dx, dy = virus.pos[0] - mx, virus.pos[1] - my
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            continue
        if dist > reach * exit_margin:
            continue
        if not _virus_pop_is_dangerous(my_r, my_virus_blob_count, virus.radius, visible_blobs):
            continue
        w = 1.0 / dist
        imminent_x -= dx * w
        imminent_y -= dy * w
        any_imminent = True

    if any_imminent:
        for blob in visible_blobs:
            if blob.radius / my_r < THREAT_MIN_RATIO:
                continue
            if math.hypot(blob.pos[0] - mx, blob.pos[1] - my) <= THREAT_DETECTION_RANGE:
                any_imminent = False
                break

    _virus_flee_active = any_imminent

    if any_imminent:
        if imminent_x == 0.0 and imminent_y == 0.0:
            imminent_x, imminent_y = 1.0, 0.0
        imminent_x, imminent_y = _redirect_along_wall(imminent_x, imminent_y, mx, my, my_r, map_size)
        return imminent_x, imminent_y, False

    my_blob_radii = [b.radius for b in me.blobs.values()]
    my_smallest = min(my_blob_radii)
    my_largest = max(my_blob_radii)
    my_own_positions = [b.pos for b in me.blobs.values()]

    action, adx, ady, split = _resolve_swarm_threats(
        mx, my, my_mass, my_smallest, my_largest, my_blob_count, my_own_positions, visible_blobs, visible_viruses
    )
    if action == "flee":
        adx, ady = _redirect_along_wall(adx, ady, mx, my, my_r, map_size)
        return adx, ady, False
    if action == "eat":
        return adx, ady, split

    attack = _find_split_attack_target(mx, my, my_r, my_mass, my_blob_count, visible_blobs, visible_viruses)
    if attack is not None:
        _, adx, ady = attack
        return adx, ady, True

    multi_attack = _find_multi_blob_split_attack_target(me, my_smallest, my_blob_count, visible_blobs, visible_viruses)
    if multi_attack is not None:
        _, adx, ady = multi_attack
        return adx, ady, True

    if my_blob_count < MAX_BLOB_COUNT:
        corner_attack = _find_corner_split_target(me, visible_blobs, visible_viruses, map_size, my_blob_count)
        if corner_attack is not None:
            _, adx, ady = corner_attack
            return adx, ady, True

    imminent_exit_margin = THREAT_IMMEDIATE_EXIT_MARGIN if _imminent_flee_active else 1.0
    imminent_threats = _find_imminent_threats(mx, my, my_smallest, my_own_positions, visible_blobs, imminent_exit_margin)
    _imminent_flee_active = bool(imminent_threats)
    if imminent_threats:
        wx, wy = _update_wall_proximity(mx, my, my_r, map_size)
        if wx != 0.0 or wy != 0.0:
            escape = _pick_escape_heading(mx, my, my_r, imminent_threats, map_size)
            if escape is not None:
                return escape[0], escape[1], False
        flee_x, flee_y = _sum_flee_vector(imminent_threats, mx, my)
        adx, ady = _apply_wall_redirect(flee_x, flee_y, wx, wy)
        return adx, ady, False

    prey_x = prey_y = 0.0
    threat_x = threat_y = 0.0
    nearest_prey_dist = None
    threat_list: list[tuple[float, float, float]] = []
    blob_threat_mag_sum = 0.0
    has_giant_threat = False
    for blob in visible_blobs:
        dx, dy = blob.pos[0] - mx, blob.pos[1] - my
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            continue
        nbx, nby = min(
            my_own_positions,
            key=lambda p: (blob.pos[0] - p[0]) ** 2 + (blob.pos[1] - p[1]) ** 2,
        )
        nearest_own_dist = math.hypot(blob.pos[0] - nbx, blob.pos[1] - nby)
        threat_ratio = blob.radius / my_smallest
        prey_ratio = blob.radius / my_largest
        if threat_ratio >= THREAT_MIN_RATIO:
            if nearest_own_dist > THREAT_DETECTION_RANGE_THREAT:
                continue
            w = (blob.radius * blob.radius) / dist**P_PREY_THREAT
            tx, ty = dx * w, dy * w
            threat_x += tx
            threat_y += ty
            blob_threat_mag_sum += math.hypot(tx, ty)
            threat_list.append((blob.pos[0], blob.pos[1], blob.radius))
            if threat_ratio >= SPLIT_EAT_RATIO:
                has_giant_threat = True
        elif prey_ratio <= PREY_MAX_RATIO:
            if nearest_own_dist > THREAT_DETECTION_RANGE:
                continue
            if nearest_own_dist < 1e-6:
                continue
            pdx, pdy = blob.pos[0] - nbx, blob.pos[1] - nby
            if nearest_prey_dist is None or nearest_own_dist < nearest_prey_dist:
                nearest_prey_dist = nearest_own_dist
            w = (blob.radius * blob.radius) / nearest_own_dist**P_PREY_THREAT
            prey_x += pdx * w
            prey_y += pdy * w

    if nearest_prey_dist is not None and _prey_chase_is_stalled(nearest_prey_dist):
        prey_x = prey_y = 0.0

    blob_threat_x, blob_threat_y = threat_x, threat_y

    dangerous_viruses = set()
    for i, virus in enumerate(visible_viruses):
        if my_r / virus.radius < VIRUS_DANGER_RATIO:
            continue
        if not _virus_pop_is_dangerous(my_r, my_virus_blob_count, virus.radius, visible_blobs):
            continue
        dx, dy = virus.pos[0] - mx, virus.pos[1] - my
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            continue
        w = (virus.radius * virus.radius) / dist**P_PREY_THREAT
        threat_x += dx * w
        threat_y += dy * w
        dangerous_viruses.add(i)

    if prey_x or prey_y or threat_x or threat_y:
        if prey_x == 0.0 and prey_y == 0.0 and len(threat_list) >= 2 and not has_giant_threat and blob_threat_mag_sum > 1e-9:
            cancel_ratio = math.hypot(blob_threat_x, blob_threat_y) / blob_threat_mag_sum
            if cancel_ratio < MULTI_THREAT_CANCELLATION_RATIO:
                escape = _pick_escape_heading(mx, my, my_r, threat_list, map_size)
                if escape is not None:
                    return escape[0], escape[1], False
            else:
                _multi_threat_heading = None
        else:
            _multi_threat_heading = None

        steer_x = C_PREY * prey_x - C_THREAT * threat_x
        steer_y = C_PREY * prey_y - C_THREAT * threat_y
        if steer_x == 0.0 and steer_y == 0.0:
            steer_x, steer_y = 1.0, 0.0
        threat_mag = math.hypot(C_THREAT * threat_x, C_THREAT * threat_y)
        prey_mag = math.hypot(C_PREY * prey_x, C_PREY * prey_y)
        if threat_mag > prey_mag:
            steer_x, steer_y = _redirect_along_wall(steer_x, steer_y, mx, my, my_r, map_size)
        return steer_x, steer_y, False

    steer_x = steer_y = 0.0
    for i, virus in enumerate(visible_viruses):
        if my_r / virus.radius < VIRUS_BAIT_RATIO:
            continue
        if i in dangerous_viruses:
            continue
        dx, dy = virus.pos[0] - mx, virus.pos[1] - my
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            continue
        w = C_VIRUS * dist ** (-P_VIRUS)
        steer_x += dx * w
        steer_y += dy * w

    if steer_x == 0.0 and steer_y == 0.0:
        for food in visible_food:
            if not _is_food_reachable(food.pos[0], food.pos[1], my_r, map_size):
                continue
            nx, ny = min(
                my_own_positions,
                key=lambda p: (food.pos[0] - p[0]) ** 2 + (food.pos[1] - p[1]) ** 2,
            )
            dx, dy = food.pos[0] - nx, food.pos[1] - ny
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                continue
            w = C_FOOD * dist ** (-P_FOOD)
            steer_x += dx * w
            steer_y += dy * w

    if steer_x == 0.0 and steer_y == 0.0:
        steer_x, steer_y = map_size / 2.0 - mx, map_size / 2.0 - my
        if steer_x == 0.0 and steer_y == 0.0:
            steer_x, steer_y = 1.0, 0.0

    sum_of_radii = sum(my_blob_radii)
    if am_leading and my_blob_count < MAX_BLOB_COUNT and sum_of_radii > VISION_SPLIT_MIN_RADIUS:
        child_radius = math.sqrt(my_smallest * my_smallest / 2.0)
        post_split_blob_count = min(2 * my_blob_count, MAX_BLOB_COUNT)
        if _vision_split_is_safe(my_own_positions, post_split_blob_count, child_radius, visible_blobs, visible_viruses):
            return steer_x, steer_y, True

    return steer_x, steer_y, False


def choose_move(game: Game) -> MovePlayer:
    me = game.state.me
    rankings = game.state.rankings
    am_leading = bool(rankings) and rankings[0] == me.player_id
    steer_x, steer_y, split = compute_steering(
        me,
        game.state.visible_food,
        game.state.visible_blobs,
        game.state.visible_viruses,
        game.state.map.size,
        am_leading,
    )
    return MovePlayer(
        player_id=me.player_id,
        direction=DirectionModel(x=steer_x, y=steer_y),
        split=split,
    )


def main() -> None:
    game = Game()

    while True:
        query = game.get_next_query()
        match query:
            case QueryMovePlayer():
                game.send_move(choose_move(game))
            case _:
                raise RuntimeError(f"Unsupported query type: {type(query)}")


if __name__ == "__main__":
    main()

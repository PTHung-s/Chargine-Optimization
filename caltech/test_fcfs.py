import json
import pulp
from datetime import datetime, timedelta
import os
import random
import math

# ========= Optional: torch for DQN =========
USE_TORCH = True
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    USE_TORCH = False
    print("[DRL] PyTorch chưa được cài. DRL sẽ fallback sang PV-first baseline.")

# ---------------------------
# Cấu hình chung
# ---------------------------
STATION_LIMIT = 80  # số cổng sạc tối đa theo giờ (áp dụng cho LP, PV-first, DRL)

# ---------------------------
# Hàm đọc dữ liệu từ file
# ---------------------------
def read_json_files():
    with open("../Loc_data_price/price.json", "r") as f:
        price_data = json.load(f)
    with open("../Loc_solar/gop_solar.json", "r") as f:
        solar_data = json.load(f)
    with open("./final_gop.json", "r") as f:
        final_gop_data = json.load(f)
    with open("./max.json", "r") as f:
        max_data = json.load(f)
    return price_data, solar_data, final_gop_data, max_data

def parse_time(time_str):
    return datetime.strptime(time_str.replace("GMT", "").strip(), "%a, %d %b %Y %H:%M:%S")

def parse_final_gop_dates(final_gop_data, start_date, end_date):
    mapping = {}
    for key in final_gop_data:
        try:
            dt = datetime.strptime(key, "%d-%b-%Y")
        except Exception:
            continue
        if start_date <= dt <= end_date:
            mapping[dt] = key
    sorted_dates = sorted(mapping.keys())
    return sorted_dates, mapping

def create_overlapping_blocks(sorted_dates, final_gop_data, mapping):
    blocks = []
    pending = []
    if len(sorted_dates) < 2:
        block = {"dates": [sorted_dates[0]], "sessions": final_gop_data[mapping[sorted_dates[0]]]}
        blocks.append(block)
        return blocks

    for i in range(1, len(sorted_dates)):
        day_prev = sorted_dates[i-1]
        day_curr = sorted_dates[i]
        block_dates = [day_prev, day_curr]
        block_start = datetime.combine(day_prev, datetime.min.time())
        block_end = datetime.combine(day_curr, datetime.min.time()) + timedelta(days=1)

        sessions_prev = final_gop_data.get(mapping[day_prev], [])
        sessions_curr = final_gop_data.get(mapping[day_curr], [])
        block_sessions = pending + sessions_prev + sessions_curr

        new_block_sessions = []
        new_pending = []
        for session in block_sessions:
            dtime = parse_time(session["disconnectTime"])
            if dtime >= block_end:
                new_pending.append(session)
            else:
                new_block_sessions.append(session)

        block = {"dates": block_dates, "sessions": new_block_sessions}
        blocks.append(block)
        pending = new_pending

    if pending:
        last_date = sorted_dates[-1]
        block_start = datetime.combine(last_date, datetime.min.time())
        block_end = block_start + timedelta(days=1)
        final_sessions = []
        new_pending = []
        for session in pending:
            dtime = parse_time(session["disconnectTime"])
            if dtime >= block_end:
                new_pending.append(session)
            else:
                final_sessions.append(session)
        block = {"dates": [last_date], "sessions": final_sessions}
        blocks.append(block)
        pending = new_pending
    return blocks

def get_block_data_from_block(block, price_data, solar_data, max_data):
    block_dates = block["dates"]
    sessions = block["sessions"]
    block_start = datetime.combine(block_dates[0], datetime.min.time())
    if len(block_dates) > 1:
        block_end = datetime.combine(block_dates[-1], datetime.min.time()) + timedelta(days=1)
    else:
        block_end = block_start + timedelta(days=1)
    T = int((block_end - block_start).total_seconds() / 3600)

    # p_grid
    p_grid = []
    current = block_start
    while current < block_end:
        day_str = current.strftime("%Y-%m-%d")
        hour_index = current.hour
        price = price_data.get(day_str, [0]*24)[hour_index] if day_str in price_data else 0
        price = price / 1000.0
        p_grid.append(price)
        current += timedelta(hours=1)

    # R
    R = []
    current = block_start
    while current < block_end:
        solar_key = current.strftime("%Y%m%d")
        hour_index = current.hour
        if solar_key in solar_data:
            R_val = solar_data[solar_key][hour_index]["R(i)"]
        else:
            R_val = 0
        R.append(R_val)
        current += timedelta(hours=1)

    sessions_sorted = sorted(sessions, key=lambda s: parse_time(s["connectionTime"]))

    A_matrix, L_req, conn_times = [], [], []
    last_t_index = []
    for session in sessions_sorted:
        conn = parse_time(session["connectionTime"])
        disc = parse_time(session["disconnectTime"])
        session_start = max(conn, block_start)
        session_end = min(disc, block_end)
        availability = []
        for t in range(T):
            slot_start = block_start + timedelta(hours=t)
            slot_end = slot_start + timedelta(hours=1)
            eff_start = max(slot_start, session_start)
            eff_end = min(slot_end, session_end)
            if eff_end > eff_start:
                fraction = (eff_end - eff_start).total_seconds() / 3600.0
                fraction = min(fraction, 1.0)
            else:
                fraction = 0.0
            availability.append(fraction)
        A_matrix.append(availability)
        L_req.append(session["kWhDelivered"])
        conn_times.append(conn)
        lt = -1
        for t in range(T):
            if availability[t] > 0:
                lt = t
        last_t_index.append(lt)

    s = max_data["doubled_max_rate"]
    data = {
        "T": T,
        "N": len(sessions_sorted),
        "sessions_sorted": sessions_sorted,
        "A": A_matrix,
        "L_req": L_req,
        "conn_times": conn_times,
        "last_t_index": last_t_index,
        "p_grid": p_grid,
        "R": R,
        "s": s,
        "eta": 0.9,
        "C_grid": 300,
        "delta_t": 1
    }
    return data

# ---------------------------
# LP — đã khóa PV và giới hạn cổng (công suất)
# ---------------------------
def build_model(data):
    T, N = data["T"], data["N"]
    A, L_req, s, eta = data["A"], data["L_req"], data["s"], data["eta"]
    p_grid, R_list, C_grid, delta_t = data["p_grid"], data["R"], data["C_grid"], data["delta_t"]

    problem = pulp.LpProblem("EV_Charging_Optimization", pulp.LpMinimize)

    Y = pulp.LpVariable.dicts("Y", ((i, t) for i in range(N) for t in range(T)), lowBound=0)
    S_plus = pulp.LpVariable.dicts("S_plus", (t for t in range(T)), lowBound=0)
    R_used = pulp.LpVariable.dicts("R_used", (t for t in range(T)), lowBound=0)

    problem += pulp.lpSum([p_grid[t] * S_plus[t] * delta_t for t in range(T)])

    for t in range(T):
        total_load = pulp.lpSum([Y[(i, t)] for i in range(N)])
        problem += total_load <= STATION_LIMIT * s, f"StationLimit_t_{t}"
        problem += total_load - R_used[t] <= C_grid, f"GridLimit_t_{t}"
        problem += R_used[t] <= R_list[t], f"PVmax_t_{t}"
        problem += R_used[t] <= total_load, f"PV_le_total_t_{t}"
        problem += S_plus[t] >= total_load - R_used[t], f"Splus_ge_net_t_{t}"

    for i in range(N):
        T_i = [t for t in range(T) if A[i][t] > 0]
        problem += eta * pulp.lpSum([Y[(i, t)] * delta_t for t in T_i]) >= L_req[i], f"EnergyReq_EV_{i}"
        for t in range(T):
            problem += Y[(i, t)] <= s, f"MaxPower_EV_{i}_t_{t}"
            problem += Y[(i, t)] <= s * A[i][t], f"Presence_EV_{i}_t_{t}"

    return problem, Y, S_plus, R_used

def solve_model(problem):
    problem.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[problem.status]
    obj_val = pulp.value(problem.objective)
    return status, obj_val

# ---------------------------
# PV-first baseline (đồng nhất với LP): giới hạn theo **công suất cổng**
# ---------------------------
def do_greedy_pv_first_block(data):
    T, N = data["T"], data["N"]
    A, L_req, s, eta = data["A"], data["L_req"], data["s"], data["eta"]
    p_grid, R_list, C_grid = data["p_grid"], data["R"], data["C_grid"]

    needed = [L_req[i] / eta for i in range(N)]
    X = [[0.0]*T for _ in range(N)]

    for t in range(T):
        present = [i for i in range(N) if needed[i] > 1e-9 and A[i][t] > 0]
        order = sorted(present, key=lambda i: data["last_t_index"][i] if data["last_t_index"][i] >= 0 else 10**9)

        pv_leftover = R_list[t]
        grid_leftover = C_grid
        port_left = STATION_LIMIT * s  # ✅ công suất cổng còn lại giờ t

        # PV phase (EDF)
        for i in order:
            if port_left <= 1e-12: break
            cap = min(s * A[i][t], port_left)
            deliver = min(cap, needed[i], pv_leftover)
            if deliver > 1e-12:
                X[i][t] += deliver
                needed[i] -= deliver
                pv_leftover -= deliver
                port_left -= deliver

        # Grid phase (EDF + scale)
        candidates = [i for i in order if needed[i] > 1e-9]
        if candidates and grid_leftover > 1e-12 and port_left > 1e-12:
            remaining_caps = []
            total_possible = 0.0
            for i in candidates:
                cap = max(0.0, min(s * A[i][t] - X[i][t], port_left))
                rem = min(cap, needed[i])
                if rem > 1e-12:
                    remaining_caps.append((i, rem))
                    total_possible += rem
            if total_possible > 1e-12:
                limit = min(grid_leftover, port_left)
                scale = min(1.0, limit / total_possible)
                for i, rem in remaining_caps:
                    give = rem * scale
                    if give > 1e-12:
                        X[i][t] += give
                        needed[i] -= give
                        grid_leftover -= give
                        port_left -= give

    total_cost = 0.0
    for t in range(T):
        total_load = sum(X[i][t] for i in range(N))
        used_R = min(R_list[t], total_load)
        total_cost += p_grid[t] * (total_load - used_R)
    return total_cost

def do_greedy_pv_first_with_deficit(data):
    T, N = data["T"], data["N"]
    A, L_req, s, eta = data["A"], data["L_req"], data["s"], data["eta"]
    p_grid, R_list, C_grid = data["p_grid"], data["R"], data["C_grid"]

    needed = [L_req[i] / eta for i in range(N)]
    X = [[0.0]*T for _ in range(N)]

    for t in range(T):
        present = [i for i in range(N) if needed[i] > 1e-9 and A[i][t] > 0]
        order = sorted(present, key=lambda i: data["last_t_index"][i] if data["last_t_index"][i] >= 0 else 10**9)

        pv_leftover = R_list[t]
        grid_leftover = C_grid
        port_left = STATION_LIMIT * s

        for i in order:
            if port_left <= 1e-12: break
            cap = min(s * A[i][t], port_left)
            deliver = min(cap, needed[i], pv_leftover)
            if deliver > 1e-12:
                X[i][t] += deliver
                needed[i] -= deliver
                pv_leftover -= deliver
                port_left -= deliver

        candidates = [i for i in order if needed[i] > 1e-9]
        if candidates and grid_leftover > 1e-12 and port_left > 1e-12:
            remaining_caps = []
            total_possible = 0.0
            for i in candidates:
                cap = max(0.0, min(s * A[i][t] - X[i][t], port_left))
                rem = min(cap, needed[i])
                if rem > 1e-12:
                    remaining_caps.append((i, rem))
                    total_possible += rem
            if total_possible > 1e-12:
                limit = min(grid_leftover, port_left)
                scale = min(1.0, limit / total_possible)
                for i, rem in remaining_caps:
                    give = rem * scale
                    if give > 1e-12:
                        X[i][t] += give
                        needed[i] -= give
                        grid_leftover -= give
                        port_left -= give

    total_cost = 0.0
    for t in range(T):
        total_load = sum(X[i][t] for i in range(N))
        used_R = min(R_list[t], total_load)
        total_cost += p_grid[t] * (total_load - used_R)

    deficit_vehicle_kWh = data["eta"] * sum(max(0.0, needed[i]) for i in range(N) if data["last_t_index"][i] >= 0)
    return total_cost, deficit_vehicle_kWh

# ---------------------------
# ======== DRL (DQN) Hard Constraint + Lookahead + Unified Cost + AUDIT ========
# ---------------------------
DRL_CFG = {
    "episodes": 2,
    "gamma": 0.98,
    "lr": 1e-3,
    "batch_size": 128,
    "replay_size": 50000,
    "epsilon_start": 0.2,
    "epsilon_end": 0.05,
    "epsilon_decay_steps": 2000,
    "hidden": 64,
    "huge_penalty": 1e6,
}

class ReplayBuffer:
    def __init__(self, cap):
        self.cap = cap
        self.buf = []
        self.idx = 0
    def push(self, item):
        if len(self.buf) < self.cap:
            self.buf.append(item)
        else:
            self.buf[self.idx] = item
        self.idx = (self.idx + 1) % self.cap
    def sample(self, bs):
        return random.sample(self.buf, bs)
    def __len__(self):
        return len(self.buf)

if USE_TORCH:
    class QNet(nn.Module):
        def __init__(self, in_dim, hidden=64, out_dim=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, out_dim)
            )
        def forward(self, x):
            return self.net(x)

def _soft_update(target, online, tau=0.01):
    for tp, p in zip(target.parameters(), online.parameters()):
        tp.data.copy_(tp.data*(1-tau) + p.data*tau)

def _build_state_vec(data, t, i, needed, norms):
    A = data["A"]; p = data["p_grid"]; R = data["R"]; s = data["s"]; Cg = data["C_grid"]
    last_idx = data["last_t_index"][i] if "last_t_index" in data else t
    ttd = max(0, last_idx - t + 1)
    avail_frac = A[i][t]; cap = s * avail_frac
    n_present = sum(1 for j in range(data["N"]) if needed[j] > 1e-9 and A[j][t] > 0)
    def nz(x, d): return x/d if d > 0 else 0.0
    return [
        nz(p[t], norms["p_max"]),
        nz(R[t], norms["R_max"]),
        nz(Cg, norms["Cg_max"]),
        nz(n_present, norms["N_max"]),
        nz(avail_frac, 1.0),
        nz(needed[i], norms["need_max"]),
        nz(cap, norms["cap_max"]),
        nz(ttd, norms["T_max"])
    ]

def _future_deliverable_upper_bound(data, t, needed):
    T, N, s = data["T"], data["N"], data["s"]
    A, R, Cg = data["A"], data["R"], data["C_grid"]
    total = 0.0
    for tau in range(t+1, T):
        cap_tau = 0.0
        for i in range(N):
            if needed[i] > 1e-9 and A[i][tau] > 0:
                cap_tau += s * A[i][tau]
        cap_tau_ports = min(STATION_LIMIT * s, cap_tau)   # ✅ trần bởi số cổng
        deliver_tau   = min(Cg + R[tau], cap_tau_ports)   # ✅ và lưới+PV
        total += deliver_tau
    return total

def do_drl_dqn_block(data, cfg=DRL_CFG):
    if not USE_TORCH:
        print("[DRL] Fallback: dùng PV-first vì thiếu PyTorch.")
        cost, deficit = do_greedy_pv_first_with_deficit(data)
        return cost, deficit, True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T, N = data["T"], data["N"]
    if N == 0 or T == 0:
        return 0.0, 0.0, False

    norms = {
        "p_max": max(1e-6, max(data["p_grid"]) if data["p_grid"] else 1.0),
        "R_max": max(1e-6, max(data["R"]) if data["R"] else 1.0),
        "Cg_max": max(1e-6, float(data["C_grid"])),
        "N_max": max(1, N),
        "need_max": max(1e-6, max((req / data["eta"]) for req in data["L_req"])) if data["L_req"] else 1.0,
        "cap_max": max(1e-6, data["s"]),
        "T_max": max(1, T)
    }

    state_dim = 8
    q = QNet(state_dim, hidden=cfg["hidden"]).to(device)
    q_tgt = QNet(state_dim, hidden=cfg["hidden"]).to(device)
    q_tgt.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=cfg["lr"])
    loss_fn = nn.MSELoss()
    rb = ReplayBuffer(cfg["replay_size"])

    eps_start, eps_end, eps_decay = cfg["epsilon_start"], cfg["epsilon_end"], cfg["epsilon_decay_steps"]
    global_step = 0
    def eps_val():
        nonlocal global_step
        return eps_end + (eps_start - eps_end) * math.exp(-1.0 * global_step / max(1, eps_decay))

    def simulate_one_episode(train=True, greedy=False):
        nonlocal global_step
        A, L_req, eta = data["A"], data["L_req"], data["eta"]
        s_max, Cg = data["s"], data["C_grid"]
        p, R = data["p_grid"], data["R"]
        last_idx = data["last_t_index"]

        needed = [L_req[i]/eta for i in range(N)]
        total_cost = 0.0

        for t in range(T):
            present = [i for i in range(N) if needed[i] > 1e-9 and A[i][t] > 0]
            if not present:
                continue

            # RL actions (on/off)
            z = {}
            states = {i: _build_state_vec(data, t, i, needed, norms) for i in present}
            for i in present:
                st = torch.tensor(states[i], dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    qv = q(st)[0]
                if greedy:
                    a = int(torch.argmax(qv).item())
                else:
                    e = eps_val()
                    a = int(torch.argmax(qv).item())
                    if random.random() < e:
                        a = random.randint(0, 1)
                z[i] = a

            # ✅ Giới hạn công suất cổng theo giờ
            port_left = STATION_LIMIT * s_max

            # ===== PV-first (EDF)
            edf = sorted(present, key=lambda i: last_idx[i] if last_idx[i] >= 0 else 10**9)
            pv_left = R[t]
            delivered_pv = {i: 0.0 for i in present}
            for i in edf:
                if port_left <= 1e-12 or pv_left <= 1e-12: break
                cap = min(s_max * A[i][t], port_left)
                deliver = min(cap, needed[i], pv_left)
                if deliver > 1e-12:
                    delivered_pv[i] = deliver
                    needed[i] -= deliver
                    pv_left -= deliver
                    port_left -= deliver

            # ===== Grid mandatory (deadline hard)
            deadline_set = [i for i in present if last_idx[i] == t and needed[i] > 1e-9]
            delivered_grid = {i: 0.0 for i in present}
            grid_left = Cg
            infeasible_deadline = False
            if deadline_set:
                tmp_caps = []
                cap_sum = 0.0
                for i in sorted(deadline_set, key=lambda i: last_idx[i]):
                    if port_left <= 1e-12 or grid_left <= 1e-12: break
                    cap_i = max(0.0, min(s_max*A[i][t] - delivered_pv[i], port_left))
                    cap_i = min(cap_i, needed[i])
                    if cap_i > 1e-12:
                        tmp_caps.append((i, cap_i))
                        cap_sum += cap_i
                if cap_sum <= 1e-12:
                    infeasible_deadline = True
                else:
                    limit = min(grid_left, port_left)
                    scale = min(1.0, limit / cap_sum)
                    for i, cap_i in tmp_caps:
                        give = cap_i * scale
                        delivered_grid[i] += give
                        needed[i] -= give
                        grid_left -= give
                        port_left -= give
                    # nếu vẫn còn nhu cầu ở deadline
                    rem_mand = sum(max(0.0, min(s_max*A[i][t] - delivered_pv[i], needed[i])) for i in deadline_set)
                    if rem_mand > 1e-9:
                        infeasible_deadline = True

            # ===== Lookahead reservation
            rem_after_pv = sum(needed[i] for i in range(N))
            fut_up = _future_deliverable_upper_bound(data, t, needed)
            req_now = max(0.0, rem_after_pv - fut_up)
            req_now = min(req_now, grid_left, port_left)

            if req_now > 1e-12:
                non_deadline = [i for i in present if i not in deadline_set and needed[i] > 1e-9]
                order_nd = sorted(non_deadline, key=lambda i: last_idx[i] if last_idx[i] >= 0 else 10**9)
                rem_caps = []
                total_cap = 0.0
                for i in order_nd:
                    if req_now <= 1e-12 or port_left <= 1e-12: break
                    cap_i = max(0.0, min(s_max*A[i][t] - delivered_pv[i] - delivered_grid[i], port_left))
                    cap_i = min(cap_i, needed[i])
                    if cap_i > 1e-12:
                        rem_caps.append((i, cap_i))
                        total_cap += cap_i
                if total_cap > 1e-12:
                    scale = min(1.0, req_now / total_cap)
                    for i, cap_i in rem_caps:
                        give = cap_i * scale
                        delivered_grid[i] += give
                        needed[i] -= give
                        grid_left -= give
                        port_left -= give

            # ===== RL actions for remaining grid
            if grid_left > 1e-12 and port_left > 1e-12:
                non_deadline = [i for i in present if needed[i] > 1e-9 and z[i] == 1]
                order_rl = sorted(non_deadline, key=lambda i: last_idx[i] if last_idx[i] >= 0 else 10**9)
                rem_caps = []
                total_want = 0.0
                for i in order_rl:
                    if grid_left <= 1e-12 or port_left <= 1e-12: break
                    cap_i = max(0.0, min(s_max*A[i][t] - delivered_pv[i] - delivered_grid[i], port_left))
                    cap_i = min(cap_i, needed[i])
                    if cap_i > 1e-12:
                        rem_caps.append((i, cap_i))
                        total_want += cap_i
                if total_want > 1e-12:
                    limit = min(grid_left, port_left)
                    scale = min(1.0, limit / total_want)
                    for i, cap_i in rem_caps:
                        give = cap_i * scale
                        delivered_grid[i] += give
                        needed[i] -= give
                        grid_left -= give
                        port_left -= give

            # Unified cost
            grid_import_t = sum(delivered_grid.values())
            total_cost += p[t] * grid_import_t

            # Replay
            for i in present:
                r_i = -p[t] * delivered_grid[i]
                if infeasible_deadline and i in deadline_set and needed[i] > 1e-6:
                    r_i -= cfg["huge_penalty"]
                if train:
                    done = (t == T-1) or (needed[i] <= 1e-9) or (last_idx[i] <= t)
                    if not done and A[i][t+1] > 0 and needed[i] > 1e-9:
                        s_next = _build_state_vec(data, t+1, i, needed, norms)
                    else:
                        s_next = [0.0]*8
                        done = True
                    rb.push((states[i], z[i], r_i, s_next, float(done)))

            if train and len(rb) >= cfg["batch_size"] and cfg["batch_size"] > 0:
                batch = rb.sample(cfg["batch_size"])
                s_batch = torch.tensor([b[0] for b in batch], dtype=torch.float32, device=device)
                a_batch = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
                r_batch = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)
                sn_batch = torch.tensor([b[3] for b in batch], dtype=torch.float32, device=device)
                d_batch = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=device)

                q_pred = q(s_batch).gather(1, a_batch.view(-1,1)).squeeze(1)
                with torch.no_grad():
                    q_next = q_tgt(sn_batch).max(1)[0]
                    target = r_batch + cfg["gamma"] * (1.0 - d_batch) * q_next
                loss = loss_fn(q_pred, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
                _soft_update(q_tgt, q, tau=0.01)
            global_step += 1

        deficit_vehicle_kWh = data["eta"] * sum(max(0.0, needed[i]) for i in range(N) if data["last_t_index"][i] >= 0)
        return total_cost, deficit_vehicle_kWh

    for ep in range(cfg["episodes"]):
        simulate_one_episode(train=True, greedy=False)

    with torch.no_grad():
        eval_cost, eval_deficit = simulate_one_episode(train=False, greedy=True)

    return eval_cost, eval_deficit, False

# ---------------------------
# Main
# ---------------------------
def main():
    start_date = datetime.strptime("25-04-2018", "%d-%m-%Y")
    end_date = datetime.strptime("31-12-2019", "%d-%m-%Y")
    
    price_data, solar_data, final_gop_data, max_data = read_json_files()
    
    filtered_price = {}
    for k, v in price_data.items():
        try:
            dt = datetime.strptime(k, "%Y-%m-%d")
            if start_date <= dt <= end_date:
                filtered_price[k] = v
        except Exception:
            continue
    price_data = filtered_price

    filtered_solar = {}
    for k, v in solar_data.items():
        try:
            dt = datetime.strptime(k, "%Y%m%d")
            if start_date <= dt <= end_date:
                filtered_solar[k] = v
        except Exception:
            continue
    solar_data = filtered_solar

    sorted_dates, mapping = parse_final_gop_dates(final_gop_data, start_date, end_date)
    if not sorted_dates:
        print("Không có dữ liệu trong khoảng ngày được chọn!")
        return

    blocks = create_overlapping_blocks(sorted_dates, final_gop_data, mapping)
    print(f"Tổng số block cần xử lý: {len(blocks)}")

    f_opt = open("results.jsonl", "a", encoding="utf-8")
    f_pv   = open("PVFIRST.jsonl", "a", encoding="utf-8")
    f_drl  = open("DRL.jsonl", "a", encoding="utf-8")
    
    monthly_results_opt = {}
    monthly_results_pv = {}
    monthly_results_drl = {}

    for idx, block in enumerate(blocks, 1):
        block_dates = block["dates"]
        if len(block_dates) == 2:
            date_range_str = f"{block_dates[0].strftime('%Y-%m-%d')} to {block_dates[1].strftime('%Y-%m-%d')}"
        else:
            date_range_str = f"{block_dates[0].strftime('%Y-%m-%d')}"
        print("\n================================================")
        print(f"Đang xử lý block {idx}: {date_range_str}")
        
        data = get_block_data_from_block(block, price_data, solar_data, max_data)
        T, N = data["T"], data["N"]
        print(f" - Số giờ trong block (T): {T}")
        print(f" - Số phiên EV (N): {N}")

        if N == 0:
            f_opt.write(json.dumps({"date_range": date_range_str, "objective_value": None, "status": "No session"}) + "\n"); f_opt.flush()
            f_pv.write(json.dumps({"date_range": date_range_str, "pvfirst_cost": None, "status": "No session"}) + "\n"); f_pv.flush()
            f_drl.write(json.dumps({"date_range": date_range_str, "drl_cost": None, "deficit_kWh": None, "status": "No session"}) + "\n"); f_drl.flush()
            continue
        
        problem, Y, S_plus, R_used = build_model(data)
        status_opt, obj_val_opt = solve_model(problem)
        print(f" - LP Solver: {status_opt} với objective value = {obj_val_opt}")
        f_opt.write(json.dumps({"date_range": date_range_str, "objective_value": obj_val_opt, "status": status_opt}) + "\n"); f_opt.flush()

        pv_cost = do_greedy_pv_first_block(data)
        print(f" - PV-first cost = {pv_cost}")
        f_pv.write(json.dumps({"date_range": date_range_str, "pvfirst_cost": pv_cost, "status": "Done"}) + "\n"); f_pv.flush()

        drl_cost, drl_deficit, is_fb = do_drl_dqn_block(data, DRL_CFG)
        status_note = "Done (DRL)" if not is_fb else "Fallback PV-first (no torch)"
        print(f" - DRL (DQN) cost = {drl_cost} | deficit_kWh = {drl_deficit} | {status_note}")
        f_drl.write(json.dumps({
            "date_range": date_range_str,
            "drl_cost": drl_cost,
            "deficit_kWh": drl_deficit,
            "status": status_note
        }) + "\n"); f_drl.flush()

        month_str = block_dates[0].strftime("%Y-%m")
        if obj_val_opt is not None:
            monthly_results_opt[month_str] = monthly_results_opt.get(month_str, 0) + obj_val_opt
        if pv_cost is not None:
            monthly_results_pv[month_str] = monthly_results_pv.get(month_str, 0) + pv_cost
        if drl_cost is not None:
            monthly_results_drl[month_str] = monthly_results_drl.get(month_str, 0) + drl_cost

    f_opt.close(); f_pv.close(); f_drl.close()

    print("\n======== TỔNG HỢP THEO THÁNG (Optimized) ========")
    for month, cost in sorted(monthly_results_opt.items()):
        print(f" - {month}: {cost}")

    print("\n======== TỔNG HỢP THEO THÁNG (PV-first) ========")
    for month, cost in sorted(monthly_results_pv.items()):
        print(f" - {month}: {cost}")

    print("\n======== TỔNG HỢP THEO THÁNG (DRL - DQN) ========")
    for month, cost in sorted(monthly_results_drl.items()):
        print(f" - {month}: {cost}")

if __name__ == "__main__":
    main()

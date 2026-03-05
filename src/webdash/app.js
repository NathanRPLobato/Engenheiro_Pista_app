const API = "http://127.0.0.1:8000";

function el(id){ return document.getElementById(id); }

function setStatus(text, kind){
  const s = el("status");
  s.textContent = text;
  s.className = "status " + (kind || "");
}

function parseCompounds(s){
  return String(s)
    .split(",")
    .map(x => x.trim())
    .filter(x => x.length)
    .map(x => parseInt(x,10))
    .filter(n => Number.isFinite(n));
}

function normalizeCompoundsInput(list){
  return [...new Set(list)].sort((a,b)=>a-b);
}

function compToLabel(comp){
  const m = {
    16: ["SOFT","soft"],
    17: ["MED","medium"],
    18: ["HARD","hard"],
    7:  ["INTER","inter"],
    22: ["WET","wet"],
  };
  return m[comp] || [`C${comp}`, "hard"];
}

function fmtSec(s){
  if(!Number.isFinite(s)) return "—";
  const mm = Math.floor(s/60);
  const ss = s - mm*60;
  return `${mm}:${ss.toFixed(3).padStart(6,"0")}`;
}
function fmtMin(min){
  if(!Number.isFinite(min)) return "—";
  return `${min.toFixed(2)} min`;
}

function renderKpis(data){
  const totalS = data?.prediction?.total_time_s;
  const totalMin = data?.prediction?.total_time_min;
  const pits = data?.strategy?.pit_stops ?? 0;
  const pitLoss = data?.strategy?.pit_loss_total_s ?? 0;
  const api = data?.perf?.elapsed_s;

  el("kpiTotal").textContent = fmtSec(totalS);
  el("kpiTotalSub").textContent = fmtMin(totalMin);

  el("kpiPits").textContent = String(pits);
  el("kpiPitsSub").textContent = `Perda total: ${pitLoss.toFixed(1)} s`;

  el("kpiApi").textContent = (Number.isFinite(api) ? `${api.toFixed(3)}s` : "—");
  el("kpiApiSub").textContent = `Drift: ${data?.resolved?.drift_s_per_lap ?? "—"} s/volta`;

  el("trackBadge").textContent = `${data?.resolved?.trackId ?? "—"} • ${data?.resolved?.race_laps ?? "—"} voltas`;
  el("setupBadge").textContent = `${data?.setup?.mode ?? "—"} • score ${Number(data?.setup?.score ?? 0).toFixed(2)}`;
}

function renderStrategy(data){
  const stints = data?.strategy?.stints || [];
  const totalLaps = Number(data?.resolved?.race_laps || 0);

  // bar
  const bar = el("strategyBar");
  bar.innerHTML = "";
  stints.forEach(st => {
    const [_, cls] = compToLabel(st.compound_id);
    const w = totalLaps ? (100 * st.laps / totalLaps) : 0;
    const seg = document.createElement("div");
    seg.className = `seg ${cls}`;
    seg.style.width = `${w}%`;
    bar.appendChild(seg);
  });

  // table
  const tb = el("stintsTable");
  const breakdown = data?.prediction?.stints_breakdown || [];
  const rows = stints.map((st, i) => {
    const b = breakdown[i] || {};
    const pace = b?.pace_summary || {};
    const [lbl, cls] = compToLabel(st.compound_id);
    return {
      i: i+1,
      compound_id: st.compound_id,
      lbl, cls,
      laps: st.laps,
      stint_time_s: b?.stint_time_s,
      base: pace?.baseline_mean_lap_s,
      avg_deg: pace?.avg_deg_penalty_s,
      max_deg: pace?.max_deg_penalty_s
    };
  });

  tb.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Pneu</th>
          <th>Voltas</th>
          <th>Tempo do stint</th>
          <th>Base/volta</th>
          <th>Deg avg / max</th>
        </tr>
      </thead>
      <tbody>
        ${rows.map(r => `
          <tr>
            <td>${r.i}</td>
            <td>
              <span class="pill">
                <span class="dot ${r.cls}"></span>
                ${r.lbl} <span style="opacity:.7">(${r.compound_id})</span>
              </span>
            </td>
            <td><b>${r.laps}</b></td>
            <td>${fmtSec(r.stint_time_s)}</td>
            <td>${Number.isFinite(r.base) ? r.base.toFixed(3)+" s" : "—"}</td>
            <td>${Number.isFinite(r.avg_deg) ? r.avg_deg.toFixed(3) : "—"} / ${Number.isFinite(r.max_deg) ? r.max_deg.toFixed(3) : "—"} s</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function groupSetup(setup){
  // setup_full tem 30+ variáveis. Agrupa pra ficar humano.
  const g = {
    "Aero": ["wing_setup_0","wing_setup_1"],
    "Diferencial": ["diff_onThrottle_setup","diff_offThrottle_setup"],
    "Geometria": [
      "camber_setup_0","camber_setup_1","camber_setup_2","camber_setup_3",
      "toe_setup_0","toe_setup_1","toe_setup_2","toe_setup_3"
    ],
    "Suspensão": [
      "susp_spring_setup_0","susp_spring_setup_1","susp_spring_setup_2","susp_spring_setup_3",
      "arb_setup_0","arb_setup_1",
      "susp_height_setup_0","susp_height_setup_1","susp_height_setup_2","susp_height_setup_3"
    ],
    "Freio": ["brake_press_setup","brake_bias_setup","brake_engine_setup","front_brake_bias"],
    "Pressão pneus": ["tyre_press_setup_0","tyre_press_setup_1","tyre_press_setup_2","tyre_press_setup_3"],
    "Assistências": ["traction_ctrl_setup","abs_setup"],
    "Carga": ["ballast_setup","fuel_setup"]
  };

  const out = [];
  const keys = Object.keys(g);

  // coloca qualquer chave extra num grupo 'Outros'
  const used = new Set(keys.flatMap(k => g[k]));
  const extra = Object.keys(setup || {}).filter(k => !used.has(k));

  keys.forEach(name => {
    const items = g[name]
      .filter(k => setup && setup[k] !== undefined)
      .map(k => [k, setup[k]]);
    if(items.length) out.push([name, items]);
  });

  if(extra.length){
    out.push(["Outros", extra.map(k => [k, setup[k]])]);
  }

  return out;
}

function prettyKey(k){
  return k.replaceAll("_setup","")
          .replaceAll("_"," ")
          .replace(/\b\w/g, c => c.toUpperCase());
}

function prettyVal(v){
  if(typeof v === "number"){
    // pressões são grandes, deixa sem muita casa
    if (Math.abs(v) > 10000) return v.toFixed(0);
    // setup fino
    return Math.abs(v) < 1 ? v.toFixed(3) : v.toFixed(2);
  }
  return String(v);
}

function renderSetup(data){
  const setup = data?.setup?.setup_full || {};
  const groups = groupSetup(setup);
  const box = el("setupGroups");

  box.innerHTML = groups.map(([name, items]) => `
    <div class="group">
      <div class="group-title">
        <div>${name}</div>
        <span>${items.length} params</span>
      </div>
      <div class="kv">
        ${items.map(([k,v]) => `
          <div class="k">${prettyKey(k)}</div>
          <div class="v">${prettyVal(v)}</div>
        `).join("")}
      </div>
    </div>
  `).join("");
}

function validateInputs(payload){
  if(!payload.trackId) return "Selecione a pista.";
  if(!Number.isFinite(payload.race_laps) || payload.race_laps < 1) return "Voltas inválidas.";
  if(!payload.allowed_compounds.length) return "Selecione pneus permitidos.";
  if(payload.min_stint_laps > payload.max_stint_laps) return "Min stint não pode ser maior que Max stint.";
  if(payload.max_stint_laps > payload.race_laps) return "Max stint não pode ser maior que voltas da corrida.";
  if(payload.min_stint_laps < 1) return "Min stint inválido.";
  return null;
}

function syncChipsFromInput(){
  const arr = new Set(parseCompounds(el("allowed_compounds").value));
  document.querySelectorAll(".chip.tyre").forEach(btn => {
    const comp = parseInt(btn.dataset.comp,10);
    btn.classList.toggle("active", arr.has(comp));
  });
}

function toggleChip(comp){
  const cur = new Set(parseCompounds(el("allowed_compounds").value));
  if(cur.has(comp)) cur.delete(comp);
  else cur.add(comp);
  const out = normalizeCompoundsInput([...cur]);
  el("allowed_compounds").value = out.join(",");
  syncChipsFromInput();
}

async function runSim(){
  setStatus("Rodando simulação…", "");
  el("raw").classList.add("hidden");

  const payload = {
    sqlite_path: el("sqlite_path").value.trim(),
    trackId: el("trackId").value,
    race_laps: parseInt(el("race_laps").value, 10),

    setup_mode: el("setup_mode").value,
    style_code: "STYLE_BALANCED",

    allowed_compounds: normalizeCompoundsInput(parseCompounds(el("allowed_compounds").value)),
    two_compounds_rule: el("two_compounds_rule").checked,

    pit_loss_s: parseFloat(el("pit_loss_s").value),
    max_stints: 25,
    max_stint_laps: parseInt(el("max_stint_laps").value, 10),
    min_stint_laps: parseInt(el("min_stint_laps").value, 10),

    min_compound_total_laps: parseInt(el("min_stint_laps").value, 10), // por enquanto igual ao min stint
    drift_s_per_lap: parseFloat(el("drift_s_per_lap").value),
  };

  const err = validateInputs(payload);
  if(err){
    setStatus(err, "bad");
    return;
  }

  try{
    const r = await fetch(`${API}/simulate`, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(payload),
    });

    if(!r.ok){
      const txt = await r.text();
      setStatus(`API erro ${r.status}`, "bad");
      el("raw").textContent = txt;
      el("raw").classList.remove("hidden");
      return;
    }

    const data = await r.json();

    renderKpis(data);
    renderStrategy(data);
    renderSetup(data);

    setStatus("OK. Resultado atualizado.", "ok");

    // JSON raw fica disponível no botão
    el("raw").textContent = JSON.stringify(data, null, 2);
  }catch(e){
    setStatus("Falha de conexão com a API.", "bad");
    el("raw").textContent = String(e);
    el("raw").classList.remove("hidden");
  }
}

window.addEventListener("DOMContentLoaded", () => {
  // chips
  document.querySelectorAll(".chip.tyre").forEach(btn => {
    btn.addEventListener("click", () => toggleChip(parseInt(btn.dataset.comp,10)));
  });

  el("allowed_compounds").addEventListener("input", syncChipsFromInput);
  syncChipsFromInput();

  el("btnRun").addEventListener("click", runSim);

  // toggle raw json
  el("btnRaw").addEventListener("click", () => {
    el("raw").classList.toggle("hidden");
  });
});
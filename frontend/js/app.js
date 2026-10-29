import { initChart, pushDelays, clearChart } from './chart.js';

// Same-origin: Flask serves this page and the API from one server.
const API = '/api';
const TICK_MS = 400;
const QUEUE_CAPACITY = 40;
const LINK_SCALE = 35; // packets/slot that fills a link meter
const LOG_LIMIT = 200;
const SLA_TARGETS = [2, 6, 20];
const QUEUE_NAMES = ['URLLC', 'eMBB', 'Best effort'];
const LINK_NAMES = ['5G sub-6 GHz', '5G mmWave', 'Wi-Fi 6E'];

const $ = sel => document.querySelector(sel);
const buffers = [...document.querySelectorAll('.buffer')];
const links = [...document.querySelectorAll('.link')];
const wires = [...document.querySelectorAll('.wires path')];
const rates = [0, 1, 2].map(i => $(`#rate${i}`));

const sim = { running: false, slot: 0, blocked: false };

// ---------- API ----------

async function post(path, body) {
    const res = await fetch(API + path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body ?? {}),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `Server returned ${res.status}`);
    return data;
}

function setStatus(text, state = 'idle') {
    const el = $('#status');
    el.textContent = text;
    el.dataset.state = state;
}

function showOffline(err) {
    console.error(err);
    setStatus(`Can't reach the simulator. Start it with "python backend/app.py", then open localhost:5000.`, 'error');
}

// ---------- Rendering ----------

function buildCells() {
    for (const buffer of buffers) {
        const cells = buffer.querySelector('.cells');
        cells.replaceChildren(...Array.from({ length: QUEUE_CAPACITY }, () => document.createElement('span')));
    }
}

function renderBuffers(packetAges) {
    buffers.forEach((buffer, k) => {
        const ages = packetAges[k];
        const hol = ages.length ? ages[0] : 0;
        buffer.querySelectorAll('.cells span').forEach((cell, i) => {
            cell.className = i >= ages.length ? '' : ages[i] > SLA_TARGETS[k] ? 'on stale' : 'on';
        });
        buffer.querySelector('[data-field="len"]').textContent = ages.length;
        buffer.querySelector('[data-field="hol"]').textContent = hol;
        buffer.classList.toggle('is-late', hol > SLA_TARGETS[k]);
    });
}

function renderLinks(capacities, activeLink) {
    links.forEach((link, i) => {
        link.querySelector('[data-field="cap"]').textContent = capacities[i].toFixed(1);
        link.querySelector('.meter-fill').style.width = `${Math.min(100, (capacities[i] / LINK_SCALE) * 100)}%`;
        link.classList.toggle('is-active', i === activeLink);
    });
    links[1].classList.toggle('is-blocked', sim.blocked);
}

function renderRoute(queueIdx, linkIdx) {
    wires.forEach(w => {
        const id = w.dataset.wire;
        w.classList.toggle('on', id === `q${queueIdx}` || id === `l${linkIdx}`);
    });
}

function renderDecision(data) {
    const { queue_idx: q, link_idx: l } = data.action_taken;
    const served = data.info.served_count;
    const el = $('#decision');
    el.replaceChildren(
        `Slot ${sim.slot}: sent `,
        bold(`${served} ${QUEUE_NAMES[q]}`),
        ` packet${served === 1 ? '' : 's'} over `,
        bold(LINK_NAMES[l]),
        `. Reward ${data.reward.toFixed(1)}.`,
    );
    if (data.done) el.append(' Episode finished, a new one starts next slot.');
}

function bold(text) {
    const b = document.createElement('strong');
    b.textContent = text;
    return b;
}

function appendLog(data) {
    const { queue_idx: q, link_idx: l } = data.action_taken;
    const li = document.createElement('li');
    const slot = document.createElement('span');
    slot.className = 'slot';
    slot.textContent = sim.slot;
    const what = document.createElement('span');
    what.textContent = `${QUEUE_NAMES[q]} via ${LINK_NAMES[l]}, ${data.info.served_count} sent`;
    const reward = document.createElement('span');
    reward.textContent = data.reward.toFixed(1);
    if (data.reward < 0) reward.className = 'neg';
    li.append(slot, what, reward);

    const log = $('#log');
    log.prepend(li);
    while (log.children.length > LOG_LIMIT) log.lastElementChild.remove();
}

function render(data) {
    sim.blocked = data.mmwave_blocked;
    renderBuffers(data.packet_ages);
    renderLinks(data.link_capacities, data.action_taken.link_idx);
    renderRoute(data.action_taken.queue_idx, data.action_taken.link_idx);
    renderDecision(data);
    appendLog(data);
    pushDelays(sim.slot, data.hol_delays);
}

function renderEmpty() {
    renderBuffers([[], [], []]);
    renderLinks([10, 25, 15], -1);
    renderRoute(-1, -1);
    $('#decision').textContent = 'Start the simulation to see the first decision.';
    $('#log').replaceChildren();
    clearChart();
}

// ---------- Simulation loop ----------

async function step() {
    const data = await post('/step', {
        policy: $('#policy').value,
        arrival_rates: rates.map(r => parseFloat(r.value)),
    });
    sim.slot += 1;
    render(data);
}

// Waits for each step to finish before scheduling the next, so slow responses
// never pile up or arrive out of order.
async function runLoop() {
    while (sim.running) {
        const started = performance.now();
        try {
            await step();
            setStatus(`Running, slot ${sim.slot}`, 'running');
        } catch (err) {
            stop();
            showOffline(err);
            return;
        }
        const wait = TICK_MS - (performance.now() - started);
        if (wait > 0) await new Promise(r => setTimeout(r, wait));
    }
}

function start() {
    if (sim.running) return;
    sim.running = true;
    $('#btnRun').textContent = 'Pause';
    $('#btnStep').disabled = true;
    runLoop();
}

function stop() {
    sim.running = false;
    $('#btnRun').textContent = 'Start';
    $('#btnStep').disabled = false;
    setStatus(`Paused at slot ${sim.slot}`);
}

// ---------- Wiring ----------

function bindControls() {
    rates.forEach(input => {
        const out = input.parentElement.querySelector('output');
        input.addEventListener('input', () => { out.textContent = parseFloat(input.value).toFixed(1); });
    });

    $('#btnRun').addEventListener('click', () => (sim.running ? stop() : start()));

    $('#btnStep').addEventListener('click', async () => {
        try {
            await step();
            setStatus(`Paused at slot ${sim.slot}`);
        } catch (err) {
            showOffline(err);
        }
    });

    $('#btnBlock').addEventListener('click', async () => {
        const btn = $('#btnBlock');
        try {
            const data = await post('/toggle-blockage', { blocked: !sim.blocked });
            sim.blocked = data.mmwave_blocked;
            btn.setAttribute('aria-pressed', String(sim.blocked));
            btn.textContent = sim.blocked ? 'Unblock mmWave' : 'Block mmWave';
            links[1].classList.toggle('is-blocked', sim.blocked);
        } catch (err) {
            showOffline(err);
        }
    });

    $('#btnReset').addEventListener('click', async () => {
        stop();
        try {
            await post('/reset');
        } catch (err) {
            showOffline(err);
            return;
        }
        sim.slot = 0;
        sim.blocked = false;
        $('#btnBlock').setAttribute('aria-pressed', 'false');
        $('#btnBlock').textContent = 'Block mmWave';
        renderEmpty();
        setStatus('Paused at slot 0');
    });

    $('#btnBench').addEventListener('click', runBenchmark);
}

async function runBenchmark() {
    const btn = $('#btnBench');
    const note = $('#benchNote');
    btn.disabled = true;
    btn.textContent = 'Running benchmark…';
    try {
        const res = await fetch(`${API}/benchmark`);
        if (!res.ok) throw new Error(`Server returned ${res.status}`);
        const { results, results_blocked, model_loaded, seed } = await res.json();
        $('#benchTables').replaceChildren(
            benchmarkTable('All links working', results),
            benchmarkTable('mmWave blocked', results_blocked),
        );
        note.textContent = `Traffic seed ${seed}. Every scheduler got this same traffic; running again draws a new seed.`;
        if (!model_loaded) {
            note.textContent += ' No trained checkpoint was found, so the DQN row is an untrained network. Run python backend/train.py first.';
        }
    } catch (err) {
        note.textContent = `Benchmark failed: ${err.message}. Check that the simulator is running.`;
    } finally {
        btn.disabled = false;
        btn.textContent = 'Run benchmark again';
    }
}

function benchmarkTable(title, results) {
    // Higher is better for reward and served; lower is better for the rest.
    const columns = [
        ['reward', 'Reward', Math.max, v => v.toFixed(1)],
        ['served', 'Served', Math.max, String],
        ['drops', 'Dropped', Math.min, String],
        ['sla_violations', 'SLA misses', Math.min, String],
    ];
    const rows = Object.entries(results);
    const best = Object.fromEntries(columns.map(([key, , pick]) => [key, pick(...rows.map(([, r]) => r[key]))]));

    const table = document.createElement('table');
    const caption = table.createCaption();
    caption.textContent = title;
    const headRow = table.createTHead().insertRow();
    for (const label of ['Scheduler', ...columns.map(c => c[1])]) {
        const th = document.createElement('th');
        th.scope = 'col';
        th.textContent = label;
        headRow.append(th);
    }

    const tbody = table.createTBody();
    tbody.append(...rows.map(([name, r]) => {
        const tr = document.createElement('tr');
        const nameCell = document.createElement('td');
        nameCell.textContent = $(`#policy option[value="${name}"]`)?.textContent ?? name;
        tr.append(nameCell);
        for (const [key, , , fmt] of columns) {
            const td = document.createElement('td');
            td.textContent = fmt(r[key]);
            if (r[key] === best[key]) td.className = 'best';
            tr.append(td);
        }
        return tr;
    }));
    return table;
}

buildCells();
initChart($('#delayChart'), SLA_TARGETS);
bindControls();
// Start from a clean episode so the page and the server agree after a reload.
post('/reset').catch(showOffline);

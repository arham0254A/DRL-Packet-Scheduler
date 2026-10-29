const WINDOW = 40;
const css = getComputedStyle(document.documentElement);
const color = name => css.getPropertyValue(name).trim();

const CLASSES = [
    { label: 'URLLC', color: color('--c0') },
    { label: 'eMBB', color: color('--c1') },
    { label: 'Best effort', color: color('--c2') },
];

let chart;

export function initChart(canvas, slaTargets) {
    const ink = color('--ink-soft');
    const grid = color('--empty');

    const delayLines = CLASSES.map(c => ({
        label: c.label,
        data: [],
        borderColor: c.color,
        backgroundColor: c.color,
        borderWidth: 2,
        pointRadius: 0,
        tension: 0,
    }));
    const slaLines = CLASSES.map((c, i) => ({
        label: `${c.label} SLA`,
        data: [],
        sla: slaTargets[i],
        borderColor: c.color,
        borderWidth: 1,
        borderDash: [4, 4],
        pointRadius: 0,
    }));

    chart = new Chart(canvas, {
        type: 'line',
        data: { labels: [], datasets: [...delayLines, ...slaLines] },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                x: {
                    title: { display: true, text: 'Slot', color: ink },
                    ticks: { color: ink, maxTicksLimit: 8 },
                    grid: { color: grid },
                },
                y: {
                    title: { display: true, text: 'Slots waited', color: ink },
                    ticks: { color: ink, precision: 0 },
                    grid: { color: grid },
                    beginAtZero: true,
                    suggestedMax: 24,
                },
            },
            plugins: {
                legend: {
                    labels: {
                        color: color('--ink'),
                        boxWidth: 14,
                        boxHeight: 3,
                        filter: item => !item.text.endsWith('SLA'),
                    },
                },
                tooltip: { filter: item => !item.dataset.label.endsWith('SLA') },
            },
        },
    });
}

export function pushDelays(slot, holDelays) {
    const { labels, datasets } = chart.data;
    labels.push(slot);
    datasets.forEach((ds, i) => ds.data.push(i < 3 ? holDelays[i] : ds.sla));

    if (labels.length > WINDOW) {
        labels.shift();
        datasets.forEach(ds => ds.data.shift());
    }
    chart.update();
}

export function clearChart() {
    chart.data.labels = [];
    chart.data.datasets.forEach(ds => { ds.data = []; });
    chart.update();
}

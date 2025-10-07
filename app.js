// app.js
// Wiring, UI, file parsing, training loops for shallow/deep/comparison, visualization.

const EMBEDDING_DIM = 32;
const BATCH_SIZE = 1024;
const EPOCHS = 10;
const LEARNING_RATE = 0.01;
const LOG_UPDATE_INTERVAL = 5;
const VISUALIZATION_SAMPLES = 1000;

class App {
  constructor() {
    // data & model
    this.interactions = [];
    this.items = [];
    this.numUsers = 0;
    this.numItems = 0;
    this.model = null;
    this.isTraining = false;

    // UI
    this.nodes = {
      interactionsFile: document.getElementById('interactions-file'),
      itemsFile: document.getElementById('items-file'),
      loadDefaults: document.getElementById('load-defaults'),
      resetApp: document.getElementById('reset-app'),
      userSelect: document.getElementById('user-select'),
      trainShallow: document.getElementById('train-shallow'),
      trainDeep: document.getElementById('train-deep'),
      trainBoth: document.getElementById('train-both'),
      trainingLog: document.getElementById('training-log'),
      epochCounter: document.getElementById('epoch-counter'),
      lossDisplay: document.getElementById('loss-display'),
      progressBar: document.getElementById('progress-bar'),
      batchProgress: document.getElementById('batch-progress'),
      trainState: document.getElementById('train-state'),
      recoList: document.getElementById('recommendation-list'),
      currentUserId: document.getElementById('current-user-id'),
      vizStatus: document.getElementById('viz-status'),
      vizContainer: document.getElementById('visualization-container'),
      comparisonChart: document.getElementById('comparison-chart'),
      epochsCount: document.getElementById('epochs-count'),
    };

    // bind
    this.nodes.interactionsFile.addEventListener('change', (e) => this.onFileSelected(e, 'interactions'));
    this.nodes.itemsFile.addEventListener('change', (e) => this.onFileSelected(e, 'items'));
    this.nodes.loadDefaults.addEventListener('click', () => this.loadDefaults());
    this.nodes.resetApp.addEventListener('click', () => this.resetApp());
    this.nodes.trainShallow.addEventListener('click', () => this.trainMode('shallow'));
    this.nodes.trainDeep.addEventListener('click', () => this.trainMode('deep'));
    this.nodes.trainBoth.addEventListener('click', () => this.trainBoth());

    this.init();
  }

  async init() {
    this.log('App initialized. Upload CSVs or click "Load defaults" to use u.data / u.item beside index.html.');
    this.updateButtonsState(false);
    this.nodes.epochsCount.textContent = EPOCHS.toString();
  }

  updateButtonsState(enabled) {
    this.nodes.trainShallow.disabled = !enabled;
    this.nodes.trainDeep.disabled = !enabled;
    this.nodes.trainBoth.disabled = !enabled;
    this.nodes.userSelect.disabled = !enabled;
  }

  log(text) {
    const now = new Date().toLocaleTimeString();
    const p = document.createElement('div');
    p.textContent = `[${now}] ${text}`;
    this.nodes.trainingLog.prepend(p);
  }

  // --- File handling ---
  async onFileSelected(event, type) {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    const text = await file.text();
    if (type === 'interactions') {
      this.interactions = this.parseInteractionsCSV(text);
      this.log(`Loaded interactions from uploaded file: ${this.interactions.length} rows.`);
    } else {
      this.items = this.parseItemsCSV(text);
      this.log(`Loaded items from uploaded file: ${this.items.length} rows.`);
    }
    this.afterDataLoad();
  }

  async loadDefaults() {
    // tries multiple common filenames
    const candidates = ['u.data', 'u.data.csv', 'u_data.csv', 'u.item', 'u.item.csv', 'u_item.csv'];
    try {
      // fetch interactions
      let interactionsText = null;
      for (const n of ['u.data', 'u.data.csv', 'u_data.csv']) {
        try {
          const r = await fetch(n);
          if (r.ok) { interactionsText = await r.text(); break; }
        } catch(e) {}
      }
      // fetch items
      let itemsText = null;
      for (const n of ['u.item', 'u.item.csv', 'u_item.csv']) {
        try {
          const r = await fetch(n);
          if (r.ok) { itemsText = await r.text(); break; }
        } catch(e) {}
      }
      if (!interactionsText || !itemsText) throw new Error('Could not fetch both u.data and u.item from server root.');
      this.interactions = this.parseInteractionsCSV(interactionsText);
      this.items = this.parseItemsCSV(itemsText);
      this.log(`Loaded defaults. Interactions: ${this.interactions.length}, Items: ${this.items.length}`);
      this.afterDataLoad();
    } catch (err) {
      console.error(err);
      this.log(`Error loading defaults: ${err.message}`);
      alert('Could not fetch default files. Upload via the file inputs or place u.data and u.item next to index.html.');
    }
  }

  resetApp() {
    this.interactions = [];
    this.items = [];
    this.numUsers = 0;
    this.numItems = 0;
    this.nodes.userSelect.innerHTML = '';
    this.nodes.recoList.innerHTML = '';
    this.nodes.vizContainer.innerHTML = '';
    this.nodes.trainingLog.innerHTML = '';
    this.updateButtonsState(false);
    if (this.model) { this.model.dispose(); this.model = null; }
    this.log('App reset.');
  }

  parseInteractionsCSV(text) {
    // detect delimiter
    const lines = text.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
    const delim = this.detectDelimiter(lines[0]);
    const rows = [];
    for (const line of lines) {
      const parts = line.split(delim).map(s => s.trim());
      // skip header rows that contain non-numeric first column
      if (!/^\d+/.test(parts[0])) continue;
      const u = parseInt(parts[0], 10) - 1;
      const it = parseInt(parts[1], 10) - 1;
      const r = parts.length > 2 ? parseInt(parts[2], 10) : 1;
      if (Number.isFinite(u) && Number.isFinite(it)) {
        rows.push({ userId: u, itemId: it, rating: r });
      }
    }
    return rows;
  }

  parseItemsCSV(text) {
    const lines = text.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
    const delim = this.detectDelimiter(lines[0]);
    const rows = [];
    for (const line of lines) {
      const parts = line.split(delim);
      if (!parts[0]) continue;
      if (!/^\d+/.test(parts[0])) continue;
      const id = parseInt(parts[0], 10) - 1;
      const title = (parts[1] || `Item ${id + 1}`).trim();
      rows.push({ id, title, raw: parts.slice(2) });
    }
    return rows;
  }

  detectDelimiter(sampleLine) {
    const counts = {
      '\t': (sampleLine.match(/\t/g) || []).length,
      ',': (sampleLine.match(/,/g) || []).length,
      '|': (sampleLine.match(/\|/g) || []).length,
      ';': (sampleLine.match(/;/g) || []).length
    };
    // pick delimiter with max occurrences
    let max = 0, delim = '\t';
    for (const k of Object.keys(counts)) {
      if (counts[k] > max) { max = counts[k]; delim = k; }
    }
    return delim;
  }

  afterDataLoad() {
    // determine sizes
    if (this.interactions.length === 0 || this.items.length === 0) {
      this.log('Need both interactions and items to proceed.');
      return;
    }
    const maxUser = Math.max(...this.interactions.map(i => i.userId));
    const maxItem = Math.max(...this.interactions.map(i => i.itemId));
    this.numUsers = maxUser + 1;
    this.numItems = maxItem + 1;

    // populate user select with sample
    this.nodes.userSelect.innerHTML = '';
    const sample = this.shuffleArray(Array.from({ length: this.numUsers }, (_, i) => i)).slice(0, 80);
    for (const uid of sample) {
      const opt = document.createElement('option');
      opt.value = uid;
      opt.textContent = `User ${uid + 1}`;
      this.nodes.userSelect.appendChild(opt);
    }
    this.nodes.userSelect.value = sample[0];
    this.nodes.currentUserId.textContent = (parseInt(sample[0], 10) + 1).toString();
    this.nodes.userSelect.addEventListener('change', async (e) => {
      this.nodes.currentUserId.textContent = (parseInt(e.target.value, 10) + 1).toString();
      if (!this.isTraining) await this.updateRecommendations(e.target.value);
    });

    this.updateButtonsState(true);
    this.log(`Prepared UI. Users: ${this.numUsers}, Items: ${this.numItems}`);
  }

  shuffleArray(arr) {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }

  // --- Training orchestrators ---

  async trainMode(mode = 'shallow') {
    if (this.isTraining) return;
    if (!this.interactions.length || !this.items.length) {
      alert('Please load data first.');
      return;
    }

    // dispose previous model if present
    if (this.model) { this.model.dispose(); this.model = null; }

    this.model = new TwoTowerModel(this.numUsers, this.numItems, EMBEDDING_DIM, LEARNING_RATE, mode, { towerHidden: [64], towerOutputDim: EMBEDDING_DIM });

    this.isTraining = true;
    this.nodes.trainState.textContent = `Training (${mode})`;
    this.nodes.trainState.style.color = '#d97706';

    const interactions = this.shuffleArray(this.interactions.slice());
    const stepsPerEpoch = Math.ceil(interactions.length / BATCH_SIZE);

    const epochLosses = [];
    for (let epoch = 1; epoch <= EPOCHS; epoch++) {
      this.nodes.epochCounter.textContent = epoch;
      let epochLoss = 0;
      let step = 0;

      for (let i = 0; i < interactions.length; i += BATCH_SIZE) {
        const batch = interactions.slice(i, i + BATCH_SIZE);
        const userIdx = batch.map(b => b.userId);
        const itemIdx = batch.map(b => b.itemId);

        const loss = await this.model.trainStep(userIdx, itemIdx);
        epochLoss += loss;
        step++;

        // update UI
        if (step % LOG_UPDATE_INTERVAL === 0 || i + BATCH_SIZE >= interactions.length) {
          const avg = epochLoss / step;
          this.nodes.lossDisplay.textContent = avg.toFixed(4);
          this.log(`Mode=${mode} Epoch ${epoch}/${EPOCHS} Step ${step}/${stepsPerEpoch} AvgLoss=${avg.toFixed(4)}`);
        }

        // progress bar (per model)
        const modelProgress = Math.round(((epoch - 1) / EPOCHS) * 100 + (i / interactions.length) * (100 / EPOCHS));
        this.nodes.progressBar.style.width = `${modelProgress}%`;
        this.nodes.batchProgress.textContent = `${Math.min(100, Math.round((i + BATCH_SIZE) / interactions.length * 100))}%`;

        // allow UI update
        await new Promise(r => setTimeout(r, 0));
      }

      const avgEpochLoss = epochLoss / step;
      epochLosses.push(avgEpochLoss);
      this.log(`Mode=${mode} Epoch ${epoch} complete. AvgLoss=${avgEpochLoss.toFixed(4)}`);

      // update recommendations and viz after each epoch
      await this.updateRecommendations(this.nodes.userSelect.value);
      await this.updateVisualization();
    }

    this.isTraining = false;
    this.nodes.trainState.textContent = 'Idle';
    this.nodes.trainState.style.color = '#111827';
    this.nodes.progressBar.style.width = '100%';
    this.nodes.batchProgress.textContent = '100%';

    // return epoch losses for plotting comparisons
    return epochLosses;
  }

  async trainBoth() {
    if (this.isTraining) return;
    this.log('Starting comparison: Shallow then Deep (same data).');

    // train shallow
    const shallowLoss = await this.trainMode('shallow');
    // dispose model & memory is freed in trainMode
    if (this.model) { this.model.dispose(); this.model = null; }

    // train deep
    const deepLoss = await this.trainMode('deep');

    // draw comparison
    this.drawComparison(shallowLoss, deepLoss);
    this.log('Comparison complete.');
  }

  // --- Recommendations & visualization ---

  async updateRecommendations(userId) {
    if (!this.model) return;
    const uid = parseInt(userId, 10);

    // get user vector (one)
    const userTensor = tf.tidy(() => {
      return this.model.userForward(tf.tensor1d([uid], 'int32')).squeeze(); // [D']
    });

    const scores = await this.model.scoreAllItems(userTensor);
    userTensor.dispose();

    // filter items user has rated (any rating)
    const rated = new Set(this.interactions.filter(it => it.userId === uid).map(it => it.itemId));

    const ranked = Array.from(scores).map((s, idx) => ({
      itemId: idx,
      score: s,
      title: (this.items[idx] && this.items[idx].title) || `Item ${idx + 1}`
    })).filter(x => !rated.has(x.itemId)).sort((a, b) => b.score - a.score).slice(0, 15);

    this.nodes.recoList.innerHTML = '';
    if (!ranked.length) {
      this.nodes.recoList.innerHTML = '<li>No recommendations (dataset small or user rated all items).</li>';
    } else {
      for (let i = 0; i < ranked.length; i++) {
        const li = document.createElement('li');
        li.textContent = `${i + 1}. ${ranked[i].title} — score: ${ranked[i].score.toFixed(3)}`;
        this.nodes.recoList.appendChild(li);
      }
    }
  }

  async updateVisualization() {
    if (!this.model) return;
    this.nodes.vizStatus.textContent = 'Computing PCA...';
    try {
      const { projection, sampleIndices } = await this.model.computePCAProjection(VISUALIZATION_SAMPLES);
      const data = projection.map((coords, i) => ({
        x: coords[0],
        y: coords[1],
        id: sampleIndices[i],
        title: (this.items[sampleIndices[i]] && this.items[sampleIndices[i]].title) || `Item ${sampleIndices[i] + 1}`
      }));
      this.renderVisualization(data);
      this.nodes.vizStatus.textContent = `Showing PCA projection for ${data.length} items (mode=${this.model.mode}).`;
    } catch (err) {
      console.error(err);
      this.nodes.vizStatus.textContent = 'Visualization error';
      this.log(`PCA error: ${err.message}`);
    }
  }

  renderVisualization(data) {
    this.nodes.vizContainer.innerHTML = '';
    const margin = { top: 20, right: 20, bottom: 30, left: 40 };
    const w = this.nodes.vizContainer.clientWidth - margin.left - margin.right;
    const h = this.nodes.vizContainer.clientHeight - margin.top - margin.bottom;
    const svg = d3.select('#visualization-container').append('svg')
      .attr('width', w + margin.left + margin.right)
      .attr('height', h + margin.top + margin.bottom)
      .append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const xExtent = d3.extent(data, d => d.x);
    const yExtent = d3.extent(data, d => d.y);
    const xScale = d3.scaleLinear().domain(xExtent).nice().range([0, w]);
    const yScale = d3.scaleLinear().domain(yExtent).nice().range([h, 0]);

    svg.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xScale).ticks(6));
    svg.append('g').call(d3.axisLeft(yScale).ticks(6));

    const tooltip = d3.select('body').append('div')
      .style('position', 'absolute').style('background', '#0b1220').style('color', '#cfe9ff')
      .style('padding', '6px 8px').style('border-radius', '6px').style('pointer-events', 'none').style('opacity', 0);

    svg.selectAll('.dot').data(data).enter().append('circle')
      .attr('cx', d => xScale(d.x)).attr('cy', d => yScale(d.y)).attr('r', 3.2).attr('fill', '#2c7be5')
      .on('mouseover', (event, d) => {
        tooltip.transition().duration(120).style('opacity', 1);
        tooltip.html(`<strong>${d.title}</strong><br/>id: ${d.id}`)
          .style('left', (event.pageX + 10) + 'px').style('top', (event.pageY - 28) + 'px');
      }).on('mousemove', (event) => {
        tooltip.style('left', (event.pageX + 10) + 'px').style('top', (event.pageY - 28) + 'px');
      }).on('mouseout', () => {
        tooltip.transition().duration(200).style('opacity', 0);
      });
  }

  // --- Comparison chart ---
  drawComparison(shallowLoss, deepLoss) {
    // draw per-epoch line chart of losses
    this.nodes.comparisonChart.innerHTML = '';
    const margin = { top: 10, right: 10, bottom: 30, left: 40 };
    const width = this.nodes.comparisonChart.clientWidth - margin.left - margin.right;
    const height = this.nodes.comparisonChart.clientHeight - margin.top - margin.bottom;

    const svg = d3.select('#comparison-chart').append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const maxEpochs = Math.max(shallowLoss.length, deepLoss.length);
    const allLoss = shallowLoss.concat(deepLoss);
    const x = d3.scaleLinear().domain([1, maxEpochs]).range([0, width]);
    const y = d3.scaleLinear().domain([0, d3.max(allLoss) * 1.05]).range([height, 0]);

    svg.append('g').attr('transform', `translate(0,${height})`).call(d3.axisBottom(x).ticks(maxEpochs));
    svg.append('g').call(d3.axisLeft(y).ticks(4));

    const line = d3.line().x((d, i) => x(i + 1)).y(d => y(d));

    svg.append('path').datum(shallowLoss).attr('fill', 'none').attr('stroke', '#2c7be5').attr('stroke-width', 2).attr('d', line);
    svg.append('path').datum(deepLoss).attr('fill', 'none').attr('stroke', '#e6554d').attr('stroke-width', 2).attr('d', line);

    // legend
    svg.append('text').attr('x', 10).attr('y', 12).attr('fill', '#2c7be5').style('font-size', 12).text('Shallow');
    svg.append('text').attr('x', 80).attr('y', 12).attr('fill', '#e6554d').style('font-size', 12).text('Deep');
  }
}

// initialize
window.onload = () => {
  window.app = new App();
};

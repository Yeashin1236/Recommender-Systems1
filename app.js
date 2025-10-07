// app.js
// Main application logic: improved structure, better progress updates, safer disposals.

const EMBEDDING_DIM = 25;
const BATCH_SIZE = 1024;
const EPOCHS = 10;
const LEARNING_RATE = 0.01;
const LOG_UPDATE_INTERVAL = 5; // update UI every N batches
const VISUALIZATION_SAMPLES = 1000;

class App {
  constructor() {
    this.model = null;
    this.interactions = [];
    this.items = [];
    this.numUsers = 0;
    this.numItems = 0;
    this.isTraining = false;

    // UI elements
    this.statusEl = document.getElementById('status');
    this.trainButton = document.getElementById('train-button');
    this.userSelect = document.getElementById('user-select');
    this.logEl = document.getElementById('training-log');
    this.epochCounter = document.getElementById('epoch-counter');
    this.lossDisplay = document.getElementById('loss-display');
    this.recoList = document.getElementById('recommendation-list');
    this.currentUserIdEl = document.getElementById('current-user-id');
    this.vizStatus = document.getElementById('viz-status');
    this.progressBar = document.getElementById('progress-bar');
    this.batchProgress = document.getElementById('batch-progress');
    this.trainStateBadge = document.getElementById('train-state');

    this.visualizationContainer = document.getElementById('visualization-container');

    this.init();
  }

  async init() {
    try {
      this.updateStatus('Loading data...', 'orange');
      await this.loadData();
      this.model = new TwoTowerModel(this.numUsers, this.numItems, EMBEDDING_DIM, LEARNING_RATE);
      this.setupUI();
      this.updateStatus('Ready', 'green');
      this.log('Ready. Click Train to start.');
    } catch (e) {
      console.error(e);
      this.updateStatus('Error loading data — see console', 'red');
      this.log(`Error: ${e.message}`);
    }
  }

  updateStatus(text, color = 'black') {
    this.statusEl.textContent = text;
    this.statusEl.style.color = color;
  }

  log(msg) {
    const now = new Date().toLocaleTimeString();
    const p = document.createElement('div');
    p.textContent = `[${now}] ${msg}`;
    this.logEl.prepend(p);
  }

  async loadData() {
    const INTERACTIONS_PATH = 'u.data';
    const ITEMS_PATH = 'u.item';

    // fetch both
    const [r1, r2] = await Promise.all([fetch(INTERACTIONS_PATH), fetch(ITEMS_PATH)]);
    if (!r1.ok || !r2.ok) {
      throw new Error('Could not fetch u.data or u.item. Ensure files are in the same folder as index.html.');
    }

    const text1 = await r1.text();
    const text2 = await r2.text();

    // Parse u.data: user\titem\trating\ttimestamp
    this.interactions = text1.split('\n').filter(Boolean).map(line => {
      const p = line.trim().split('\t');
      return {
        userId: parseInt(p[0], 10) - 1,
        itemId: parseInt(p[1], 10) - 1,
        rating: parseInt(p[2], 10)
      };
    });

    // Parse u.item: id|title|...|genres...
    this.items = text2.split('\n').filter(Boolean).map(line => {
      const p = line.split('|');
      return {
        id: parseInt(p[0], 10) - 1,
        title: p[1] || `Item ${p[0]}`,
        // store raw genre vector if present
        genres: (p.length > 5) ? p.slice(5).map(x => parseInt(x, 10)) : []
      };
    });

    // compute sizes
    const maxUser = Math.max(...this.interactions.map(i => i.userId));
    const maxItem = Math.max(...this.interactions.map(i => i.itemId));
    this.numUsers = maxUser + 1;
    this.numItems = maxItem + 1;

    this.log(`Loaded ${this.interactions.length} interactions and ${this.numItems} items.`);
    this.log(`Num users: ${this.numUsers}, num items: ${this.numItems}`);
  }

  setupUI() {
    this.trainButton.disabled = false;
    this.userSelect.disabled = false;

    // populate user select with a sample of users
    const allUserIds = Array.from({ length: this.numUsers }, (_, i) => i);
    const sample = this.shuffleArray(allUserIds).slice(0, 80);

    sample.forEach(uid => {
      const opt = document.createElement('option');
      opt.value = uid;
      opt.textContent = `User ${uid + 1}`;
      this.userSelect.appendChild(opt);
    });

    this.userSelect.value = sample[0];
    this.currentUserIdEl.textContent = parseInt(sample[0], 10) + 1;
    this.userSelect.addEventListener('change', async (e) => {
      const uid = e.target.value;
      this.currentUserIdEl.textContent = parseInt(uid, 10) + 1;
      if (!this.isTraining) await this.updateRecommendations(uid);
    });

    this.trainButton.addEventListener('click', async () => {
      if (!this.isTraining) await this.startTraining();
    });
  }

  shuffleArray(arr) {
    // Fisher-Yates
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }

  async startTraining() {
    if (this.isTraining) return;
    if (!this.model) return this.log('Model not initialized');

    this.isTraining = true;
    this.trainButton.disabled = true;
    this.trainStateBadge.textContent = 'Training';
    this.trainStateBadge.style.background = '#fff4e6';
    this.trainStateBadge.style.color = '#8a5a00';

    // Shuffle interactions copy
    const interactions = this.shuffleArray(this.interactions.slice());

    try {
      const totalSteps = Math.ceil(interactions.length / BATCH_SIZE) * EPOCHS;
      let globalStep = 0;

      for (let epoch = 1; epoch <= EPOCHS; epoch++) {
        this.epochCounter.textContent = epoch;
        let epochLoss = 0;
        let stepCount = 0;

        // process batches
        for (let i = 0; i < interactions.length; i += BATCH_SIZE) {
          const batch = interactions.slice(i, i + BATCH_SIZE);
          const userIndices = batch.map(b => b.userId);
          const itemIndices = batch.map(b => b.itemId);

          // single train step
          const lossVal = await this.model.trainStep(userIndices, itemIndices);

          epochLoss += lossVal;
          stepCount++;
          globalStep++;

          // update UI periodically
          if (stepCount % LOG_UPDATE_INTERVAL === 0 || i + BATCH_SIZE >= interactions.length) {
            const avgLoss = epochLoss / stepCount;
            this.lossDisplay.textContent = avgLoss.toFixed(4);
            this.log(`Epoch ${epoch} - Step ${stepCount} - AvgLoss: ${avgLoss.toFixed(4)}`);
          }

          // update batch progress
          const batchPercent = Math.min(100, Math.round(((i + BATCH_SIZE) / interactions.length) * 100));
          this.progressBar.style.width = `${Math.round(((epoch - 1) / EPOCHS) * 100 + (batchPercent / EPOCHS))}%`;
          this.batchProgress.textContent = `${Math.min(100, batchPercent)}%`;

          // small yield to UI
          await new Promise(resolve => setTimeout(resolve, 0));
        }

        const finalEpochLoss = epochLoss / stepCount;
        this.log(`Epoch ${epoch}/${EPOCHS} completed. Avg Loss: ${finalEpochLoss.toFixed(4)}`);
        this.lossDisplay.textContent = finalEpochLoss.toFixed(4);

        // Update recommendations and visualization after each epoch
        await this.updateRecommendations(this.userSelect.value);
        await this.updateVisualization();
      }

      this.log('Training finished.');
      this.updateStatus('Training complete', 'green');
      this.progressBar.style.width = '100%';
      this.batchProgress.textContent = '100%';
    } catch (err) {
      console.error(err);
      this.log(`Training error: ${err.message}`);
      this.updateStatus('Error during training', 'red');
    } finally {
      this.isTraining = false;
      this.trainButton.disabled = false;
      this.trainStateBadge.textContent = 'Idle';
      this.trainStateBadge.style.background = '#eef7ff';
      this.trainStateBadge.style.color = '#1366d6';
    }
  }

  async updateRecommendations(userId) {
    if (!this.model) return;

    const uid = parseInt(userId, 10);
    this.currentUserIdEl.textContent = uid + 1;

    // get user embedding tensor, then score all items
    const userEmbTensor = tf.tidy(() => {
      return this.model.userForward(tf.tensor1d([uid], 'int32')).squeeze(); // [D]
    });

    const scores = await this.model.scoreAllItems(userEmbTensor);
    userEmbTensor.dispose();

    // gather items user already rated (exclude all rated items)
    const ratedSet = new Set(this.interactions.filter(it => it.userId === uid).map(it => it.itemId));

    // build ranked list
    const ranked = Array.from(scores).map((s, idx) => ({
      itemId: idx,
      score: s,
      title: (this.items[idx] && this.items[idx].title) || `Item ${idx + 1}`
    })).filter(x => !ratedSet.has(x.itemId)).sort((a, b) => b.score - a.score).slice(0, 15);

    // render
    this.recoList.innerHTML = '';
    if (ranked.length === 0) {
      this.recoList.innerHTML = '<li>No recommendations available — maybe your dataset is small.</li>';
    } else {
      ranked.forEach((r, i) => {
        const li = document.createElement('li');
        li.textContent = `${i + 1}. ${r.title} — score: ${r.score.toFixed(3)}`;
        this.recoList.appendChild(li);
      });
    }
  }

  async updateVisualization() {
    if (!this.model) return;
    this.updateStatus('Computing PCA...', 'orange');
    this.vizStatus.textContent = 'Computing 2D PCA projection...';

    try {
      const { projection, sampleIndices } = await this.model.computePCAProjection(VISUALIZATION_SAMPLES);

      // format data
      const data = projection.map((coords, i) => ({
        x: coords[0],
        y: coords[1],
        id: sampleIndices[i],
        title: (this.items[sampleIndices[i]] && this.items[sampleIndices[i]].title) || `Item ${sampleIndices[i] + 1}`
      }));

      this.renderVisualization(data);

      this.vizStatus.textContent = `Showing PCA projection for ${data.length} items.`;
      this.updateStatus(this.isTraining ? 'Training...' : 'Ready', this.isTraining ? 'orange' : 'green');
    } catch (err) {
      console.error(err);
      this.vizStatus.textContent = 'Error generating visualization.';
      this.updateStatus('Visualization error', 'red');
      this.log(`PCA error: ${err.message}`);
    }
  }

  renderVisualization(data) {
    // clear
    this.visualizationContainer.innerHTML = '';

    const margin = { top: 20, right: 20, bottom: 30, left: 40 };
    const width = this.visualizationContainer.clientWidth - margin.left - margin.right;
    const height = this.visualizationContainer.clientHeight - margin.top - margin.bottom;

    const svg = d3.select('#visualization-container').append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .attr('viewBox', `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
      .append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const xExtent = d3.extent(data, d => d.x);
    const yExtent = d3.extent(data, d => d.y);

    const xScale = d3.scaleLinear().domain(xExtent).nice().range([0, width]);
    const yScale = d3.scaleLinear().domain(yExtent).nice().range([height, 0]);

    svg.append('g').attr('transform', `translate(0,${height})`).call(d3.axisBottom(xScale).ticks(6));
    svg.append('g').call(d3.axisLeft(yScale).ticks(6));

    // tooltip
    const tooltip = d3.select('body').append('div')
      .attr('class', 'tooltip-d3')
      .style('position', 'absolute')
      .style('background', '#0b1220')
      .style('color', '#cfe9ff')
      .style('padding', '6px 8px')
      .style('border-radius', '6px')
      .style('pointer-events', 'none')
      .style('opacity', 0);

    svg.selectAll('.dot')
      .data(data)
      .enter().append('circle')
      .attr('class', 'dot')
      .attr('cx', d => xScale(d.x))
      .attr('cy', d => yScale(d.y))
      .attr('r', 3.2)
      .attr('fill', '#2c7be5')
      .attr('opacity', 0.92)
      .on('mouseover', (event, d) => {
        tooltip.transition().duration(120).style('opacity', 1);
        tooltip.html(`<strong>${d.title}</strong><br/>id: ${d.id}`)
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 28) + 'px');
      })
      .on('mousemove', (event) => {
        tooltip.style('left', (event.pageX + 10) + 'px').style('top', (event.pageY - 28) + 'px');
      })
      .on('mouseout', () => {
        tooltip.transition().duration(200).style('opacity', 0);
      });
  }
}

// start
window.onload = () => {
  window.app = new App();
};

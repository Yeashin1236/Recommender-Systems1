// app.js

const EMBEDDING_DIM = 25;
const BATCH_SIZE = 1000;
const EPOCHS = 10;
const LEARNING_RATE = 0.01;
const LOG_UPDATE_INTERVAL = 10; // Log loss every N steps
const VISUALIZATION_SAMPLES = 1000;

class App {
    constructor() {
        this.model = null;
        this.interactions = [];
        this.items = [];
        this.numUsers = 0;
        this.numItems = 0;
        this.isTraining = false;
        this.logElement = document.getElementById('training-log');
        this.statusElement = document.getElementById('status');
        this.trainButton = document.getElementById('train-button');
        this.userSelect = document.getElementById('user-select');
        this.visualizationContainer = document.getElementById('visualization-container');

        this.init();
    }

    async init() {
        this.updateStatus('Loading data...', 'blue');
        try {
            await this.loadData();
            this.model = new TwoTowerModel(this.numUsers, this.numItems, EMBEDDING_DIM, LEARNING_RATE);
            this.setupUI();
            this.updateStatus('Ready to Train', 'green');
        } catch (error) {
            console.error(error);
            // Updated error message reflects the flat file structure
            this.updateStatus(`Error: Could not load data. Ensure u.data and u.item are in the **root** directory.`, 'red');
            this.log(`Error: ${error.message}`);
        }
    }

    async loadData() {
        // --- CHANGE HERE: REMOVED 'data/' SUBDIRECTORY ---
        const INTERACTIONS_PATH = 'u.data'; 
        const ITEMS_PATH = 'u.item';
        // --------------------------------------------------

        this.log(`Fetching interactions from ${INTERACTIONS_PATH}...`);
        
        const [interactionsResponse, itemsResponse] = await Promise.all([
            fetch(INTERACTIONS_PATH),
            fetch(ITEMS_PATH)
        ]);

        if (!interactionsResponse.ok || !itemsResponse.ok) {
            throw new Error('Could not find u.data or u.item. Ensure files are in the same directory as index.html.');
        }

        // Parse u.data (User, Item, Rating, Timestamp - TAB separated)
        const interactionsText = await interactionsResponse.text();
        this.interactions = interactionsText.split('\n')
            .filter(line => line.trim().length > 0)
            .map(line => {
                const parts = line.split('\t');
                // The dataset is 1-indexed, so we subtract 1 to make it 0-indexed
                return {
                    userId: parseInt(parts[0]) - 1, 
                    itemId: parseInt(parts[1]) - 1,
                    rating: parseInt(parts[2]),
                };
            });

        // Parse u.item (Item ID, Title, ... , Genres - PIPE separated)
        const itemsText = await itemsResponse.text();
        this.items = itemsText.split('\n')
            .filter(line => line.trim().length > 0)
            .map(line => {
                const parts = line.split('|');
                // The dataset is 1-indexed, so we subtract 1 to make it 0-indexed
                return {
                    id: parseInt(parts[0]) - 1, 
                    title: parts[1],
                    genres: parts.slice(5).map(g => parseInt(g)),
                };
            });
        
        // Find maximum IDs to set the embedding table size (standard MovieLens 100k)
        const maxUserId = Math.max(...this.interactions.map(i => i.userId));
        const maxItemId = Math.max(...this.interactions.map(i => i.itemId));

        this.numUsers = maxUserId + 1; // 943 users
        this.numItems = maxItemId + 1; // 1682 items

        this.log(`Loaded ${this.interactions.length} interactions and ${this.numItems} items.`);
        this.log(`Num Users: ${this.numUsers}, Num Items: ${this.numItems}, Embedding Dim: ${EMBEDDING_DIM}`);
    }

    setupUI() {
        this.trainButton.disabled = false;
        this.userSelect.disabled = false;
        
        // Populate user dropdown (select 50 random users for demonstration)
        const allUserIds = Array.from({ length: this.numUsers }, (_, i) => i);
        const sampleUserIds = allUserIds.sort(() => 0.5 - Math.random()).slice(0, 50);

        sampleUserIds.forEach(id => {
            const option = document.createElement('option');
            option.value = id;
            option.textContent = `User ${id + 1}`;
            this.userSelect.appendChild(option);
        });

        // Set initial user info
        this.userSelect.value = sampleUserIds[0];
        document.getElementById('current-user-id').textContent = sampleUserIds[0] + 1;
        document.getElementById('viz-status').textContent = 'Ready to visualize after training.';
    }

    updateStatus(message, color = 'black') {
        this.statusElement.textContent = message;
        this.statusElement.style.color = color;
    }

    log(message) {
        const p = document.createElement('p');
        p.textContent = message;
        this.logElement.prepend(p);
    }

    async startTraining() {
        if (this.isTraining) return;

        this.isTraining = true;
        this.trainButton.disabled = true;
        this.updateStatus('Training...', 'orange');
        
        // Shuffle interactions before training
        const shuffledInteractions = this.interactions.sort(() => 0.5 - Math.random());

        for (let epoch = 1; epoch <= EPOCHS; epoch++) {
            document.getElementById('epoch-counter').textContent = epoch;
            let epochLoss = 0;
            let stepCount = 0;

            for (let i = 0; i < shuffledInteractions.length; i += BATCH_SIZE) {
                const batch = shuffledInteractions.slice(i, i + BATCH_SIZE);
                const userIndices = batch.map(b => b.userId);
                const itemIndices = batch.map(b => b.itemId);

                const lossTensor = await this.model.trainStep(userIndices, itemIndices);
                const loss = await lossTensor.data();
                const scalarLoss = loss[0];
                lossTensor.dispose();

                epochLoss += scalarLoss;
                stepCount++;

                if (stepCount % LOG_UPDATE_INTERVAL === 0) {
                    const avgLoss = epochLoss / stepCount;
                    document.getElementById('loss-display').textContent = avgLoss.toFixed(4);
                }
            }

            const finalEpochLoss = epochLoss / stepCount;
            this.log(`Epoch ${epoch}/${EPOCHS} completed. Avg Loss: ${finalEpochLoss.toFixed(4)}`);
            document.getElementById('loss-display').textContent = finalEpochLoss.toFixed(4);

            // Update recommendations and visualization after each epoch
            await this.updateRecommendations(this.userSelect.value);
            await this.updateVisualization();
        }

        this.isTraining = false;
        this.trainButton.disabled = false;
        this.updateStatus('Training Complete', 'green');
    }

    async updateRecommendations(userId) {
        const targetUserId = parseInt(userId);
        
        // 1. Get the user's current embedding
        const userEmbTensor = tf.tidy(() => {
            return this.model.userForward(tf.tensor1d([targetUserId], 'int32')).squeeze(); // [D]
        });

        // 2. Score all items
        const scores = await this.model.scoreAllItems(userEmbTensor);
        userEmbTensor.dispose();
        
        // 3. Get items already rated by the user (to filter them out)
        const userRatedItems = new Set(
            this.interactions
                .filter(i => i.userId === targetUserId && i.rating >= 4) // Only filter out highly-rated items
                .map(i => i.itemId)
        );

        // 4. Rank items
        const rankedItems = scores
            .map((score, index) => ({ 
                itemId: index, 
                score: score, 
                title: this.items[index].title 
            }))
            .filter(item => !userRatedItems.has(item.itemId)) // Filter out already rated items
            .sort((a, b) => b.score - a.score) // Sort by score descending
            .slice(0, 10); // Take top 10

        // 5. Update UI
        const listElement = document.getElementById('recommendation-list');
        listElement.innerHTML = '';
        rankedItems.forEach((item, index) => {
            const li = document.createElement('li');
            li.textContent = `${index + 1}. ${item.title} (Score: ${item.score.toFixed(3)})`;
            listElement.appendChild(li);
        });
    }

    async updateUserInfo(userId) {
        document.getElementById('current-user-id').textContent = parseInt(userId) + 1;
        if (!this.isTraining) {
            await this.updateRecommendations(userId);
        }
    }

    async updateVisualization() {
        this.updateStatus('Generating visualization...', 'orange');
        document.getElementById('viz-status').textContent = 'Computing 2D PCA projection...';
        
        const { projection, sampleIndices } = await this.model.computePCAProjection(VISUALIZATION_SAMPLES);
        
        // Format data for D3
        const data = projection.map((coords, i) => ({
            x: coords[0],
            y: coords[1],
            id: sampleIndices[i],
            title: this.items[sampleIndices[i]].title
        }));

        this.renderVisualization(data);
        document.getElementById('viz-status').textContent = `Showing PCA projection for ${data.length} item embeddings.`;
        this.updateStatus(this.isTraining ? 'Training...' : 'Training Complete', this.isTraining ? 'orange' : 'green');
    }

    renderVisualization(data) {
        const margin = { top: 20, right: 20, bottom: 30, left: 40 };
        const width = this.visualizationContainer.clientWidth - margin.left - margin.right;
        const height = this.visualizationContainer.clientHeight - margin.top - margin.bottom;

        this.visualizationContainer.innerHTML = ''; // Clear previous chart

        const svg = d3.select("#visualization-container").append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        // Scales
        const xScale = d3.scaleLinear()
            .domain(d3.extent(data, d => d.x)).nice()
            .range([0, width]);

        const yScale = d3.scaleLinear()
            .domain(d3.extent(data, d => d.y)).nice()
            .range([height, 0]);

        // Axes
        svg.append("g")
            .attr("transform", `translate(0,${height})`)
            .call(d3.axisBottom(xScale));

        svg.append("g")
            .call(d3.axisLeft(yScale));

        // Tooltip setup
        const tooltip = d3.select("body").append("div")
            .attr("class", "tooltip")
            .style("position", "absolute")
            .style("background", "#333")
            .style("color", "white")
            .style("padding", "5px")
            .style("border-radius", "3px")
            .style("pointer-events", "none")
            .style("opacity", 0);

        // Dots
        svg.selectAll(".dot")
            .data(data)
            .enter().append("circle")
            .attr("class", "dot")
            .attr("r", 3.5)
            .attr("cx", d => xScale(d.x))
            .attr("cy", d => yScale(d.y))
            .style("fill", "#3498DB")
            .on("mouseover", (event, d) => {
                tooltip.transition()
                    .duration(200)
                    .style("opacity", .9);
                tooltip.html(d.title)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 28) + "px");
            })
            .on("mouseout", () => {
                tooltip.transition()
                    .duration(500)
                    .style("opacity", 0);
            });
    }
}

// Initialize the app when the window loads
window.onload = () => {
    window.app = new App();
};

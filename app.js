// app.js

class MovieLensApp {
    constructor() {
        // Data and Indexing
        this.interactions = []; 
        this.indexedInteractions = []; 
        this.items = new Map(); 
        this.userIds = []; 
        this.itemIds = []; 
        this.userIdxMap = new Map(); 
        this.itemIdxMap = new Map(); 
        this.userTopRated = new Map(); 
        this.model = null;
        
        // Configuration
        this.config = {
            maxInteractions: 80000, 
            embeddingDim: 32,
            batchSize: 512,
            epochs: 10,
            learningRate: 0.001
        };
        
        // UI/Training State
        this.lossHistory = [];
        this.isTraining = false;
        // Ensure contexts are retrieved correctly after the canvas elements exist
        this.lossChartCtx = document.getElementById('lossChart')?.getContext('2d');
        this.embeddingChartCtx = document.getElementById('embeddingChart')?.getContext('2d');
        this.lossChart = null; // New property for Chart.js instance
        this.embeddingChart = null; // New property for Chart.js instance

        this.initializeUI();
    }
    
    initializeUI() {
        document.getElementById('loadData')?.addEventListener('click', () => this.loadData());
        document.getElementById('train')?.addEventListener('click', () => this.train());
        document.getElementById('test')?.addEventListener('click', () => this.test());
        
        this.updateStatus('TensorFlow.js loaded. Click "Load Data" to start.');
    }

    setControls(isLoading, isTrained) {
        document.getElementById('loadData').disabled = isLoading;
        document.getElementById('train').disabled = isLoading || this.isTraining || !isTrained;
        document.getElementById('test').disabled = isLoading || !isTrained;
    }

    // --- Data Loading and Preprocessing ---
    async loadData() {
        this.updateStatus('Loading data files (u.data, u.item)...');
        this.setControls(true, false);

        try {
            // 1. Fetch data
            const [interactionsResponse, itemsResponse] = await Promise.all([
                fetch('u.data'),
                fetch('u.item')
            ]);

            if (!interactionsResponse.ok || !itemsResponse.ok) {
                throw new Error('Could not find u.data or u.item. Ensure files are in a "data" folder and you are running a local server.');
            }

            const rawInteractions = await interactionsResponse.text();
            const rawItems = await itemsResponse.text();

            // 2. Parse and Index Items (u.item)
            rawItems.split('\n').forEach(line => {
                const parts = line.split('|');
                if (parts.length > 1) {
                    const itemId = parseInt(parts[0]);
                    const title = parts[1];
                    this.items.set(itemId, { title: title, genres: parts.slice(5) });
                    this.itemIds.push(itemId);
                }
            });

            // 3. Parse and Index Interactions (u.data)
            const interactions = rawInteractions.split('\n').map(line => {
                const parts = line.split('\t');
                if (parts.length >= 3) {
                    return {
                        userId: parseInt(parts[0]),
                        itemId: parseInt(parts[1]),
                        rating: parseInt(parts[2]),
                        timestamp: parseInt(parts[3])
                    };
                }
                return null;
            }).filter(i => i !== null && this.items.has(i.itemId)); // Filter out nulls and items not in u.item

            // Sort by timestamp and limit
            interactions.sort((a, b) => a.timestamp - b.timestamp);
            this.interactions = interactions.slice(0, this.config.maxInteractions);

            // 4. Create User/Item Maps and Indexed Interactions
            this.userIds = [...new Set(this.interactions.map(i => i.userId))].sort((a, b) => a - b);
            this.userIds.forEach((id, idx) => this.userIdxMap.set(id, idx));

            // Only use items that actually appear in the filtered interactions
            const usedItemIds = [...new Set(this.interactions.map(i => i.itemId))].sort((a, b) => a - b);
            this.itemIds = usedItemIds;
            this.itemIds.forEach((id, idx) => this.itemIdxMap.set(id, idx));

            this.indexedInteractions = this.interactions.map(i => ({
                userIdx: this.userIdxMap.get(i.userId),
                itemIdx: this.itemIdxMap.get(i.itemId),
                rating: i.rating,
                userId: i.userId // Keep original ID for result rendering
            }));

            // 5. Pre-calculate User's Top Rated Items (for testing/comparison)
            const userRatings = new Map();
            this.indexedInteractions.forEach(i => {
                if (!userRatings.has(i.userId)) {
                    userRatings.set(i.userId, []);
                }
                userRatings.get(i.userId).push({ 
                    itemIdx: i.itemIdx, 
                    rating: i.rating, 
                    itemId: this.itemIds[i.itemIdx] // Original Item ID
                });
            });

            this.userTopRated = new Map();
            userRatings.forEach((ratings, userId) => {
                const topItems = ratings
                    .sort((a, b) => b.rating - a.rating || b.itemId - a.itemId) // Sort by rating (desc) then item ID (desc)
                    .slice(0, 10) // Take top 10
                    .map(r => ({ ...r, title: this.items.get(r.itemId).title }));
                this.userTopRated.set(this.userIdxMap.get(userId), topItems);
            });


            // 6. Initialize Model
            const numUsers = this.userIds.length;
            const numItems = this.itemIds.length;
            this.model = new TwoTowerModel(
                numUsers, 
                numItems, 
                this.config.embeddingDim, 
                this.config.learningRate
            );

            this.updateStatus(`Data loaded: ${numUsers} users, ${numItems} items, ${this.interactions.length} interactions.`);
            this.setControls(false, true);

        } catch (error) {
            console.error('Data Loading Error:', error);
            this.updateStatus(`**ERROR:** Could not load data. Ensure 1) You are running a local server (not file://), and 2) **u.data** and **u.item** exist in a 'data/' directory.`);
            this.setControls(false, false);
        }
    }
    
    // --- Training ---
    async train() {
        if (this.isTraining) return;
        this.isTraining = true;
        this.setControls(false, true); // Keep loadData disabled, enable test after train

        this.lossHistory = [];
        this.updateStatus(`Training started for ${this.config.epochs} epochs...`);
        tf.engine().startScope();

        const numInteractions = this.indexedInteractions.length;
        const numBatches = Math.ceil(numInteractions / this.config.batchSize);

        for (let epoch = 1; epoch <= this.config.epochs; epoch++) {
            this.updateStatus(`Epoch ${epoch}/${this.config.epochs}: Preparing data...`);
            
            // Shuffle data before each epoch
            this.indexedInteractions.sort(() => Math.random() - 0.5);

            let epochLoss = 0;
            let batchCount = 0;
            
            for (let i = 0; i < numInteractions; i += this.config.batchSize) {
                const batch = this.indexedInteractions.slice(i, i + this.config.batchSize);
                const userIndices = batch.map(d => d.userIdx);
                const itemIndices = batch.map(d => d.itemIdx);
                
                // Perform a training step
                const lossTensor = await this.model.trainStep(userIndices, itemIndices);
                const loss = await lossTensor.data();
                lossTensor.dispose(); // Clean up the loss tensor

                epochLoss += loss[0];
                batchCount++;
                
                if (batchCount % 20 === 0) { // Update status periodically
                    this.updateStatus(`Epoch ${epoch}/${this.config.epochs}: Batch ${batchCount}/${numBatches}. Loss: ${loss[0].toFixed(4)}`);
                    await tf.nextFrame();
                }
            }

            const avgLoss = epochLoss / batchCount;
            this.lossHistory.push(avgLoss);
            this.plotLoss(this.lossHistory);
            this.updateStatus(`Epoch ${epoch}/${this.config.epochs} finished. Avg Loss: ${avgLoss.toFixed(4)}`);
            await tf.nextFrame();
        }

        tf.engine().endScope();
        this.isTraining = false;
        this.setControls(false, true);
        this.updateStatus(`Training complete. Final Loss: ${this.lossHistory[this.lossHistory.length - 1].toFixed(4)}. Click 'Test Recommendations' or visualize embeddings.`);

        await this.visualizeEmbeddings(); // Visualize after training
    }

    plotLoss(history) {
        if (!this.lossChartCtx) return;

        const data = {
            labels: history.map((_, i) => `Epoch ${i + 1}`),
            datasets: [{
                label: 'Training Loss',
                data: history,
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1,
                fill: false
            }]
        };

        const config = {
            type: 'line',
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    yAxes: [{
                        scaleLabel: {
                            display: true,
                            labelString: 'Loss'
                        }
                    }]
                }
            }
        };

        if (this.lossChart) {
            this.lossChart.data.labels = data.labels;
            this.lossChart.data.datasets[0].data = data.datasets[0].data;
            this.lossChart.update();
        } else {
            this.lossChart = new Chart(this.lossChartCtx, config);
        }
    }
    
    // --- Visualization ---
    async visualizeEmbeddings() {
        if (!this.model || !this.embeddingChartCtx) return;

        this.updateStatus('Computing PCA projection for item embeddings (sampling 1000 items)...');
        await tf.nextFrame();

        const { projection, sampleIndices } = await this.model.computePCAProjection();

        // Map the sampled indices back to original item data
        const itemData = sampleIndices.map(idx => {
            const itemId = this.itemIds[idx];
            return {
                title: this.items.get(itemId).title,
                genre: this.items.get(itemId).genres.findIndex(g => g === '1') // Find first genre
            };
        });
        
        const projectionData = projection.map(([x, y], i) => ({
            x: x, 
            y: y,
            title: itemData[i].title,
            genreIdx: itemData[i].genre
        }));

        // Cleanup existing chart if any
        if (this.embeddingChart) {
            this.embeddingChart.destroy();
        }

        // Simple color mapping for genres (can be expanded)
        const genreColors = [
            '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', 
            '#FF9F40', '#E7E9ED', '#A0E426', '#E42646', '#26E4A0'
        ];
        
        // Group data by genre for better chart legend/coloring
        const datasets = [];
        const uniqueGenres = [...new Set(itemData.map(d => d.genreIdx))].sort((a, b) => a - b);

        uniqueGenres.forEach(genreIdx => {
            const genreData = projectionData.filter(d => d.genreIdx === genreIdx);
            
            // Get the title of the first genre (index 5 in u.item is 'unknown', 6 is 'Action', etc.)
            // The full genre list is ['unknown', 'Action', 'Adventure', 'Animation', ...]
            // genreIdx = -1 if no genre is '1', otherwise 0 to 18.
            const genreTitles = [
                'Unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 
                'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 
                'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
                'Sci-Fi', 'Thriller', 'War', 'Western'
            ];
            const genreLabel = genreIdx >= 0 ? genreTitles[genreIdx] : 'Other/No Genre';
            
            datasets.push({
                label: genreLabel,
                data: genreData,
                backgroundColor: genreColors[genreIdx % genreColors.length],
                pointRadius: 4,
                hoverRadius: 6
            });
        });

        const config = {
            type: 'scatter',
            data: {
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                title: {
                    display: true,
                    text: 'Item Embeddings 2D Projection (PCA)'
                },
                scales: {
                    xAxes: [{
                        scaleLabel: {
                            display: true,
                            labelString: 'PCA Component 1'
                        }
                    }],
                    yAxes: [{
                        scaleLabel: {
                            display: true,
                            labelString: 'PCA Component 2'
                        }
                    }]
                },
                tooltips: {
                    callbacks: {
                        label: function(tooltipItem, data) {
                            const datasetIndex = tooltipItem.datasetIndex;
                            const index = tooltipItem.index;
                            const point = data.datasets[datasetIndex].data[index];
                            return point.title;
                        }
                    }
                }
            }
        };

        this.embeddingChart = new Chart(this.embeddingChartCtx, config);
        this.updateStatus('Embedding visualization complete. Click "Test Recommendations" to proceed.');
    }
    
    // --- Testing/Recommendation ---
    async test() {
        if (!this.model) {
            this.updateStatus('Error: Model not trained or data not loaded.');
            return;
        }

        this.updateStatus('Testing model: Generating recommendations for a random user...');
        
        // Select a random user who has ratings (i.e., is in userIds)
        const randomUserIdx = Math.floor(Math.random() * this.userIds.length);
        const randomUserId = this.userIds[randomUserIdx];
        
        const topRated = this.userTopRated.get(randomUserIdx) || [];

        let recommendations = [];
        
        try {
            await tf.nextFrame();
            
            // 1. Get user embedding
            const userEmbedding = tf.tidy(() => {
                const userTensor = tf.tensor1d([randomUserIdx], 'int32');
                return this.model.userForward(userTensor).squeeze(); // [D]
            });

            // 2. Compute scores for all items
            const allScores = await this.model.scoreAllItems(userEmbedding); // standard JS array of length numItems
            
            userEmbedding.dispose();

            // 3. Find top N items (excluding items the user has already rated)
            const userRatedItemIds = new Set(topRated.map(i => i.itemId));
            const topN = 10;
            
            // Create a list of {itemIdx, score} for sorting
            const scoredItems = allScores.map((score, idx) => ({
                itemIdx: idx,
                score: score,
                itemId: this.itemIds[idx] // Original Item ID
            }));

            // Filter out items the user has already rated (or seen/rated highly)
            const candidateItems = scoredItems.filter(item => !userRatedItemIds.has(item.itemId));

            // Sort by score (descending)
            candidateItems.sort((a, b) => b.score - a.score);

            // Take top N recommendations
            recommendations = candidateItems.slice(0, topN).map(r => ({
                title: this.items.get(r.itemId).title,
                score: r.score.toFixed(4)
            }));
            
            this.updateStatus(`Recommendations generated for User ID: ${randomUserId}.`);

        } catch (e) {
            console.error('Recommendation Error:', e);
            this.updateStatus('Recommendation failed. Check console for details.');
        } finally {
            this.renderResults(randomUserId, topRated, recommendations);
        }
    }

    renderResults(userId, topRated, recommendations) {
        const resultsContainer = document.getElementById('results-container');
        
        const topRatedList = topRated.map(r => 
            `<li>**${r.title}** (Rating: ${r.rating})</li>`
        ).join('');

        const recsList = recommendations.map(r => 
            `<li>**${r.title}** (Score: ${r.score})</li>`
        ).join('');

        resultsContainer.innerHTML = `
            <h3>Recommendations for User ID: ${userId}</h3>
            <div class="user-info">
                <div>
                    <h4>User's Top Rated Movies (for comparison)</h4>
                    <ul>${topRatedList || '<li>No top-rated items found for this user.</li>'}</ul>
                </div>
                <div>
                    <h4>Top ${recommendations.length} Recommendations</h4>
                    <ul>${recsList || '<li>No recommendations generated.</li>'}</ul>
                </div>
            </div>
        `;
    }

    updateStatus(message) {
        const statusElement = document.getElementById('status');
        if (statusElement) {
            statusElement.innerHTML = message; 
        }
    }
}

let app;
document.addEventListener('DOMContentLoaded', () => {
    if (typeof tf === 'undefined') {
        document.getElementById('status').textContent = 'Error: TensorFlow.js not loaded. Check network connection.';
        return;
    }
    // FIX: Set backend and initialize app after DOM is ready and TF.js is loaded
    tf.setBackend('webgl').then(() => {
        app = new MovieLensApp();
    }).catch(e => {
        console.warn("WebGL backend failed, falling back to CPU. Performance may be degraded.", e);
        tf.setBackend('cpu');
        app = new MovieLensApp();
    });
});

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
        document.getElementById('train').disabled = isLoading || !this.interactions.length;
        document.getElementById('test').disabled = isLoading || !isTrained;
    }
    
    async loadData() {
        this.setControls(true, false);
        this.updateStatus('Loading and parsing data... **Check console for errors if this step fails.**');
        
        try {
            // FIX: Updated to fetch directly from the root directory
            const [interactionsResponse, itemsResponse] = await Promise.all([
                fetch('u.data'), // Updated path
                fetch('u.item')  // Updated path
            ]);

            if (!interactionsResponse.ok || !itemsResponse.ok) {
                 throw new Error(`Failed to fetch data. Status: u.data ${interactionsResponse.status}, u.item ${itemsResponse.status}. Verify file paths and server/CORS settings.`);
            }

            const [interactionsText, itemsText] = await Promise.all([
                interactionsResponse.text(),
                itemsResponse.text()
            ]);
            
            // --- Parse u.item (Movie Metadata) ---
            itemsText.trim().split('\n').forEach(line => {
                const fields = line.split('|');
                const itemId = parseInt(fields[0]);
                
                const titleMatch = fields[1]?.match(/(.*) \((\d{4})\)$/);
                let title = fields[1];
                let year = 'N/A';
                if (titleMatch) {
                    title = titleMatch[1];
                    year = titleMatch[2];
                }
                
                this.items.set(itemId, { itemId, title, year });
            });

            // --- Parse u.data (Interactions) and Build Indexers ---
            const rawInteractions = interactionsText.trim().split('\n')
                .slice(0, this.config.maxInteractions) 
                .map(line => {
                    const parts = line.split('\t');
                    if (parts.length < 4) return null; // Skip malformed lines
                    const [userId, itemId, rating] = parts.map(Number);
                    return { userId, itemId, rating };
                }).filter(i => i !== null); // Filter out nulls

            // Build 0-based unique indexers
            this.userIds = [...new Set(rawInteractions.map(i => i.userId))];
            this.itemIds = [...new Set(rawInteractions.map(i => i.itemId))].filter(id => this.items.has(id));
            
            this.userIds.forEach((id, index) => this.userIdxMap.set(id, index));
            this.itemIds.forEach((id, index) => this.itemIdxMap.set(id, index));
            
            this.interactions = rawInteractions.filter(i => this.itemIdxMap.has(i.itemId));
            
            this.indexedInteractions = this.interactions.map(i => ({
                userIdx: this.userIdxMap.get(i.userId),
                itemIdx: this.itemIdxMap.get(i.itemId),
            }));
            
            // Pre-calculate user top-rated for test comparison
            this.interactions.forEach(i => {
                const ratings = this.userTopRated.get(i.userId) || [];
                ratings.push({ itemId: i.itemId, rating: i.rating });
                this.userTopRated.set(i.userId, ratings);
            });


            this.updateStatus(`Data loaded: ${this.userIds.length} users, ${this.itemIds.length} items, ${this.interactions.length} interactions.`);
            this.setControls(false, false);

        } catch (error) {
            console.error('Data Loading Error:', error);
            this.updateStatus(`**ERROR:** Could not load data. Ensure 1) You are running a local server (not file://), and 2) **u.data** and **u.item** exist.`);
            this.setControls(false, false);
        }
    }

    async train() {
        if (this.isTraining) return;
        this.isTraining = true;
        this.setControls(true, false);
        this.updateStatus('Training started. Check console for progress...');

        const { numUsers, numItems, embeddingDim, epochs, batchSize } = {
            numUsers: this.userIds.length,
            numItems: this.itemIds.length,
            ...this.config
        };

        // Initialize model if it doesn't exist
        if (!this.model) {
            this.model = new TwoTowerModel(numUsers, numItems, embeddingDim, this.config.learningRate);
        }

        const userIndices = this.indexedInteractions.map(i => i.userIdx);
        const itemIndices = this.indexedInteractions.map(i => i.itemIdx);
        const trainingData = tf.data.zip({
            user: tf.data.array(userIndices),
            item: tf.data.array(itemIndices)
        }).shuffle(userIndices.length).batch(batchSize);

        this.lossHistory = [];
        let epochCounter = 0;
        
        await trainingData.forEachAsync(async batch => {
            const loss = await this.model.trainStep(batch.user.arraySync(), batch.item.arraySync());
            this.lossHistory.push(loss);
            
            // Update loss chart frequently but not on every batch
            if (this.lossHistory.length % 50 === 0) {
                this.plotLoss(this.lossHistory);
                this.updateStatus(`Epoch ${Math.floor(epochCounter / (userIndices.length / batchSize)) + 1}/${epochs}. Loss: ${loss.toFixed(4)}`);
                await tf.nextFrame(); // Yield to the UI thread
            }

            epochCounter++;
            if (Math.floor(epochCounter / (userIndices.length / batchSize)) >= epochs) {
                return false; // Stop iteration
            }
        });
        
        this.isTraining = false;
        this.updateStatus(`Training finished after ${epochs} epochs. Final Loss: ${this.lossHistory[this.lossHistory.length - 1].toFixed(4)}`);
        
        await this.visualizeEmbeddings();
        this.setControls(false, true);
    }

    plotLoss(history) {
        if (!this.lossChartCtx) return;

        const maxLoss = Math.max(...history);
        const labels = history.map((_, i) => i);

        this.lossChartCtx.clearRect(0, 0, 800, 300);
        this.lossChartCtx.fillStyle = '#ccc';
        this.lossChartCtx.fillRect(0, 0, 800, 300);
        this.lossChartCtx.fillStyle = '#333';
        this.lossChartCtx.fillText(`Loss data points: ${history.length}. Max Loss: ${maxLoss.toFixed(2)}`, 10, 20);
    }
    
    async visualizeEmbeddings() {
        this.updateStatus('Computing PCA projection for item embeddings...');
        const itemMap = new Map(this.itemIds.map(id => [this.itemIdxMap.get(id), this.items.get(id)]));
        
        const { projection, sampleIndices } = await this.model.computePCAProjection(1000);

        if (!this.embeddingChartCtx) return;
        
        this.embeddingChartCtx.clearRect(0, 0, 800, 600);
        this.embeddingChartCtx.fillStyle = '#ccc';
        this.embeddingChartCtx.fillRect(0, 0, 800, 600);
        this.embeddingChartCtx.fillStyle = '#333';
        this.embeddingChartCtx.fillText(`PCA calculated for ${projection.length} items. Ready for visualization.`, 10, 20);
        this.updateStatus('Embeddings visualized. Ready for testing.');
    }
    
    async test() {
        this.setControls(true, true);
        this.updateStatus('Testing model...');
        
        // Pick a random user to test
        const randomUserIdx = Math.floor(Math.random() * this.userIds.length);
        const randomUserId = this.userIds[randomUserIdx];
        
        // 1. Get user embedding
        const userEmbeddingTensor = this.model.userForward(tf.tensor1d([randomUserIdx], 'int32')).squeeze();

        // 2. Score all items
        const allScores = await this.model.scoreItems(userEmbeddingTensor);
        
        // 3. Find top 5 items
        const itemScores = Array.from(allScores).map((score, idx) => ({
            itemIdx: idx,
            score: score
        }));

        itemScores.sort((a, b) => b.score - a.score);
        
        const top5Recommendations = itemScores.slice(0, 5).map(i => {
            const itemId = this.itemIds[i.itemIdx];
            return {
                ...this.items.get(itemId),
                score: i.score
            };
        });

        const topRatedOriginal = this.userTopRated.get(randomUserId)
            .sort((a, b) => b.rating - a.rating)
            .slice(0, 5)
            .map(i => this.items.get(i.itemId));
        
        this.renderResults(randomUserId, topRatedOriginal, top5Recommendations);
        this.setControls(false, true);
        this.updateStatus('Test completed. Results rendered.');
    }

    renderResults(userId, topRated, recommendations) {
        const resultsContainer = document.getElementById('results-container');
        resultsContainer.innerHTML = `
            <h2>Recommendations for User ${userId}</h2>
            <div class="test-sections">
                <div class="top-rated-section">
                    <h3>User's Top 5 Rated Movies (From Training Data)</h3>
                    <ol>
                        ${topRated.map(m => `<li>${m.title} (${m.year})</li>`).join('')}
                    </ol>
                </div>
                <div class="recommendations-section">
                    <h3>Top 5 Recommendations (Model Prediction)</h3>
                    <ol>
                        ${recommendations.map(m => `<li>${m.title} (${m.year}) - Score: ${m.score.toFixed(3)}</li>`).join('')}
                    </ol>
                </div>
            </div>
        `;
    }

    updateStatus(message) {
        document.getElementById('status').textContent = message; 
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

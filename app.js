// app.js

class MovieLensApp {
    constructor() {
        // Data and Indexing
        this.interactions = [];
        this.items = new Map();
        this.userMap = new Map(); // Map original userId to 0-based index
        this.itemMap = new Map(); // Map original itemId to 0-based index
        this.reverseUserMap = new Map(); // Map 0-based index back to original userId
        this.reverseItemMap = new Map(); // Map 0-based index back to original itemId
        this.userTopRated = new Map(); // Stores all interactions grouped by user
        this.model = null;

        // Configuration
        this.config = {
            maxInteractions: 80000,
            embeddingDim: 32,
            batchSize: 512,
            epochs: 20,
            learningRate: 0.001
        };

        // UI/Training State
        this.lossHistory = [];
        this.isTraining = false;
        
        // Canvas contexts (fetched upon initialization)
        this.lossChartCtx = null;
        this.embeddingChartCtx = null;

        this.initializeUI();
    }

    initializeUI() {
        document.getElementById('loadData')?.addEventListener('click', () => this.loadData());
        document.getElementById('train')?.addEventListener('click', () => this.train());
        document.getElementById('test')?.addEventListener('click', () => this.test());
        
        this.lossChartCtx = document.getElementById('lossChart')?.getContext('2d');
        this.embeddingChartCtx = document.getElementById('embeddingChart')?.getContext('2d');

        this.setControls(false, false);
        this.updateStatus('TensorFlow.js loaded. Click "Load Data" to start.');
    }

    setControls(isLoading, isTrained) {
        document.getElementById('loadData').disabled = isLoading;
        // Check for presence of elements before setting disabled
        const trainButton = document.getElementById('train');
        if (trainButton) trainButton.disabled = isLoading || !this.interactions.length;
        const testButton = document.getElementById('test');
        if (testButton) testButton.disabled = isLoading || !isTrained;
    }

    async loadData() {
        this.setControls(true, false);
        this.updateStatus('Loading and parsing data...');

        try {
            // FIX: Corrected paths to fetch directly from the root directory
            const [interactionsResponse, itemsResponse] = await Promise.all([
                fetch('u.data'), // Corrected path
                fetch('u.item')  // Corrected path
            ]);

            if (!interactionsResponse.ok || !itemsResponse.ok) {
                throw new Error('Server request failed. Check file paths and server status.');
            }

            const [interactionsText, itemsText] = await Promise.all([
                interactionsResponse.text(),
                itemsResponse.text()
            ]);

            // --- 1. Parse u.data (Interactions) ---
            const interactionsLines = interactionsText.trim().split('\n');
            this.interactions = interactionsLines.slice(0, this.config.maxInteractions).map(line => {
                const [userId, itemId, rating, timestamp] = line.split('\t');
                return {
                    userId: parseInt(userId),
                    itemId: parseInt(itemId),
                    rating: parseFloat(rating),
                    timestamp: parseInt(timestamp)
                };
            });

            // --- 2. Parse u.item (Movie Metadata) ---
            const itemsLines = itemsText.trim().split('\n');
            itemsLines.forEach(line => {
                const parts = line.split('|');
                const itemId = parseInt(parts[0]);
                const title = parts[1];
                const yearMatch = title.match(/\((\d{4})\)$/);
                const year = yearMatch ? parseInt(yearMatch[1]) : null;

                this.items.set(itemId, {
                    title: title.replace(/\(\d{4}\)$/, '').trim(),
                    year: year
                });
            });

            // --- 3. Create Mappings and Identify Qualified Users ---
            this.createMappings();
            this.findQualifiedUsers();

            const numUsers = this.userMap.size;
            const numItems = this.itemMap.size;
            const numQualified = this.qualifiedUsers.length;

            this.updateStatus(`Data loaded: ${numUsers} users, ${numItems} items, ${this.interactions.length} interactions. ${numQualified} users qualified for testing.`);
            this.setControls(false, false);

        } catch (error) {
            console.error('Data Loading Error:', error);
            this.updateStatus(`**ERROR:** Could not load data. Ensure 1) You are running a local server (not file://), and 2) **u.data** and **u.item** exist in the root folder.`);
            this.setControls(false, false);
        }
    }

    createMappings() {
        // Create user and item mappings to 0-based indices
        const userSet = new Set(this.interactions.map(i => i.userId));
        const itemSet = new Set(this.interactions.map(i => i.itemId));

        Array.from(userSet).forEach((userId, index) => {
            this.userMap.set(userId, index);
            this.reverseUserMap.set(index, userId);
        });

        Array.from(itemSet).forEach((itemId, index) => {
            this.itemMap.set(itemId, index);
            this.reverseItemMap.set(index, itemId);
        });

        // Group interactions by user
        const userInteractions = new Map();
        this.interactions.forEach(interaction => {
            const userId = interaction.userId;
            if (!userInteractions.has(userId)) {
                userInteractions.set(userId, []);
            }
            userInteractions.get(userId).push(interaction);
        });

        // Sort each user's interactions by rating (desc) and timestamp (desc)
        userInteractions.forEach((interactions, userId) => {
            interactions.sort((a, b) => {
                if (b.rating !== a.rating) return b.rating - a.rating;
                return b.timestamp - a.timestamp;
            });
        });

        this.userTopRated = userInteractions;
    }

    findQualifiedUsers() {
        // Filter users with at least 20 ratings
        this.qualifiedUsers = [];
        this.userTopRated.forEach((interactions, userId) => {
            if (interactions.length >= 20) {
                this.qualifiedUsers.push(userId);
            }
        });
    }

    async train() {
        if (this.isTraining) return;

        this.isTraining = true;
        this.setControls(true, false);
        this.lossHistory = [];

        this.updateStatus('Initializing model...');

        // Initialize model
        this.model = new TwoTowerModel(
            this.userMap.size,
            this.itemMap.size,
            this.config.embeddingDim,
            this.config.learningRate // Pass learning rate to model
        );

        // Prepare training data (0-based indices)
        const userIndices = this.interactions.map(i => this.userMap.get(i.userId));
        const itemIndices = this.interactions.map(i => this.itemMap.get(i.itemId));

        this.updateStatus('Starting training...');

        // Training loop
        const numBatches = Math.ceil(userIndices.length / this.config.batchSize);

        for (let epoch = 0; epoch < this.config.epochs; epoch++) {
            let epochLoss = 0;

            // Simple shuffle by iterating over shuffled indices (not fully randomized)
            const shuffledIndices = Array.from({length: userIndices.length}, (_, i) => i).sort(() => Math.random() - 0.5);

            for (let batch = 0; batch < numBatches; batch++) {
                const start = batch * this.config.batchSize;
                const end = Math.min(start + this.config.batchSize, userIndices.length);
                
                // Get batch indices from the shuffled global indices
                const batchGlobalIndices = shuffledIndices.slice(start, end);

                const batchUsers = batchGlobalIndices.map(i => userIndices[i]);
                const batchItems = batchGlobalIndices.map(i => itemIndices[i]);

                const loss = await this.model.trainStep(batchUsers, batchItems);
                epochLoss += loss;

                this.lossHistory.push(loss);
                this.updateLossChart();

                if (batch % 50 === 0) { // Update status less frequently to save performance
                    this.updateStatus(`Epoch ${epoch + 1}/${this.config.epochs}, Batch ${batch}/${numBatches}, Loss: ${loss.toFixed(4)}`);
                    await tf.nextFrame(); // Yield to the UI thread
                }
            }

            epochLoss /= numBatches;
            this.updateStatus(`Epoch ${epoch + 1}/${this.config.epochs} completed. Average loss: ${epochLoss.toFixed(4)}`);
        }

        this.isTraining = false;
        this.setControls(false, true); // Set isTrained to true
        
        this.updateStatus('Training completed! Click "Test" to see recommendations.');

        // Visualize embeddings
        this.visualizeEmbeddings();
    }

    updateLossChart() {
        if (!this.lossChartCtx) return;

        const canvas = document.getElementById('lossChart');
        const ctx = this.lossChartCtx;
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (this.lossHistory.length === 0) return;
        
        const history = this.lossHistory;
        const maxLoss = Math.max(...history);
        const minLoss = Math.min(...history);
        // Add a small buffer for visualization
        const range = (maxLoss - minLoss) * 1.05 || 1; 
        
        ctx.strokeStyle = '#007acc';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        history.forEach((loss, index) => {
            const x = (index / history.length) * canvas.width;
            // Map loss value to canvas y-coordinate, accounting for buffer
            const y = canvas.height - 10 - (((loss - minLoss) / range) * (canvas.height - 20));
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.stroke();
        
        // Add labels
        ctx.fillStyle = '#000';
        ctx.font = '12px Arial';
        ctx.fillText(`Min: ${minLoss.toFixed(4)}`, 10, canvas.height - 2);
        ctx.fillText(`Max: ${maxLoss.toFixed(4)}`, 10, 15);
    }

    async visualizeEmbeddings() {
        if (!this.model || !this.embeddingChartCtx) return;
        
        this.updateStatus('Computing embedding visualization (PCA)...');
        
        await tf.nextFrame();

        const canvas = document.getElementById('embeddingChart');
        const ctx = this.embeddingChartCtx;
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        try {
            // Use the model's more robust PCA function
            const { projection, sampleIndices } = await this.model.computePCAProjection(1000);
            
            // Get item metadata for the sampled indices
            const sampledItemIds = sampleIndices.map(i => this.reverseItemMap.get(i));
            const sampledItems = sampledItemIds.map(id => this.items.get(id));

            // Normalize to canvas coordinates
            const xs = projection.map(p => p[0]);
            const ys = projection.map(p => p[1]);
            
            const xMin = Math.min(...xs);
            const xMax = Math.max(...xs);
            const yMin = Math.min(...ys);
            const yMax = Math.max(...ys);
            
            const xRange = xMax - xMin || 1;
            const yRange = yMax - yMin || 1;
            
            // Draw points
            ctx.fillStyle = 'rgba(0, 122, 204, 0.6)';
            
            projection.forEach((proj, i) => {
                // Map x,y coordinates to canvas space
                const x = ((proj[0] - xMin) / xRange) * (canvas.width - 40) + 20;
                // Invert Y axis for standard chart visualization
                const y = canvas.height - (((proj[1] - yMin) / yRange) * (canvas.height - 40) + 20); 
                
                ctx.beginPath();
                ctx.arc(x, y, 3, 0, 2 * Math.PI);
                ctx.fill();

                // Optional: Draw text label for a few items
                if (i % 100 === 0) {
                    ctx.fillStyle = '#555';
                    ctx.font = '10px Arial';
                    ctx.fillText(sampledItems[i].title, x + 5, y);
                    ctx.fillStyle = 'rgba(0, 122, 204, 0.6)'; // reset color
                }
            });
            
            // Add title and labels
            ctx.fillStyle = '#000';
            ctx.font = '14px Arial';
            ctx.fillText('Item Embeddings Projection (PCA)', 10, 20);
            ctx.font = '12px Arial';
            ctx.fillText(`Showing ${projection.length} sampled items`, 10, 40);
            
            this.updateStatus('Embedding visualization completed.');
        } catch (error) {
            console.error('Visualization Error:', error);
            this.updateStatus(`Error in visualization: ${error.message}`);
        }
    }
    
    // Note: The previous custom computePCA is removed in favor of the one in two-tower.js

    async test() {
        if (!this.model || this.qualifiedUsers.length === 0) {
            this.updateStatus('Model not trained or no qualified users found.');
            return;
        }

        this.setControls(true, true);
        this.updateStatus('Generating recommendations...');

        try {
            // Pick random qualified user
            const randomUser = this.qualifiedUsers[Math.floor(Math.random() * this.qualifiedUsers.length)];
            const userInteractions = this.userTopRated.get(randomUser);
            const userIndex = this.userMap.get(randomUser);

            let topRecommendations = await tf.tidy(async () => {
                // 1. Get user embedding
                const userEmbTensor = this.model.userForward(tf.tensor1d([userIndex], 'int32')).squeeze();

                // 2. Get scores for all items
                const allItemScores = await this.model.scoreItems(userEmbTensor);
                
                // Array of { itemId, score, itemIndex }
                const candidateScores = [];
                const ratedItemIds = new Set(userInteractions.map(i => i.itemId));

                allItemScores.forEach((score, itemIndex) => {
                    const itemId = this.reverseItemMap.get(itemIndex);
                    // Filter out items the user has already rated
                    if (!ratedItemIds.has(itemId)) {
                        candidateScores.push({ itemId, score, itemIndex });
                    }
                });

                // Sort by score descending and take top 10
                candidateScores.sort((a, b) => b.score - a.score);
                return candidateScores.slice(0, 10);
            }); // End tf.tidy

            // Display results
            this.displayResults(randomUser, userInteractions, topRecommendations);
            this.setControls(false, true);

        } catch (error) {
            console.error('Test Error:', error);
            this.updateStatus(`Error generating recommendations: ${error.message}`);
            this.setControls(false, true);
        }
    }

    displayResults(userId, userInteractions, recommendations) {
        const resultsDiv = document.getElementById('results-container'); // Corrected ID usage from previous version
        
        const topRated = userInteractions.slice(0, 10);

        let html = `
            <h2>Recommendations for User ${userId}</h2>
            <div class="side-by-side">
                <div style="flex: 1;">
                    <h3>Top 10 Rated Movies (Historical)</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Movie</th>
                                <th>Rating</th>
                                <th>Year</th>
                            </tr>
                        </thead>
                        <tbody>
        `;
        
        topRated.forEach((interaction, index) => {
            const item = this.items.get(interaction.itemId);
            html += `
                <tr>
                    <td>${index + 1}</td>
                    <td>${item.title}</td>
                    <td>${interaction.rating}</td>
                    <td>${item.year || 'N/A'}</td>
                </tr>
            `;
        });
        
        html += `
                        </tbody>
                    </table>
                </div>
                <div style="flex: 1;">
                    <h3>Top 10 Recommended Movies (Unseen)</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Movie</th>
                                <th>Score</th>
                                <th>Year</th>
                            </tr>
                        </thead>
                        <tbody>
        `;
        
        recommendations.forEach((rec, index) => {
            const item = this.items.get(rec.itemId);
            html += `
                <tr>
                    <td>${index + 1}</td>
                    <td>${item.title}</td>
                    <td>${rec.score.toFixed(4)}</td>
                    <td>${item.year || 'N/A'}</td>
                </tr>
            `;
        });
        
        html += `
                        </tbody>
                    </table>
                </div>
            </div>
        `;
        
        resultsDiv.innerHTML = html;
        this.updateStatus('Recommendations generated successfully!');
    }

    updateStatus(message) {
        document.getElementById('status').textContent = message;
    }
}

// Initialize app when page loads
let app;
document.addEventListener('DOMContentLoaded', () => {
    // Check if TensorFlow.js is loaded (should be loaded via <script> tag in index.html)
    if (typeof tf === 'undefined') {
        document.getElementById('status').textContent = 'Error: TensorFlow.js not loaded. Check network connection.';
        return;
    }
    
    // Set backend and initialize app
    tf.setBackend('webgl').then(() => {
        app = new MovieLensApp();
    }).catch(e => {
        console.warn("WebGL backend failed, falling back to CPU. Performance may be degraded.", e);
        tf.setBackend('cpu');
        app = new MovieLensApp();
    });
});

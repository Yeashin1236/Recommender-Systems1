// app.js

class MovieLensApp {
    constructor() {
        this.interactions = []; 
        this.indexedInteractions = []; 
        this.items = new Map(); 
        this.userIds = []; 
        this.itemIds = []; 
        this.userIdxMap = new Map(); 
        this.itemIdxMap = new Map(); 
        this.userTopRated = new Map(); 
        this.model = null;
        
        this.config = {
            maxInteractions: 80000, 
            embeddingDim: 32,
            batchSize: 512,
            epochs: 10,
            learningRate: 0.001
        };
        
        this.lossHistory = [];
        this.isTraining = false;
        this.lossChartCtx = document.getElementById('lossChart').getContext('2d');
        this.embeddingChartCtx = document.getElementById('embeddingChart').getContext('2d');

        this.initializeUI();
    }
    
    initializeUI() {
        document.getElementById('loadData').addEventListener('click', () => this.loadData());
        document.getElementById('train').addEventListener('click', () => this.train());
        document.getElementById('test').addEventListener('click', () => this.test());
        
        this.updateStatus('TensorFlow.js loaded. Click "Load Data" to start.');
    }

    setControls(isLoading, isTrained) {
        document.getElementById('loadData').disabled = isLoading;
        document.getElementById('train').disabled = isLoading || !this.interactions.length;
        document.getElementById('test').disabled = isLoading || !isTrained;
    }
    
    async loadData() {
        this.setControls(true, false);
        this.updateStatus('Loading and parsing data...');
        
        try {
            // Fetch data from relative paths (data/u.data and data/u.item)
            const [interactionsResponse, itemsResponse] = await Promise.all([
                fetch('data/u.data'),
                fetch('data/u.item')
            ]);
            const [interactionsText, itemsText] = await Promise.all([
                interactionsResponse.text(),
                itemsResponse.text()
            ]);
            
            // --- Parse u.item (Movie Metadata) ---
            itemsText.trim().split('\n').forEach(line => {
                const fields = line.split('|');
                const itemId = parseInt(fields[0]);
                const titleMatch = fields[1].match(/(.*) \((\d{4})\)$/);
                
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
                    const [userId, itemId, rating, timestamp] = line.split('\t').map(Number);
                    return { userId, itemId, rating, timestamp };
                });

            // Build 0-based unique indexers (Maps for forward lookup, Arrays for reverse lookup)
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
            this.updateStatus(`Error loading data. Check console. Ensure "data/u.data" and "data/u.item" exist.`);
            this.setControls(false, false);
        }
    }

    async train() {
        if (this.isTraining || !this.indexedInteractions.length) return;
        this.isTraining = true;
        this.setControls(true, false);
        document.getElementById('results-container').innerHTML = '';
        
        this.model = new TwoTowerModel(
            this.userIds.length, 
            this.itemIds.length, 
            this.config.embeddingDim,
            this.config.learningRate
        );
        
        this.lossHistory = [];
        this.plotLoss([]); 

        const { batchSize, epochs } = this.config;
        const numInteractions = this.indexedInteractions.length;
        const numBatches = Math.ceil(numInteractions / batchSize);
        
        this.updateStatus(`Training started for ${epochs} epochs...`);

        const userIndices = this.indexedInteractions.map(i => i.userIdx);
        const itemIndices = this.indexedInteractions.map(i => i.itemIdx);

        for (let epoch = 0; epoch < epochs; epoch++) {
            const shuffledIndices = tf.util.createShuffledIndices(numInteractions);
            
            let totalLoss = 0;

            for (let b = 0; b < numBatches; b++) {
                const start = b * batchSize;
                const end = Math.min((b + 1) * batchSize, numInteractions);
                const batchIndices = shuffledIndices.slice(start, end);

                const batchUserIdx = batchIndices.map(i => userIndices[i]);
                const batchItemIdx = batchIndices.map(i => itemIndices[i]);
                
                const loss = await this.model.trainStep(batchUserIdx, batchItemIdx);
                totalLoss += loss;
                
                // Yield control to the UI for responsiveness
                if (b % 50 === 0 || b === numBatches - 1) {
                    this.updateStatus(`Epoch ${epoch + 1}/${epochs} - Batch ${b + 1}/${numBatches}. Current Loss: ${loss.toFixed(6)}`);
                    await tf.nextFrame(); 
                }
            }

            const avgLoss = totalLoss / numBatches;
            this.lossHistory.push(avgLoss);
            this.plotLoss(this.lossHistory);
            this.updateStatus(`Epoch ${epoch + 1}/${epochs} completed. Avg Loss: ${avgLoss.toFixed(6)}`);
            await tf.nextFrame();
        }

        this.isTraining = false;
        this.updateStatus('Training complete. Generating visualizations...');
        
        await this.visualizeEmbeddings();
        this.setControls(false, true);
        this.updateStatus('Training complete. Click "Test" for recommendations.');
    }

    plotLoss(history) {
        const ctx = this.lossChartCtx;
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;
        
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = '#f9f9f9';
        ctx.fillRect(0, 0, width, height);

        if (history.length === 0) return;

        const maxLoss = Math.max(...history);
        const minLoss = Math.min(...history);
        const yRange = maxLoss - minLoss;
        const padding = 20;
        const plotWidth = width - 2 * padding;
        const plotHeight = height - 2 * padding;

        ctx.strokeStyle = 'blue';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        history.forEach((loss, i) => {
            const x = padding + (i / (history.length - 1)) * plotWidth;
            const y = padding + plotHeight * (1 - (loss - minLoss) / yRange);
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        ctx.stroke();

        ctx.strokeStyle = '#ccc';
        ctx.lineWidth = 1;
        ctx.strokeRect(padding, padding, plotWidth, plotHeight);

        ctx.fillStyle = '#000';
        ctx.font = '10px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Epoch', width / 2, height - 5);
        ctx.textAlign = 'left';
        ctx.fillText(maxLoss.toFixed(2), padding / 4, padding + 10);
        ctx.fillText(minLoss.toFixed(2), padding / 4, height - padding - 5);
    }

    async visualizeEmbeddings() {
        if (!this.model) return;
        this.updateStatus('Computing 2D PCA projection of item embeddings...');

        const { projection, sampleIndices } = await this.model.computePCAProjection(1000);

        const ctx = this.embeddingChartCtx;
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = '#f9f9f9';
        ctx.fillRect(0, 0, width, height);

        const xCoords = projection.map(p => p[0]);
        const yCoords = projection.map(p => p[1]);
        const minX = Math.min(...xCoords);
        const maxX = Math.max(...xCoords);
        const minY = Math.min(...yCoords);
        const maxY = Math.max(...yCoords);
        
        const padding = 40;
        const plotWidth = width - 2 * padding;
        const plotHeight = height - 2 * padding;
        const xRange = maxX - minX;
        const yRange = maxY - minY;

        ctx.fillStyle = 'rgba(75, 192, 192, 0.5)';
        const pointSize = 3;

        projection.forEach(p => {
            const x = padding + ((p[0] - minX) / xRange) * plotWidth;
            // Flip Y-axis:
            const y = padding + plotHeight * (1 - (p[1] - minY) / yRange); 
            
            ctx.beginPath();
            ctx.arc(x, y, pointSize, 0, 2 * Math.PI);
            ctx.fill();
        });

        // Add hover listener for titles
        const canvas = ctx.canvas;
        canvas.onmousemove = (event) => {
            const rect = canvas.getBoundingClientRect();
            const mouseX = event.clientX - rect.left;
            const mouseY = event.clientY - rect.top;

            let closestItem = null;
            let minDist = Infinity;

            projection.forEach((p, i) => {
                const x = padding + ((p[0] - minX) / xRange) * plotWidth;
                const y = padding + plotHeight * (1 - (p[1] - minY) / yRange);
                const dist = Math.sqrt(Math.pow(mouseX - x, 2) + Math.pow(mouseY - y, 2));

                if (dist < pointSize * 2 && dist < minDist) {
                    minDist = dist;
                    const itemIdx = sampleIndices[i];
                    const originalId = this.itemIds[itemIdx];
                    closestItem = this.items.get(originalId);
                }
            });

            canvas.title = closestItem ? `${closestItem.title} (${closestItem.year})` : '';
        };

        this.updateStatus('2D Item Embeddings projection complete.');
    }


    async test() {
        if (!this.model) {
            this.updateStatus('Please train the model first.');
            return;
        }
        this.setControls(true, true);
        this.updateStatus('Generating recommendations...');

        // 1. Select a random user with at least 20 ratings
        const eligibleUsers = this.userIds.filter(id => (this.userTopRated.get(id)?.length || 0) >= 20);
        if (eligibleUsers.length === 0) {
            this.updateStatus('No users with 20+ ratings found for testing. Cannot proceed.');
            this.setControls(false, true);
            return;
        }

        const randomUserOriginalId = eligibleUsers[Math.floor(Math.random() * eligibleUsers.length)];
        const randomUserIdx = this.userIdxMap.get(randomUserOriginalId);
        
        // 2. Get user's top-rated movies (for comparison)
        const userRatings = this.userTopRated.get(randomUserOriginalId) || [];
        const topRatedMovies = userRatings
            .sort((a, b) => b.rating - a.rating)
            .slice(0, 10)
            .map(r => ({
                ...r,
                title: this.items.get(r.itemId).title,
                year: this.items.get(r.itemId).year,
            }));

        const alreadyRatedItemIds = new Set(userRatings.map(r => r.itemId));

        // 3. Get user embedding
        const userEmbedding = this.model.getUserEmbedding(randomUserIdx);

        // 4. Compute scores against all items
        const allScores = await this.model.getScoresForAllItems(userEmbedding); 
        userEmbedding.dispose(); // CRITICAL FIX: Clean up tensor after use

        // 5. Build and filter recommendations
        const recommendations = [];
        for (let i = 0; i < allScores.length; i++) {
            const originalItemId = this.itemIds[i];
            
            // Filter out items the user has already rated (required)
            if (!alreadyRatedItemIds.has(originalItemId)) {
                recommendations.push({
                    itemId: originalItemId,
                    score: allScores[i],
                });
            }
        }
        
        // 6. Sort and get top 10
        const top10Recommendations = recommendations
            .sort((a, b) => b.score - a.score)
            .slice(0, 10);
        
        // 7. Render results
        this.renderResults(randomUserOriginalId, topRatedMovies, top10Recommendations);

        this.setControls(false, true);
    }
    
    renderResults(userId, topRated, recommendations) {
        const resultsDiv = document.getElementById('results-container');
        
        let html = `
            <h2>Recommendations for User ${userId}</h2>
            <div class="side-by-side">
                <div>
                    <h3>Top 10 Rated Movies (Observed)</h3>
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
        
        topRated.forEach((rated, index) => {
            html += `
                <tr>
                    <td>${index + 1}</td>
                    <td>${rated.title}</td>
                    <td>${rated.rating}</td>
                    <td>${rated.year}</td>
                </tr>
            `;
        });
        
        html += `
                        </tbody>
                    </table>
                </div>
                <div>
                    <h3>Top 10 Recommended Movies (Unrated)</h3>
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
    if (typeof tf === 'undefined') {
        document.getElementById('status').textContent = 'Error: TensorFlow.js not loaded. Check network connection.';
        return;
    }
    // Prioritize WebGL for performance
    tf.setBackend('webgl').then(() => {
        app = new MovieLensApp();
    }).catch(e => {
        console.warn("WebGL backend failed, falling back to CPU. Performance may be degraded.", e);
        tf.setBackend('cpu');
        app = new MovieLensApp();
    });
});

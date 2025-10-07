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
            // FIX: Ensure correct relative paths. The files MUST be in a 'data/' subdirectory.
            const [interactionsResponse, itemsResponse] = await Promise.all([
                fetch('data/u.data'), // Ensure this path is correct
                fetch('data/u.item')  // Ensure this path is correct
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
            this.updateStatus(`**ERROR:** Could not load data. Ensure 1) You are running a local server (not file://), and 2) **data/u.data** and **data/u.item** exist.`);
            this.setControls(false, false);
        }
    }

    // ... (rest of the class methods: train, plotLoss, visualizeEmbeddings, test, renderResults, updateStatus)
    // ... (The rest of the methods remain as previously updated)
    
    async train() { /* ... */ }
    plotLoss(history) { /* ... */ }
    async visualizeEmbeddings() { /* ... */ }
    async test() { /* ... */ }
    renderResults(userId, topRated, recommendations) { /* ... */ }
    updateStatus(message) { /* ... */ document.getElementById('status').textContent = message; }
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

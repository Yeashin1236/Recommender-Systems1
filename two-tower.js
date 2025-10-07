// two-tower.js

/**
 * Implements the Two-Tower Retrieval Model for Collaborative Filtering.
 */
class TwoTowerModel {
    constructor(numUsers, numItems, embeddingDim, learningRate = 0.001) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.embeddingDim = embeddingDim;
        
        // Initialize embedding tables with small random values
        this.userEmbeddings = tf.variable(
            tf.randomNormal([numUsers, embeddingDim], 0, 0.05), 
            true, 
            'user_embeddings'
        );
        
        this.itemEmbeddings = tf.variable(
            tf.randomNormal([numItems, embeddingDim], 0, 0.05), 
            true, 
            'item_embeddings'
        );
        
        this.optimizer = tf.train.adam(learningRate);
    }
    
    userForward(userIndices) {
        return tf.gather(this.userEmbeddings, userIndices);
    }
    
    itemForward(itemIndices) {
        return tf.gather(this.itemEmbeddings, itemIndices);
    }
    
    // Implements the In-Batch Sampled Softmax Loss
    async trainStep(userIndices, itemIndices) {
        const lossTensor = this.optimizer.minimize(() => {
            return tf.tidy(() => {
                // 1. Get embeddings for the batch
                const userEmbs = this.userForward(tf.tensor1d(userIndices, 'int32')); // [B, D]
                const itemEmbs = this.itemForward(tf.tensor1d(itemIndices, 'int32')); // [B, D]
                
                // 2. Compute Logits Matrix: L = U * I_T+
                // Result: [B, B] matrix. Diagonal L[i, i] is positive score.
                const logits = tf.matMul(userEmbs, itemEmbs, false, true);
                
                // 3. Create Labels: Diagonal is 1 (positives)
                const batchSize = userIndices.length;
                const labels = tf.oneHot(
                    tf.range(0, batchSize, 1, 'int32'), 
                    batchSize
                );
                
                // 4. Compute Softmax Cross Entropy Loss
                const loss = tf.losses.softmaxCrossEntropy(labels, logits);
                return loss;
            });
        }, true, [this.userEmbeddings, this.itemEmbeddings]);

        const lossValue = await lossTensor.data();
        lossTensor.dispose();
        return lossValue[0];
    }
    
    getUserEmbedding(userIndex) {
        return tf.tidy(() => {
            return this.userForward([userIndex]).squeeze(); // [D]
        });
    }
    
    /**
     * Computes the score for a user embedding against all item embeddings.
     */
    async getScoresForAllItems(userEmbedding) {
        return await tf.tidy(() => {
            const uEmb = userEmbedding.expandDims(1); // [D, 1]
            // Scores = [N_items, D] * [D, 1] -> [N_items, 1]
            const scoresTensor = tf.matMul(this.itemEmbeddings, uEmb); 
            
            return scoresTensor.squeeze().dataSync(); // [N_items] array
        });
    }
    
    getItemEmbeddings() {
        return this.itemEmbeddings;
    }
    
    /**
     * Computes a 2D PCA approximation for a sample of item embeddings.
     */
    async computePCAProjection(numSamples = 1000) {
        return await tf.tidy(async () => {
            let itemEmbs = this.itemEmbeddings;
            let sampleIndices = tf.randomUniform([numSamples], 0, this.numItems, 'int32');
            let sampleEmbs = tf.gather(itemEmbs, sampleIndices);

            const mean = sampleEmbs.mean(0);
            const centered = sampleEmbs.sub(mean);

            // Compute SVD
            const { V } = tf.linalg.svd(centered, true);
            
            // Get the first two principal components
            const principalComponents = V.slice([0, 0], [this.embeddingDim, 2]);

            // Project the embeddings
            const projected = centered.matMul(principalComponents);
            
            return { 
                projection: await projected.array(), 
                sampleIndices: await sampleIndices.array() 
            };
        });
    }
}

// two-tower.js

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
        // userIndices: [B]
        // Output: [B, D]
        return tf.gather(this.userEmbeddings, userIndices);
    }
    
    itemForward(itemIndices) {
        // itemIndices: [B]
        // Output: [B, D]
        return tf.gather(this.itemEmbeddings, itemIndices);
    }
    
    // Implements the In-Batch Sampled Softmax Loss
    async trainStep(userIndices, itemIndices) {
        // Use optimizer.minimize for safe gradient computation and tensor cleanup
        const lossTensor = this.optimizer.minimize(() => {
            return tf.tidy(() => {
                // 1. Get embeddings for the batch
                const userEmbs = this.userForward(tf.tensor1d(userIndices, 'int32')); // [B, D]
                const itemEmbs = this.itemForward(tf.tensor1d(itemIndices, 'int32')); // [B, D]
                
                // 2. Positive Scores (B x D) * (D x B) = B x B
                // We only care about the diagonal (B x 1)
                const positiveScores = tf.sum(tf.mul(userEmbs, itemEmbs), 1); // [B]

                // 3. Negative Scores (In-batch sampling)
                // (B x D) * (D x B) = B x B. All pair-wise scores within the batch.
                // Corrected: Use transposeB=true for A * B_transpose
                const negativeScores = tf.matMul(userEmbs, itemEmbs, false, true); // [B, B]
                
                // 4. Sampled Softmax Loss (The objective is to maximize positiveScore and minimize all others)
                // Formula: -log(exp(posScore) / sum(exp(allScores)))
                // = -posScore + log(sum(exp(allScores))) -> this is log-sum-exp
                
                // Reshape positiveScores to [B, 1] for concat
                const positiveScoresReshaped = positiveScores.expandDims(1);

                // Concatenate the positive scores and negative scores
                // This gives a [B, B+1] tensor of scores for the positive item and B negative items
                const allScores = tf.concat([positiveScoresReshaped, negativeScores], 1); // [B, B+1]

                // The true index for the positive score in each row is 0
                const labels = tf.zeros([allScores.shape[0]], 'int32'); // [B]

                // Compute the Sparse Softmax Cross-Entropy Loss
                const loss = tf.losses.sparseSoftmaxCrossEntropy(labels, allScores);
                
                return loss;
            });
        }, [this.userEmbeddings, this.itemEmbeddings]);

        // This minimizes loss, applying gradients to this.userEmbeddings and this.itemEmbeddings.
        // The lossTensor is what we want to return for logging.
        return lossTensor;
    }
    
    /**
      * Computes the score (dot product) between a user embedding and all item embeddings.
      * @param {tf.Tensor1D} userEmbedding - A tensor of shape [D].
      * @returns {Promise<Float32Array>} An array of scores, length N_items.
      */
    async scoreAllItems(userEmbedding) {
        return await tf.tidy(() => {
            // userEmbedding: [D] -> uEmb: [D, 1] (Transposing for matrix multiplication)
            const uEmb = userEmbedding.expandDims(1); 
            
            // Scores = [N_items, D] * [D, 1] -> [N_items, 1]
            // tf.matMul(A, B) computes A * B
            const scoresTensor = tf.matMul(this.itemEmbeddings, uEmb); 
            
            // Squeeze to [N_items] and return as a standard JS array
            return scoresTensor.squeeze().dataSync(); 
        });
    }
    
    /**
      * Computes a 2D PCA approximation for a sample of item embeddings.
      */
    async computePCAProjection(numSamples = 1000) {
        return await tf.tidy(async () => {
            let itemEmbs = this.itemEmbeddings;
            let numAvailableItems = this.numItems;
            
            // Adjust sample size if there aren't enough items
            if (numAvailableItems < numSamples) {
                numSamples = numAvailableItems;
            }

            // Generate random indices to sample
            let sampleIndices = tf.randomUniform([numSamples], 0, numAvailableItems, 'int32');
            let sampleEmbs = tf.gather(itemEmbs, sampleIndices);

            // Center the data (crucial for correct PCA)
            const mean = sampleEmbs.mean(0); // [D]
            const centered = sampleEmbs.sub(mean); // [N_sample, D]

            // Compute SVD: centered = U * S * V^T
            // We use the full SVD which returns {u, v, s} where v is the right singular vectors (V in the formula).
            const { v } = tf.linalg.svd(centered); // v is [D, D] (the V matrix)
            
            // Get the first two principal components (the first two columns of V)
            // slice([startRow, startCol], [height, width])
            const principalComponents = v.slice([0, 0], [this.embeddingDim, 2]); // [D, 2]

            // Project the embeddings: centered * principalComponents = [N_sample, D] * [D, 2] -> [N_sample, 2]
            const projected = centered.matMul(principalComponents);
            
            return { 
                projection: await projected.array(), // [[x1, y1], [x2, y2], ...]
                sampleIndices: await sampleIndices.data() // indices corresponding to the projection points
            };
        });
    }
}

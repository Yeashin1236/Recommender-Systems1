// two-tower.js
// TwoTowerModel: simplified, memory-aware, and optimized PCA via covariance SVD.

class TwoTowerModel {
  /**
   * @param {number} numUsers
   * @param {number} numItems
   * @param {number} embeddingDim
   * @param {number} learningRate
   */
  constructor(numUsers, numItems, embeddingDim = 25, learningRate = 0.01) {
    this.numUsers = numUsers;
    this.numItems = numItems;
    this.embeddingDim = embeddingDim;

    // Initialize variables
    this.userEmbeddings = tf.variable(tf.randomNormal([numUsers, embeddingDim], 0, 0.05), true, 'user_embeddings');
    this.itemEmbeddings = tf.variable(tf.randomNormal([numItems, embeddingDim], 0, 0.05), true, 'item_embeddings');

    this.optimizer = tf.train.adam(learningRate);
  }

  userForward(userIndicesTensor) {
    // userIndicesTensor: tf.Tensor1D (int32) shape [B]
    return tf.gather(this.userEmbeddings, userIndicesTensor);
  }

  itemForward(itemIndicesTensor) {
    return tf.gather(this.itemEmbeddings, itemIndicesTensor);
  }

  /**
   * Train step using in-batch negatives and sparseSoftmaxCrossEntropy.
   * Accepts arrays of indices.
   * Returns scalar loss (number).
   */
  async trainStep(userIndicesArray, itemIndicesArray) {
    const userTensor = tf.tensor1d(userIndicesArray, 'int32');
    const itemTensor = tf.tensor1d(itemIndicesArray, 'int32');

    // Use optimizer.minimize with returnCost = true for safe gradient handling and automatic disposal.
    const lossTensor = this.optimizer.minimize(() => {
      return tf.tidy(() => {
        const userEmbs = this.userForward(userTensor); // [B, D]
        const itemEmbs = this.itemForward(itemTensor); // [B, D]

        // Positive scores: dot per row -> [B]
        const posScores = tf.sum(tf.mul(userEmbs, itemEmbs), 1); // [B]

        // All pairwise scores: [B, B] = userEmbs * itemEmbs^T
        const allPairwise = tf.matMul(userEmbs, itemEmbs, false, true); // [B, B]

        // Construct allScores: [B, B+1] with positive score in column 0
        const posScoresCol = posScores.expandDims(1); // [B,1]
        const allScores = tf.concat([posScoresCol, allPairwise], 1); // [B, B+1]

        // Labels: positive at index 0
        const labels = tf.zeros([allScores.shape[0]], 'int32'); // [B]

        // Loss: sparse softmax cross entropy
        const loss = tf.losses.sparseSoftmaxCrossEntropy(labels, allScores);
        // Return scalar loss
        return loss;
      });
    }, /* returnCost */ true, /* varList */ [this.userEmbeddings, this.itemEmbeddings]);

    // Loss tensor returned; get number and dispose
    const lossVal = (await lossTensor.data())[0];
    lossTensor.dispose();
    userTensor.dispose();
    itemTensor.dispose();

    return lossVal;
  }

  /**
   * Score all items for a single user embedding tensor (1-D).
   * Returns Float32Array of scores length numItems.
   */
  async scoreAllItems(userEmbeddingTensor) {
    // Expects userEmbeddingTensor shape [D]
    return tf.tidy(() => {
      const u = userEmbeddingTensor.reshape([this.embeddingDim, 1]); // [D,1]
      // itemEmbeddings: [N, D] -> we want [N,1] = itemEmbeddings * u
      const scores = tf.matMul(this.itemEmbeddings, u); // [N,1]
      const squeezed = scores.squeeze(); // [N]
      return squeezed.dataSync(); // returns Float32Array synchronously
    });
  }

  /**
   * PCA projection improved:
   * - Sample up to numSamples item embeddings
   * - Compute centered embeddings and D x D covariance matrix
   * - SVD on covariance (small matrix) -> principal components
   * - Project centered embeddings onto top-2 components
   *
   * Returns { projection: Array<[x,y]>, sampleIndices: Uint32Array }
   */
  async computePCAProjection(numSamples = 1000) {
    // We'll do heavy ops inside tf.tidy
    const result = await tf.tidy(() => {
      const total = this.numItems;
      const nSamples = Math.min(numSamples, total);

      // Sample random indices (int32)
      const sampleIdx = tf.randomUniform([nSamples], 0, total, 'int32');
      const sampleEmbs = tf.gather(this.itemEmbeddings, sampleIdx); // [n, D]

      // Compute mean and center
      const mean = sampleEmbs.mean(0); // [D]
      const centered = sampleEmbs.sub(mean); // [n, D]

      // Compute covariance (D x D) = centered^T * centered / (n-1)
      const centeredT = centered.transpose(); // [D, n]
      const cov = centeredT.matMul(centered).div(tf.scalar(nSamples - 1, 'float32')); // [D, D]

      // SVD on covariance (small D x D)
      const { v } = tf.linalg.svd(cov); // v: [D, D] (right singular vectors)
      // principal components: first two columns of v -> [D,2]
      const pcs = v.slice([0, 0], [this.embeddingDim, 2]); // [D,2]

      // Project centered embeddings -> [n, 2]
      const proj = centered.matMul(pcs); // [n, 2]

      return {
        projection: proj.arraySync(), // [[x,y], ...]
        sampleIndices: sampleIdx.dataSync() // Uint32Array
      };
    });

    return result;
  }

  // Dispose variables when done
  dispose() {
    if (this.userEmbeddings) this.userEmbeddings.dispose();
    if (this.itemEmbeddings) this.itemEmbeddings.dispose();
  }
}

// two-tower.js
// TwoTowerModel supports two modes:
//  - 'shallow' : simple embedding lookup and dot-product scoring
//  - 'deep'    : embedding lookup followed by small MLP towers for user & item

class TwoTowerModel {
  /**
   * @param {number} numUsers
   * @param {number} numItems
   * @param {number} embeddingDim
   * @param {number} learningRate
   * @param {'shallow'|'deep'} mode
   * @param {Object} options - { towerHidden: [units,...], towerOutputDim }
   */
  constructor(numUsers, numItems, embeddingDim = 25, learningRate = 0.01, mode = 'shallow', options = {}) {
    this.numUsers = numUsers;
    this.numItems = numItems;
    this.embeddingDim = embeddingDim;
    this.mode = mode;
    this.learningRate = learningRate;

    this.towerHidden = options.towerHidden || [64];
    this.towerOutputDim = options.towerOutputDim || embeddingDim;

    // Embedding tables
    this.userEmbeddings = tf.variable(tf.randomNormal([numUsers, embeddingDim], 0, 0.05), true, 'user_embeddings');
    this.itemEmbeddings = tf.variable(tf.randomNormal([numItems, embeddingDim], 0, 0.05), true, 'item_embeddings');

    // Optionally build towers
    if (this.mode === 'deep') {
      // user tower
      this.userTower = tf.sequential();
      this.userTower.add(tf.layers.dense({ units: this.towerHidden[0], activation: 'relu', inputShape: [embeddingDim] }));
      for (let i = 1; i < this.towerHidden.length; i++) {
        this.userTower.add(tf.layers.dense({ units: this.towerHidden[i], activation: 'relu' }));
      }
      this.userTower.add(tf.layers.dense({ units: this.towerOutputDim, activation: 'linear' }));

      // item tower
      this.itemTower = tf.sequential();
      this.itemTower.add(tf.layers.dense({ units: this.towerHidden[0], activation: 'relu', inputShape: [embeddingDim] }));
      for (let i = 1; i < this.towerHidden.length; i++) {
        this.itemTower.add(tf.layers.dense({ units: this.towerHidden[i], activation: 'relu' }));
      }
      this.itemTower.add(tf.layers.dense({ units: this.towerOutputDim, activation: 'linear' }));
    }

    this.optimizer = tf.train.adam(this.learningRate);
  }

  userForward(userIdxTensor) {
    // userIdxTensor: int32 1-D
    const raw = tf.gather(this.userEmbeddings, userIdxTensor); // [B,D]
    if (this.mode === 'deep') {
      // apply tower -> returns [B, towerOutputDim]
      return this.userTower.apply(raw);
    }
    return raw;
  }

  itemForward(itemIdxTensor) {
    const raw = tf.gather(this.itemEmbeddings, itemIdxTensor);
    if (this.mode === 'deep') {
      return this.itemTower.apply(raw);
    }
    return raw;
  }

  /**
   * Train one step (batch arrays of indices).
   * Returns a numeric loss value.
   */
  async trainStep(userIndicesArray, itemIndicesArray) {
    const userTensor = tf.tensor1d(userIndicesArray, 'int32');
    const itemTensor = tf.tensor1d(itemIndicesArray, 'int32');

    // minimize and return cost
    const lossTensor = this.optimizer.minimize(() => {
      return tf.tidy(() => {
        const userEmbs = this.userForward(userTensor); // [B, D']
        const itemEmbs = this.itemForward(itemTensor); // [B, D']

        // positive scores per row: dot product
        const pos = tf.sum(tf.mul(userEmbs, itemEmbs), 1); // [B]

        // all pairwise scores: [B, B]
        const allPairwise = tf.matMul(userEmbs, itemEmbs, false, true); // [B,B]

        // place positive score in column 0
        const posCol = pos.expandDims(1); // [B,1]
        const allScores = tf.concat([posCol, allPairwise], 1); // [B, B+1]

        const labels = tf.zeros([allScores.shape[0]], 'int32'); // [B]

        const loss = tf.losses.sparseSoftmaxCrossEntropy(labels, allScores);
        return loss;
      });
    }, /* returnCost */ true);

    const v = (await lossTensor.data())[0];
    lossTensor.dispose();
    userTensor.dispose();
    itemTensor.dispose();
    return v;
  }

  /**
   * Score all items for one user embedding tensor (1-D tensor).
   * If deep mode, applies item tower to the whole item embedding table.
   * Returns Float32Array of length numItems.
   */
  async scoreAllItems(userEmbeddingTensor) {
    return tf.tidy(() => {
      // userEmbeddingTensor [D'] or [D] depending on mode; ensure column vector
      const u = userEmbeddingTensor.reshape([userEmbeddingTensor.shape[0], 1]); // [D,1] only if passed final vector; but ensure it's [D'] or [D]
      // For our pipeline, userEmbeddingTensor is [D'] 1-D so reshape to [D',1].
      // Need item vectors as [N, D']
      let itemVecs;
      if (this.mode === 'deep') {
        // apply tower to all item embeddings
        itemVecs = this.itemTower.apply(this.itemEmbeddings); // [N, D']
      } else {
        itemVecs = this.itemEmbeddings; // [N, D]
      }
      // matMul: [N, D'] * [D', 1] => [N, 1]
      const scores = tf.matMul(itemVecs, u);
      return scores.squeeze().dataSync();
    });
  }

  /**
   * PCA projection on a sample of item vectors.
   * If deep mode, applies item tower to sampled embeddings before PCA.
   * Returns { projection: Array<[x,y]>, sampleIndices: Uint32Array }
   */
  async computePCAProjection(numSamples = 1000) {
    return tf.tidy(() => {
      const total = this.numItems;
      const n = Math.min(numSamples, total);
      const sampleIdx = tf.randomUniform([n], 0, total, 'int32');
      const sampleEmbs = tf.gather(this.itemEmbeddings, sampleIdx); // [n, D]

      let vectors;
      if (this.mode === 'deep') {
        vectors = this.itemTower.apply(sampleEmbs); // [n, D']
      } else {
        vectors = sampleEmbs; // [n, D]
      }

      const mean = vectors.mean(0); // [D']
      const centered = vectors.sub(mean); // [n, D']

      // Covariance: D' x D' = centered^T * centered / (n-1)
      const ct = centered.transpose(); // [D', n]
      const cov = ct.matMul(centered).div(tf.scalar(n - 1, 'float32')); // [D', D']

      const { v } = tf.linalg.svd(cov); // v: [D', D']
      const pcs = v.slice([0, 0], [v.shape[0], 2]); // [D',2]
      const proj = centered.matMul(pcs); // [n,2]

      return {
        projection: proj.arraySync(),
        sampleIndices: sampleIdx.dataSync()
      };
    });
  }

  // Dispose variables and any sub-model weights
  dispose() {
    try {
      if (this.userEmbeddings) this.userEmbeddings.dispose();
      if (this.itemEmbeddings) this.itemEmbeddings.dispose();
      if (this.userTower) this.userTower.getWeights().forEach(w => w.dispose());
      if (this.itemTower) this.itemTower.getWeights().forEach(w => w.dispose());
    } catch (e) {
      console.warn('Error disposing model resources', e);
    }
  }
}

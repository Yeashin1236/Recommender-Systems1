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
            // Note: Assuming TwoTowerModel has a userForward method that takes an index array and returns a tensor
            const userEmbTensor = this.model.userForward(tf.tensor1d([userIndex], 'int32')).squeeze();

            // 2. Get scores for all items - FIX: scoreItems -> scoreAllItems (or verify two-tower.js function name)
            // Assuming two-tower.js defines the method as 'scoreAllItems'
            const allItemScores = await this.model.scoreAllItems(userEmbTensor); 
            
            // Array of { itemId, score, itemIndex }
            const candidateScores = [];
            const ratedItemIds = new Set(userInteractions.map(i => i.itemId));

            // Convert tensor result to a standard JS array for sorting
            const allScoresArray = allItemScores.dataSync();

            for (let itemIndex = 0; itemIndex < allScoresArray.length; itemIndex++) {
                const score = allScoresArray[itemIndex];
                const itemId = this.reverseItemMap.get(itemIndex);

                // Filter out items the user has already rated
                if (!ratedItemIds.has(itemId)) {
                    candidateScores.push({ itemId, score, itemIndex });
                }
            }

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

This function calculates a "spread" score that quantifies the relationship between the compactness of classes and their separation in a multi-dimensional space. It achieves this by comparing the statistical properties of distances between points within the same class to the distances between points in different classes.

***

### 1. Distance Calculation and Partitioning 📏

The procedure begins with a set of $M$ points $\{\mathbf{p}_1, \dots, \mathbf{p}_M\}$ in an $N$-dimensional space, where each point has a class label.

* **Pairwise Distances**: First, the Euclidean distance, $d(\mathbf{p}_i, \mathbf{p}_j)$, is computed for every unique pair of points $(\mathbf{p}_i, \mathbf{p}_j)$ in the dataset.

* **Distance Partitioning**: These distances are then partitioned into two distinct sets based on the class labels of the corresponding points:
    * **Within-Class Distances**: The set of all distances $d(\mathbf{p}_i, \mathbf{p}_j)$ where point $\mathbf{p}_i$ and point $\mathbf{p}_j$ belong to the **same** class.
    * **Between-Class Distances**: The set of all distances $d(\mathbf{p}_i, \mathbf{p}_j)$ where point $\mathbf{p}_i$ and point $\mathbf{p}_j$ belong to **different** classes.

The distance from a point to itself (which is always zero) is excluded from both sets.

***

### 2. Statistical Analysis 📊

Next, the arithmetic **mean** and **standard deviation** are calculated for each of the two sets of distances.

* For the set of within-class distances, we compute:
    * $\mu_{\text{within}}$ (mean intra-class distance)
    * $\sigma_{\text{within}}$ (standard deviation of intra-class distances)

* For the set of between-class distances, we compute:
    * $\mu_{\text{between}}$ (mean inter-class distance)
    * $\sigma_{\text{between}}$ (standard deviation of inter-class distances)

***

### 3. Final Score Calculation 🏁

The final score is a combination of these four statistical values, scaled by two external factors, $f_h$ and $f_k$.

The unscaled score is defined as the product of the mean distances minus the product of the standard deviations. The final score is given by the formula:

$$\text{Final Score} = ((\mu_{\text{between}} \cdot \mu_{\text{within}}) - (\sigma_{\text{between}} \cdot \sigma_{\text{within}})) \cdot (1 - f_h) \cdot (1 - f_k)$$

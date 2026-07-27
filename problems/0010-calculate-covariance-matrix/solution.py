def calculate_covariance_matrix(
    vectors: list[list[float]]
) -> list[list[float]]:

    if not vectors:
        return []

    num_feat = len(vectors)
    num_obs = len(vectors[0])

    if num_obs < 2:
        raise ValueError("At least two observations are required.")

    for feature in vectors:
        if len(feature) != num_obs:
            raise ValueError("All features must have equal lengths.")

    means = []

    for feature in vectors:
        feature_mean = sum(feature) / num_obs
        means.append(feature_mean)

    covariance_matrix = []

    for i in range(num_feat):
        row = []

        for j in range(num_feat):
            covariance_sum = 0

            for k in range(num_obs):
                covariance_sum += (
                    (vectors[i][k] - means[i])
                    * (vectors[j][k] - means[j])
                )

            covariance = covariance_sum / (num_obs - 1)
            row.append(covariance)

        covariance_matrix.append(row)

    return covariance_matrix
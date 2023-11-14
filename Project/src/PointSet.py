import sys
from enum import Enum
from typing import List, Tuple

import numpy as np


class FeaturesTypes(Enum):
    """Enumerate possible features types"""

    BOOLEAN = 0
    CLASSES = 1
    REAL = 2


class PointSet:
    """A class representing set of training points.

    Attributes
    ----------
        types : List[FeaturesTypes]
            Each element of this list is the type of one of the
            features of each point
        features : np.array[float]
            2D array containing the features of the points. Each line
            corresponds to a point, each column to a feature.
        labels : np.array[bool]
            1D array containing the labels of the points.
    """

    def __init__(
        self,
        features: List[List[float]],
        labels: List[bool],
        types: List[FeaturesTypes],
    ):
        """
        Parameters
        ----------
        features : List[List[float]]
            The features of the points. Each sublist contained in the
            list represents a point, each of its elements is a feature
            of the point. All the sublists should have the same size as
            the `types` parameter, and the list itself should have the
            same size as the `labels` parameter.
        labels : List[bool]
            The labels of the points.
        types : List[FeaturesTypes]
            The types of the features of the points.
        """

        self.types = types
        self.features = np.array(features)
        self.labels = np.array(labels)

        self.best_cat_val = None  # best categorical value to split (best partition)
        self.best_real_val = None  # best real value to split (best partition interval)
        self.idx_split = None
        self.min_split_points = None

    def get_gini(self) -> float:
        """Computes the Gini score of the set of points

        Returns
        -------
        float
            The Gini score of the set of points
        """

        num_samples = len(self.labels)
        p_false = np.sum(self.labels == False) / num_samples
        p_true = np.sum(self.labels == True) / num_samples

        gini = 1 - (p_false**2 + p_true**2)

        return gini

    def get_best_threshold(self) -> float:
        if self.idx_split is None:
            raise Exception("No split has been computed yet")

        if self.types[self.idx_split] == FeaturesTypes.BOOLEAN:
            return None
        elif self.types[self.idx_split] == FeaturesTypes.CLASSES:
            return self.best_cat_val
        elif self.types[self.idx_split] == FeaturesTypes.REAL:
            return self.best_real_val

    def get_best_gain(self) -> Tuple[int, float]:
        """Compute the feature along which splitting provides the best gain

        Returns
        -------
        int
            The ID of the feature along which splitting the set provides the
            best Gini gain.
        float
            The best Gini gain achievable by splitting this set along one of
            its features.
        """

        best_gain = -1
        best_feature = -1

        for i in range(0, self.features.shape[1]):
            best_cat_val = -1
            best_real_val = -1
            valid_split = False

            gini_split = 0

            # if a feature has only one value for all the points, discard that feature
            if len(np.unique(self.features[:, i])) == 1:
                continue

            if (
                self.types[i] == FeaturesTypes.BOOLEAN
                or (
                    self.types[i] == FeaturesTypes.CLASSES
                    and len(np.unique(self.features[:, i])) == 2
                )
                or (
                    self.types[i] == FeaturesTypes.REAL
                    and len(np.unique(self.features[:, i])) == 2
                )
            ):
                for j in np.unique(self.features[:, i]):
                    mask = self.features[:, i] == j
                    new_ps = PointSet(
                        self.features[mask], self.labels[mask], self.types
                    )
                    n_child = len(new_ps.labels)
                    n = len(self.labels)

                    if self.min_split_points is not None:
                        if (
                            n_child < self.min_split_points
                            or (n - n_child) < self.min_split_points
                        ):
                            break

                    valid_split = True

                    gini_i = new_ps.get_gini()
                    gini_split += (n_child / n) * gini_i

                if self.types[i] == FeaturesTypes.CLASSES:
                    best_cat_val = np.unique(self.features[:, i])[0]
                elif self.types[i] == FeaturesTypes.REAL:
                    best_real_val = self.features[:, i].sum() / 2
            elif self.types[i] == FeaturesTypes.CLASSES:
                gini_split = float("inf")

                # go over all the possible partitions of the feature
                for j in np.unique(self.features[:, i]):
                    tmp_gini_split = 0  # gini of the current partition

                    mask = self.features[:, i] == j
                    new_ps_left = PointSet(
                        self.features[mask], self.labels[mask], self.types
                    )
                    new_ps_right = PointSet(
                        self.features[~mask], self.labels[~mask], self.types
                    )

                    n_child_left = len(new_ps_left.labels)
                    n_child_right = len(new_ps_right.labels)
                    n = len(self.labels)

                    if self.min_split_points is not None:
                        if (
                            n_child_left < self.min_split_points
                            or n_child_right < self.min_split_points
                        ):
                            continue

                    valid_split = True

                    gini_left = new_ps_left.get_gini()
                    gini_right = new_ps_right.get_gini()

                    tmp_gini_split += (n_child_left / n) * gini_left
                    tmp_gini_split += (n_child_right / n) * gini_right

                    if tmp_gini_split < gini_split:
                        gini_split = tmp_gini_split
                        best_cat_val = j
            elif self.types[i] == FeaturesTypes.REAL:
                gini_split = float("inf")

                confusion_matrix = np.zeros((2, 2))

                # go over all the possible partitions of the feature
                for j in np.unique(self.features[:, i])[:-1]:
                    tmp_gini_split = 0  # gini of the current partition

                    # if it's the first iteration, compute the confusion matrix
                    if np.sum(confusion_matrix) == 0:
                        mask = self.features[:, i] <= j
                        new_ps_left = PointSet(
                            self.features[mask], self.labels[mask], self.types
                        )
                        new_ps_right = PointSet(
                            self.features[~mask], self.labels[~mask], self.types
                        )

                        n_child_left = len(new_ps_left.labels)
                        n_child_right = len(new_ps_right.labels)
                        n = len(self.labels)

                        confusion_matrix[0, 0] = np.sum(new_ps_left.labels == True)
                        confusion_matrix[0, 1] = np.sum(new_ps_left.labels == False)
                        confusion_matrix[1, 0] = np.sum(new_ps_right.labels == True)
                        confusion_matrix[1, 1] = np.sum(new_ps_right.labels == False)
                    else:
                        labels = self.labels[self.features[:, i] == j]

                        for l in labels:
                            if l == False:
                                confusion_matrix[0, 1] += 1
                                confusion_matrix[1, 1] -= 1
                            else:
                                confusion_matrix[0, 0] += 1
                                confusion_matrix[1, 0] -= 1

                        n_child_left = np.sum(confusion_matrix[0])
                        n_child_right = np.sum(confusion_matrix[1])
                        n = len(self.labels)

                    if self.min_split_points is not None:
                        if (
                            n_child_left < self.min_split_points
                            or n_child_right < self.min_split_points
                        ):
                            continue

                    valid_split = True

                    gini_left = 1 - (
                        (confusion_matrix[0, 0] / n_child_left) ** 2
                        + (confusion_matrix[0, 1] / n_child_left) ** 2
                    )
                    gini_right = 1 - (
                        (confusion_matrix[1, 0] / n_child_right) ** 2
                        + (confusion_matrix[1, 1] / n_child_right) ** 2
                    )

                    tmp_gini_split += (n_child_left / n) * gini_left
                    tmp_gini_split += (n_child_right / n) * gini_right

                    if tmp_gini_split < gini_split:
                        gini_split = tmp_gini_split
                        mask = self.features[:, i] <= j

                        left = self.features[mask, i]
                        right = self.features[~mask, i]

                        best_real_val = (np.max(left) + np.min(right)) / 2
            if valid_split:
                gini_gain = self.get_gini() - gini_split

                if gini_gain > best_gain:
                    best_gain = gini_gain
                    best_feature = i
                    self.idx_split = best_feature
                    self.best_cat_val = best_cat_val
                    self.best_real_val = best_real_val

        if best_gain == -1 and best_feature == -1:
            return None, None

        return best_feature, best_gain

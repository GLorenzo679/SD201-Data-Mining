from typing import List

import numpy as np

from PointSet import FeaturesTypes, PointSet


class Tree:
    """A decision Tree

    Attributes
    ----------
        points : PointSet
            The training points of the tree
    """

    def __init__(
        self,
        features: List[List[float]],
        labels: List[bool],
        types: List[FeaturesTypes],
        h: int = 1,
        min_split_points: int = 1,
        beta: float = 0.0,
    ):
        """
        Parameters
        ----------
            labels : List[bool]
                The labels of the training points.
            features : List[List[float]]
                The features of the training points. Each sublist
                represents a single point. The sublists should have
                the same length as the `types` parameter, while the
                list itself should have the same length as the
                `labels` parameter.
            types : List[FeaturesTypes]
                The types of the features.
        """

        self.l_tree = None
        self.r_tree = None
        self.h = h

        self.assigned_label = None  # assigned label if we are in a leaf
        self.idx_split = None  # index of the feature to split on

        self.min_split_points = min_split_points

        self.num_updates = 0
        self.num_points = len(labels)
        self.beta = beta

        # we are in a leaf
        if len(features) == 0:
            self.assigned_label = True  # default class
            return

        # height is 0, max depth reached
        if h == 0:
            self.assigned_label = np.bincount(labels).argmax()
            return

        # all points have the same label, automatically a leaf
        if np.unique(labels).size == 1:
            self.assigned_label = labels[0]
            return

        self.points = PointSet(features, labels, types)
        self.points.min_split_points = self.min_split_points

        self.idx_split, _ = self.points.get_best_gain()

        # we are in a leaf
        if self.idx_split is None:
            self.assigned_label = np.bincount(labels).argmax()
            return

        if self.points.types[self.idx_split] == FeaturesTypes.BOOLEAN:
            mask = self.points.features[:, (self.idx_split)] == 1
        elif self.points.types[self.idx_split] == FeaturesTypes.CLASSES:
            mask = self.points.features[:, self.idx_split] == self.points.best_cat_val
        elif self.points.types[self.idx_split] == FeaturesTypes.REAL:
            mask = self.points.features[:, self.idx_split] < self.points.best_real_val

        left_points = (
            self.points.features[mask],
            self.points.labels[mask],
            self.points.types,
        )
        right_points = (
            self.points.features[~mask],
            self.points.labels[~mask],
            self.points.types,
        )

        self.l_tree = Tree(
            left_points[0],
            left_points[1],
            left_points[2],
            h - 1,
            min_split_points,
            beta,
        )
        self.r_tree = Tree(
            right_points[0],
            right_points[1],
            right_points[2],
            h - 1,
            min_split_points,
            beta,
        )

    def add_training_point(self, features: List[float], label: bool) -> None:
        """Add a training point to the tree

        Parameters
        ----------
            features : List[float]
                The features of the training point.
            label : bool
                The label of the training point.
        """

        self.points.add_point(features, label)
        self.num_updates += 1

        if self.num_updates >= self.beta * len(self.points.labels):
            self = Tree(
                self.points.features,
                self.points.labels,
                self.points.types,
                self.h,
                self.min_split_points,
                self.beta,
            )
        else:
            if self.assigned_label is None or self.idx_split is None or self.h == 0:
                return
            else:
                if self.points.types[self.idx_split] == FeaturesTypes.BOOLEAN:
                    if features[self.idx_split] == 1:
                        return self.l_tree.add_training_point(features)
                    else:
                        return self.r_tree.add_training_point(features)
                elif self.points.types[self.idx_split] == FeaturesTypes.REAL:
                    if features[self.idx_split] < self.points.best_real_val:
                        return self.l_tree.add_training_point(features)
                    else:
                        return self.r_tree.add_training_point(features)
                elif self.points.types[self.idx_split] == FeaturesTypes.CLASSES:
                    if features[self.idx_split] == self.points.best_cat_val:
                        return self.l_tree.add_training_point(features)
                    else:
                        return self.r_tree.add_training_point(features)

    def del_training_point(self, features: List[float], label: bool) -> None:
        """Delete a training point from the tree

        Parameters
        ----------
            features : List[float]
                The features of the training point.
            label : bool
                The label of the training point.
        """

        self.points.del_point(features, label)
        self.num_updates += 1

        if self.num_updates >= self.beta * len(self.points.labels):
            self = Tree(
                self.points.features,
                self.points.labels,
                self.points.types,
                self.h,
                self.min_split_points,
                self.beta,
            )
        else:
            if self.assigned_label is None or self.idx_split is None or self.h == 0:
                return
            else:
                if self.points.types[self.idx_split] == FeaturesTypes.BOOLEAN:
                    if features[self.idx_split] == 1:
                        return self.l_tree.del_training_point(features)
                    else:
                        return self.r_tree.del_training_point(features)
                elif self.points.types[self.idx_split] == FeaturesTypes.REAL:
                    if features[self.idx_split] < self.points.best_real_val:
                        return self.l_tree.del_training_point(features)
                    else:
                        return self.r_tree.del_training_point(features)
                elif self.points.types[self.idx_split] == FeaturesTypes.CLASSES:
                    if features[self.idx_split] == self.points.best_cat_val:
                        return self.l_tree.del_training_point(features)
                    else:
                        return self.r_tree.del_training_point(features)

    def decide(self, features: List[float]) -> bool:
        """Give the guessed label of the tree to an unlabeled point

        Parameters
        ----------
            features : List[float]
                The features of the unlabeled point.

        Returns
        -------
            bool
                The label of the unlabeled point,
                guessed by the Tree
        """

        # termination condition for the recursion
        if self.assigned_label is not None:
            return self.assigned_label

        if self.points.types[self.idx_split] == FeaturesTypes.BOOLEAN:
            if features[self.idx_split] == 1:
                return self.l_tree.decide(features)
            else:
                return self.r_tree.decide(features)
        elif self.points.types[self.idx_split] == FeaturesTypes.REAL:
            if features[self.idx_split] < self.points.best_real_val:
                return self.l_tree.decide(features)
            else:
                return self.r_tree.decide(features)
        elif self.points.types[self.idx_split] == FeaturesTypes.CLASSES:
            if features[self.idx_split] == self.points.best_cat_val:
                return self.l_tree.decide(features)
            else:
                return self.r_tree.decide(features)
